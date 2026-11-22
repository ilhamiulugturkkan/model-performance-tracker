import json
import math
import random
from pathlib import Path

from .data import load_window, model_provider
from .result import AlertContext, DetectorResult

THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")
REPEATS_PER_RUN = 3  # repeats per probe in one scheduled run


def window_pass_rate(window):
    # A dict where probe_id: list[scores]
    probe_scores = {}
    for row in window:
        probe_id = row["probe_id"]
        score = row["score"]
        probe_scores.setdefault(probe_id, []).append(score)
    probe_avg_scores = {
        probe_id: sum(scores) / len(scores) for probe_id, scores in probe_scores.items()
    }
    pass_rate = (
        sum(probe_avg_scores.values()) / len(probe_avg_scores)
        if probe_avg_scores
        else None
    )
    return pass_rate


def compare_windows(reference_window, current_window):
    ref_pass_rate = window_pass_rate(reference_window)
    current_pass_rate = window_pass_rate(current_window)
    if ref_pass_rate is None or current_pass_rate is None:
        return None
    pass_rate_diff = current_pass_rate - ref_pass_rate
    effect = (pass_rate_diff) * 100
    return ref_pass_rate, current_pass_rate, pass_rate_diff, effect


def permutation_p_value(reference_window, current_window, n_perm=10000, seed=0):
    """Two-sided permutation p-value for the difference in window pass rates."""
    # Only probes present in both windows can be compared. A probe seen in one window only
    # can't be shuffled, so it would add a fixed offset and de-centre the null distribution.
    common = {row["probe_id"] for row in reference_window} & {
        row["probe_id"] for row in current_window
    }
    if not common:
        return None
    reference_window = [row for row in reference_window if row["probe_id"] in common]
    current_window = [row for row in current_window if row["probe_id"] in common]

    observed = compare_windows(reference_window, current_window)
    if observed is None:
        return None
    observed_diff = observed[2]

    rng = random.Random(seed)
    # Stratify by probe: probes differ in difficulty
    strata = {}
    for row in reference_window:
        entry = strata.setdefault(row["probe_id"], [[], 0])
        entry[0].append(row["score"])
        entry[1] += 1
    for row in current_window:
        strata[row["probe_id"]][0].append(row["score"])
    strata = list(strata.values())

    extreme = 0
    for _ in range(n_perm):
        ref_means = []
        cur_means = []
        for scores, n_ref in strata:
            rng.shuffle(scores)
            # Every probe here is in both windows, so neither slice is empty.
            ref_means.append(sum(scores[:n_ref]) / n_ref)
            cur_means.append(sum(scores[n_ref:]) / (len(scores) - n_ref))
        # Same definition as window_pass_rate: mean of per-probe means.
        diff = sum(cur_means) / len(cur_means) - sum(ref_means) / len(ref_means)
        # Small epsilon guards against float noise when diffs are equal in theory.
        if abs(diff) >= abs(observed_diff) - 1e-12:
            extreme += 1

    # +1 on both sides: the observed split is itself one valid permutation, so p is never 0.
    return (extreme + 1) / (n_perm + 1)


def _percentile(sorted_values, q):
    # Linear interpolation between closest ranks (same as numpy's default), q in [0, 100].
    pos = (len(sorted_values) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(reference_window, current_window, n_boot=2000, seed=0):
    """95% percentile-bootstrap interval for the pass-rate change (current - reference), in points.

    Within each probe, the reference and current scores are resampled with replacement
    separately, keeping each window's count for that probe.
    Returns (low, high), or None if the windows share no probes.
    """
    ref_by_probe = {}
    for row in reference_window:
        ref_by_probe.setdefault(row["probe_id"], []).append(row["score"])
    cur_by_probe = {}
    for row in current_window:
        cur_by_probe.setdefault(row["probe_id"], []).append(row["score"])

    # Same rule as permutation_p_value: only probes present in both windows are compared.
    common = ref_by_probe.keys() & cur_by_probe.keys()
    if not common:
        return None
    pairs = [(ref_by_probe[p], cur_by_probe[p]) for p in common]

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        ref_means = []
        cur_means = []
        for ref_scores, cur_scores in pairs:
            ref_means.append(
                sum(rng.choices(ref_scores, k=len(ref_scores))) / len(ref_scores)
            )
            cur_means.append(
                sum(rng.choices(cur_scores, k=len(cur_scores))) / len(cur_scores)
            )
        diffs.append(
            (sum(cur_means) / len(cur_means) - sum(ref_means) / len(ref_means)) * 100
        )

    diffs.sort()
    return _percentile(diffs, 2.5), _percentile(diffs, 97.5)


def load_thresholds(path=THRESHOLDS_PATH):
    # get thresholds as a dict.
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scaled_threshold(threshold12, n_ref_runs, n_cur_runs):
    """Threshold for windows of n runs, given the one for 12 repeats per probe (= 4 runs) in each window.

    Equal windows of n runs: threshold12 * sqrt(12 / (3 * n)). The general form below scales by the
    noise of the difference, sqrt(1/r_ref + 1/r_cur), and gives the same number when n_ref == n_cur.
    """
    r_ref, r_cur = n_ref_runs * REPEATS_PER_RUN, n_cur_runs * REPEATS_PER_RUN
    return threshold12 * math.sqrt((1 / r_ref + 1 / r_cur) / (2 / 12))


def detect_windows(
    model,
    category,
    reference_window,
    current_window,
    n_ref_runs,
    n_cur_runs,
    thresholds,
    provider="unknown",
    window_runs=(),
):
    # Pure pass-rate detector: rows in, DetectorResult out (no database).

    # Alarms when |current - reference| (points) exceeds the scaled noise threshold
    key = f"{model}|{category}"

    def result(status, reason, threshold=None, effect=None, p=None, ci=None):
        return DetectorResult(
            detector="pass_rate",
            metric_unit="points",
            statistic=effect,
            p_value=p,
            effect_size=effect,
            ci_low=ci[0] if ci else None,
            ci_high=ci[1] if ci else None,
            reason=reason,
            status=status,
            context=AlertContext(
                provider=provider,
                model=model,
                category=category,
                window_runs=list(window_runs),
                threshold=threshold,
            ),
        )

    if n_ref_runs == 0 or n_cur_runs == 0:
        return result("no_data", "A window has no runs.")

    # The thresholds come from windows holding the same probes, so compare only probes seen in both.
    common = {r["probe_id"] for r in reference_window} & {
        r["probe_id"] for r in current_window
    }
    reference_window = [r for r in reference_window if r["probe_id"] in common]
    current_window = [r for r in current_window if r["probe_id"] in common]
    compared = compare_windows(reference_window, current_window)
    if compared is None:
        return result("no_data", "No scored probes appear in both windows.")
    effect = compared[3]

    # No threshold = the burst showed no noise (or this key was never measured). Not a power problem:
    # a drop is easy to see, there is just no alarm rule for it yet.
    threshold12 = thresholds["thresholds"].get(key)
    if threshold12 is None:
        return result(
            "no_threshold",
            f"No threshold for {key}: the burst showed no noise (or the key is missing), so no alarm rule exists yet.",
            effect=effect,
        )

    threshold = scaled_threshold(threshold12, n_ref_runs, n_cur_runs)
    p = permutation_p_value(reference_window, current_window)
    ci = bootstrap_ci(reference_window, current_window)
    evidence = f"p = {p:.4f}, 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}] points"
    approx = (
        " Threshold is approximate (few noisy probes)."
        if key in thresholds.get("approximate", ())
        else ""
    )
    alarm = abs(effect) > threshold
    return result(
        "alarm" if alarm else "ok",
        f"Pass rate changed {effect:+.1f} points, {'beyond' if alarm else 'within'} {threshold:.1f} "
        f"({n_cur_runs} runs). {evidence}.{approx}",
        threshold,
        effect,
        p,
        ci,
    )


def detect(model, category, reference_runs, current_runs, conn=None, thresholds=None):
    """Load both windows from the database and run detect_windows on them."""
    return detect_windows(
        model,
        category,
        load_window(model, category, reference_runs, conn),
        load_window(model, category, current_runs, conn),
        len(reference_runs),
        len(current_runs),
        thresholds if thresholds is not None else load_thresholds(),
        provider=model_provider(model, conn) or "unknown",
        window_runs=current_runs,
    )


if __name__ == "__main__":
    print(detect("claude-sonnet-5-5", "math", [1, 2, 3, 4], [5, 6, 7, 8]))
    print(detect("claude-sonnet-5-5", "all", [1, 2, 3, 4], [5, 6, 7, 8]).reason)
    print(detect("claude-sonnet-5-5", "coding", [1, 2, 3, 4], [5, 6, 7, 8]).status)
