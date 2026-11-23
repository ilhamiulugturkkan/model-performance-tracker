import math
import random
import statistics

from .data import load_window, model_provider
from .pass_rate import (
    _percentile,
    load_thresholds,
    restrict_to_common_probes,
    scaled_threshold,
)
from .result import AlertContext, DetectorResult


def window_out_len(window):
    # A dict where probe_id: list[lens]
    probe_lens = {}
    for row in window:
        probe_id = row["probe_id"]
        length = row["output_tokens"]
        if length is None:
            continue
        probe_lens.setdefault(probe_id, []).append(length)
    return probe_lens


def _probe_changes(ref_lens, curr_lens, min_mean_tokens):
    """Per-probe percent change of mean length, for probes present in both windows.

    `ref_lens` / `curr_lens` are {probe_id: list[lens]} (as from window_out_len).
    Probes whose reference mean is below `min_mean_tokens` are skipped.
    Returns ({probe_id: percent change}, skipped_count).
    """
    changes = {}
    skipped = 0
    for probe_id in ref_lens:
        if probe_id in curr_lens:
            ref_avg = sum(ref_lens[probe_id]) / len(ref_lens[probe_id])
            if ref_avg < min_mean_tokens:
                skipped += 1
                continue
            curr_avg = sum(curr_lens[probe_id]) / len(curr_lens[probe_id])
            changes[probe_id] = (curr_avg - ref_avg) / ref_avg * 100
    return changes, skipped


def compare_window_output_lengths(
    reference_window, current_window, min_mean_tokens=5.0, top_n=3
):
    """Compare per-probe mean output length between two windows.

    Probes whose reference mean is below `min_mean_tokens` are skipped (a mean
    of 0 would divide by zero, and tiny means give wild percentages).

    Returns a dict:
      changes:     {probe_id: percent change of the probe's mean length}
      effect:      mean of the per-probe changes, in percent (None if no usable probes)
      median:      median of the per-probe changes (descriptive; None if no usable probes)
      longer / shorter / same: how many probes got longer, shorter, or stayed (nearly) the same
      skipped:     probes dropped by the min_mean_tokens rule
      top_changes: the `top_n` largest changes by absolute size, as [(probe_id, percent)]
    """
    ref_lens = window_out_len(reference_window)
    curr_lens = window_out_len(current_window)
    changes, skipped = _probe_changes(ref_lens, curr_lens, min_mean_tokens)

    # The effect: the mean of the per-probe changes, in percent.
    # The median is kept as a descriptive extra.
    sorted_changes = sorted(changes.values())
    effect = statistics.fmean(sorted_changes) if sorted_changes else None
    median = statistics.median(sorted_changes) if sorted_changes else None

    # The counts: how many probes got longer, shorter, and stayed (nearly) the same.
    longer = sum(1 for v in changes.values() if v > 1e-9)
    shorter = sum(1 for v in changes.values() if v < -1e-9)
    same = sum(1 for v in changes.values() if abs(v) < 1e-9)

    # The biggest movers by absolute size; probe id breaks ties so output is stable.
    top_changes = sorted(changes.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:top_n]

    return {
        "changes": changes,
        "effect": effect,
        "median": median,
        "longer": longer,
        "shorter": shorter,
        "same": same,
        "skipped": skipped,
        "top_changes": top_changes,
    }


def _average_ranks(values):
    """Ranks of `values` (1 = smallest), ties get the average of their ranks.

    Values within 1e-9 of their neighbour in sorted order count as tied, so
    float noise doesn't break ties.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while (
            end + 1 < len(order) and values[order[end + 1]] - values[order[end]] < 1e-9
        ):
            end += 1
        avg_rank = (
            start + end
        ) / 2 + 1  # positions start..end hold ranks start+1..end+1
        for pos in range(start, end + 1):
            ranks[order[pos]] = avg_rank
        start = end + 1
    return ranks


def signed_rank_p_value(changes, n_perm=10000, seed=0):
    """Two-sided Wilcoxon signed-rank p-value by sign-flip permutation.

    `changes` is a list of per-probe changes. Zero changes are dropped, the rest
    are ranked by absolute size (ties share the average rank), and the statistic
    is the sum of ranks of positive changes minus the sum of ranks of negative
    ones. The null keeps the ranks and gives each a random +/- sign, `n_perm`
    times. p = (count + 1) / (n_perm + 1), where count is the number of random
    sums with abs(sum) >= abs(observed).

    Returns None if there are no nonzero changes.
    """
    nonzero = [c for c in changes if abs(c) >= 1e-9]
    if not nonzero:
        return None

    ranks = _average_ranks([abs(c) for c in nonzero])
    observed = sum(r if c > 0 else -r for r, c in zip(ranks, nonzero))

    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        total = sum(r if rng.random() < 0.5 else -r for r in ranks)
        if abs(total) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def bootstrap_ci_length(
    reference_window, current_window, n_boot=2000, seed=0, min_mean_tokens=5
):
    """95% percentile-bootstrap interval for the length effect, in percent.

    The effect is the mean per-probe percent change (as in
    compare_window_output_lengths, so the same skip rules for missing tokens and
    tiny means apply). Within each probe, the reference and current reply
    lengths are resampled with replacement separately, keeping each window's
    count for that probe. Probes themselves are not resampled: the question is
    whether these specific probes changed.

    Returns (low, high), or None if there are no usable probes. Note that when
    every repeat of every probe is identical the interval has zero width
    (e.g. (0.0, 0.0)): that is expected, not a bug, but callers phrasing a
    reason string like "CI [0.0, 0.0]" may want to word it differently.
    """
    ref_lens = window_out_len(reference_window)
    curr_lens = window_out_len(current_window)

    observed, _ = _probe_changes(ref_lens, curr_lens, min_mean_tokens)
    if not observed:
        return None

    # Only probes present in both windows take part; each is resampled per window.
    common = [p for p in ref_lens if p in curr_lens]

    rng = random.Random(seed)
    effects = []
    for _ in range(n_boot):
        boot_ref = {p: rng.choices(ref_lens[p], k=len(ref_lens[p])) for p in common}
        boot_cur = {p: rng.choices(curr_lens[p], k=len(curr_lens[p])) for p in common}
        changes, _ = _probe_changes(boot_ref, boot_cur, min_mean_tokens)
        if changes:  # a resample can push every probe under the tiny-mean cutoff
            effects.append(statistics.fmean(changes.values()))

    if not effects:
        return None
    effects.sort()
    return _percentile(effects, 2.5), _percentile(effects, 97.5)


def known_threshold(thresholds, key):
    """The 12-repeat length threshold for `key`, or None if it is not known.

    Not known = no "length" section, key missing, null (deterministic category),
    or anything that is not a positive finite number. None means: do not alert.
    """
    value = thresholds.get("length", {}).get("thresholds", {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


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
    """Output-length detector: rows in, DetectorResult out (no database).

    Alarms when |effect| (mean per-probe % change in mean output tokens) exceeds
    the noise threshold for this model and category, scaled to the window sizes.
    If that threshold is not known the detector does nothing ("no_threshold"):
    it never alarms on a key it has no calibration for.
    """
    key = f"{model}|{category}"

    def result(status, reason, threshold=None, effect=None, p=None, ci=None):
        return DetectorResult(
            detector="length",
            metric_unit="percent",
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
    reference_window, current_window = restrict_to_common_probes(
        reference_window, current_window
    )
    compared = compare_window_output_lengths(reference_window, current_window)
    effect = compared["effect"]
    if effect is None:
        return result(
            "no_data",
            "No probes with usable token counts appear in both windows "
            "(missing counts and tiny means are skipped).",
        )

    threshold12 = known_threshold(thresholds, key)
    if threshold12 is None:
        return result(
            "no_threshold",
            f"No length threshold for {key} (deterministic category, or never calibrated), "
            "so no alarm rule exists and nothing is checked.",
            effect=effect,
        )

    threshold = scaled_threshold(threshold12, n_ref_runs, n_cur_runs)
    p = signed_rank_p_value(list(compared["changes"].values()))
    ci = bootstrap_ci_length(reference_window, current_window)
    evidence = "p = " + (f"{p:.4f}" if p is not None else "n/a")
    if ci:
        evidence += f", 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}]%"
    movers = ", ".join(
        f"{pid} {change:+.1f}%" for pid, change in compared["top_changes"]
    )
    notes = ""
    if key in thresholds["length"].get("approximate", ()):
        notes += " Threshold is approximate (few probes vary in length)."
    if compared["skipped"]:
        notes += (
            f" {compared['skipped']} probe(s) skipped (reference mean under 5 tokens)."
        )

    alarm = abs(effect) > threshold
    return result(
        "alarm" if alarm else "ok",
        f"Output length changed {effect:+.1f}% on average, {'beyond' if alarm else 'within'} "
        f"{threshold:.1f}% ({n_cur_runs} runs). {evidence}. Largest changes: {movers}.{notes}",
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
