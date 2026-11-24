"""Latency detector: did replies get slower or faster?

Per probe, the median latency in each window (robust to the occasional slow call:
the data has spikes of 5-30x the median). Per probe, the percent change of that
median between the windows. The effect is the mean of those changes over the probes
seen in both windows, as in the length detector.

An alarm means "slower or faster", NOT "the model changed". Provider load, time of
day, region or network can move latency with no change to the model at all.

Day and night are separate. Runs happen at about 03:00 and 15:00 UTC, and the GPT models are
9-23% slower in the 15:00 runs, every day. A window that mixes the two slots would average a
change in one slot with the other. So detect_windows can be asked for one slot ("night" or
"day"); it then compares only that slot's replies, against that slot's own limit
("model|category|slot" in thresholds.json). A row's slot comes from its UTC time_stamp.
"""

import math
import random
import statistics

from .data import load_window, model_provider
from .output_length import signed_rank_p_value
from .pass_rate import (
    _percentile,
    load_thresholds,
    restrict_to_common_probes,
    scaled_threshold,
)
from .result import AlertContext, DetectorResult

MIN_MEDIAN_SECONDS = 0.1  # the smallest probe median in the burst is about 0.6 s
SLOTS = ("night", "day")
NIGHT_END_HOUR = (
    9  # UTC hours 0-8 are "night" (the 03:00 runs), the rest "day" (the 15:00 runs)
)


def row_slot(row):
    """ "night" or "day", from the row's UTC time_stamp (ISO text, e.g. 2026-11-14T03:00:04+00:00)."""
    return "night" if int(row["time_stamp"][11:13]) < NIGHT_END_HOUR else "day"


def split_by_slot(window):
    """{"night": rows, "day": rows} for a window's rows."""
    out = {slot: [] for slot in SLOTS}
    for row in window:
        out[row_slot(row)].append(row)
    return out


def window_latency(window):
    """{probe_id: [latencies]}; a null latency is skipped."""
    probes = {}
    for row in window:
        latency = row["latency"]
        if latency is None:
            continue
        probes.setdefault(row["probe_id"], []).append(latency)
    return probes


def _probe_changes(ref, cur, min_median):
    """Per-probe percent change of the median latency, for probes present in both windows.

    `ref` / `cur` are {probe_id: list[latencies]} (as from window_latency). Probes whose
    reference median is below `min_median` seconds are skipped (tiny medians give wild
    percentages). Returns ({probe_id: percent change}, skipped_count).
    """
    changes = {}
    skipped = 0
    for probe_id in ref:
        if probe_id in cur:
            ref_med = statistics.median(ref[probe_id])
            if ref_med < min_median:
                skipped += 1
                continue
            changes[probe_id] = (
                (statistics.median(cur[probe_id]) - ref_med) / ref_med * 100
            )
    return changes, skipped


def compare_window_latencies(
    reference_window, current_window, min_median=MIN_MEDIAN_SECONDS, top_n=3
):
    """Compare per-probe median latency between two windows.

    Returns a dict:
      changes:     {probe_id: percent change of the probe's median latency}
      effect:      mean of the per-probe changes, in percent (None if no usable probes)
      median:      median of the per-probe changes (descriptive; None if no usable probes)
      slower / faster / same: how many probes got slower, faster, or stayed the same
      skipped:     probes dropped by the min_median rule
      top_changes: the `top_n` largest changes by absolute size, as [(probe_id, percent)]
    """
    ref = window_latency(reference_window)
    cur = window_latency(current_window)
    changes, skipped = _probe_changes(ref, cur, min_median)
    values = sorted(changes.values())
    return {
        "changes": changes,
        "effect": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "slower": sum(1 for v in values if v > 1e-9),
        "faster": sum(1 for v in values if v < -1e-9),
        "same": sum(1 for v in values if abs(v) < 1e-9),
        "skipped": skipped,
        "top_changes": sorted(changes.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[
            :top_n
        ],
    }


def bootstrap_ci_latency(
    reference_window, current_window, n_boot=2000, seed=0, min_median=MIN_MEDIAN_SECONDS
):
    """95% percentile-bootstrap interval for the latency effect, in percent.

    Same effect and skip rules as compare_window_latencies. Within each probe the
    reference and current latencies are resampled with replacement separately,
    keeping each window's count. Probes themselves are not resampled.
    Returns (low, high), or None if there are no usable probes.
    """
    ref = window_latency(reference_window)
    cur = window_latency(current_window)
    observed, _ = _probe_changes(ref, cur, min_median)
    if not observed:
        return None
    common = [p for p in ref if p in cur]

    rng = random.Random(seed)
    effects = []
    for _ in range(n_boot):
        boot_ref = {p: rng.choices(ref[p], k=len(ref[p])) for p in common}
        boot_cur = {p: rng.choices(cur[p], k=len(cur[p])) for p in common}
        changes, _ = _probe_changes(boot_ref, boot_cur, min_median)
        if changes:
            effects.append(statistics.fmean(changes.values()))
    if not effects:
        return None
    effects.sort()
    return _percentile(effects, 2.5), _percentile(effects, 97.5)


def latency_limit(threshold12, n_ref_runs, n_cur_runs):
    """The alarm limit for windows of n runs, given the 12-repeat (4-run) limit.

    Smaller windows are noisier, so the limit grows (the usual scaling). Larger windows do NOT get
    a smaller limit: real latency drifts like a random walk (windows 2 slot-runs apart spread about
    1.4 times more than windows 1 apart), so averaging more runs does not shrink the noise by the
    square root of the number of runs. The limit never goes below the 12-repeat one.
    """
    return max(scaled_threshold(threshold12, n_ref_runs, n_cur_runs), threshold12)


def known_threshold(thresholds, key):
    """The 12-repeat latency threshold for `key`, or None if it is not known.

    Not known = no "latency" section, key missing, null, or anything that is not a
    positive finite number. None means: do not alarm.
    """
    value = thresholds.get("latency", {}).get("thresholds", {}).get(key)
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
    slot=None,
):
    """Latency detector: rows in, DetectorResult out (no database).

    Alarms when |effect| (mean per-probe % change in median latency) exceeds the
    threshold for this model and category, scaled to the window sizes. A key with no
    threshold is never checked ("no_threshold"). An alarm means slower or faster, not
    that the model changed.

    slot: None compares all rows together (use windows with equal numbers of night and day
    runs, or the slot effect shows up as a change). "night" or "day" keeps only that slot's
    rows in both windows, uses the limit for "model|category|slot", and takes the window
    sizes from the number of distinct run_ids left in each window (n_ref_runs and n_cur_runs
    are then only used when the rows carry no run_id).
    """
    key = f"{model}|{category}"
    label = ""
    if slot is not None:
        if slot not in SLOTS:
            raise ValueError(f"slot must be one of {SLOTS} or None, got {slot!r}")
        key += f"|{slot}"
        label = f"[{slot}] "
        reference_window = split_by_slot(reference_window)[slot]
        current_window = split_by_slot(current_window)[slot]
        n_ref_runs = len({r["run_id"] for r in reference_window if "run_id" in r}) or (
            n_ref_runs if reference_window else 0
        )
        n_cur_runs = len({r["run_id"] for r in current_window if "run_id" in r}) or (
            n_cur_runs if current_window else 0
        )

    def result(status, reason, threshold=None, effect=None, p=None, ci=None):
        return DetectorResult(
            detector="latency",
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
        return result("no_data", f"{label}A window has no runs.")

    reference_window, current_window = restrict_to_common_probes(
        reference_window, current_window
    )
    compared = compare_window_latencies(reference_window, current_window)
    effect = compared["effect"]
    if effect is None:
        return result(
            "no_data",
            f"{label}No probes with usable latencies appear in both windows "
            "(missing latencies and tiny medians are skipped).",
        )

    threshold12 = known_threshold(thresholds, key)
    if threshold12 is None:
        in_file = key in thresholds.get("latency", {}).get("thresholds", {})
        why = (
            "calibrated, but a usable limit cannot be determined: latency did not vary in "
            "the burst, or real run-to-run noise is so large that the limit would be too wide to use"
            if in_file
            else "never calibrated"
        )
        return result(
            "no_threshold",
            f"{label}No latency threshold for {key} ({why}), so no alarm rule exists "
            "and nothing is checked.",
            effect=effect,
        )

    threshold = latency_limit(threshold12, n_ref_runs, n_cur_runs)
    p = signed_rank_p_value(list(compared["changes"].values()))
    ci = bootstrap_ci_latency(reference_window, current_window)
    evidence = "p = " + (f"{p:.4f}" if p is not None else "n/a")
    if ci:
        evidence += f", 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}]%"
    movers = ", ".join(
        f"{pid} {change:+.1f}%" for pid, change in compared["top_changes"]
    )
    notes = ""
    if key in thresholds["latency"].get("approximate", ()):
        notes += " Threshold is approximate."
    if compared["skipped"]:
        notes += (
            f" {compared['skipped']} probe(s) skipped (reference median under "
            f"{MIN_MEDIAN_SECONDS:g} s)."
        )

    alarm = abs(effect) > threshold
    direction = "slower" if effect > 0 else "faster"
    if alarm:
        head = (
            f"{label}Replies got {direction}: median latency changed {effect:+.1f}% on average, "
            f"beyond {threshold:.1f}% ({n_cur_runs} runs)."
        )
        caveat = (
            " This means slower or faster, not that the model changed: provider load, "
            "time of day or network can move latency with no model change."
        )
    else:
        head = (
            f"{label}Median latency changed {effect:+.1f}% on average, within "
            f"{threshold:.1f}% ({n_cur_runs} runs)."
        )
        caveat = ""
    return result(
        "alarm" if alarm else "ok",
        f"{head} {evidence}. Largest changes: {movers}.{notes}{caveat}",
        threshold,
        effect,
        p,
        ci,
    )


def detect(
    model, category, reference_runs, current_runs, conn=None, thresholds=None, slot=None
):
    """Load both windows from the database and run detect_windows on them (see its `slot`)."""
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
        slot=slot,
    )


def detect_slots(
    model, category, reference_runs, current_runs, conn=None, thresholds=None
):
    """{"night": result, "day": result}: the detector run separately on each slot's runs.

    Slot results are secondary evidence: they add up to 50 more tests on top of the mixed ones
    and their limits rest on very few real windows, so one slot alarming is a lead to look at,
    not an alert on its own until the many-tests control exists.
    """
    return {
        slot: detect(
            model, category, reference_runs, current_runs, conn, thresholds, slot=slot
        )
        for slot in SLOTS
    }
