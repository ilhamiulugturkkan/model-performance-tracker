"""Synthetic power test of the latency detector on burst splits (the limits are the BURST ones).

Take a burst key, split its replies at random into two no-change windows, then make the current
window slower by a known percentage on a known share of probes, and see whether the detector
alarms. The final limits in thresholds.json include the real-run inflation factor and are far too
wide for burst data, so this uses the burst thresholds (thresholds["latency"]["burst_thresholds"]).
Burst windows hold 10 repeats, passed as 10/3 runs so the limit is the burst-calibrated one.

Run:  python -m experiments.power_latency [n_trials]   (prints a table, writes nothing)
"""

import random
import sys

from experiments.calibrate_length import burst_rows, split_burst
from src.detectors.output_latency import (
    compare_window_latencies,
    known_threshold,
)
from src.detectors.pass_rate import (
    load_thresholds,
    restrict_to_common_probes,
    scaled_threshold,
)

WINDOW_RUNS = 10 / 3
SHIFTS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20)  # fractional slowdown


def burst_thresholds(thresholds=None):
    """Thresholds dict whose "latency" limits are the burst limits (the same shape the detector reads)."""
    thresholds = thresholds or load_thresholds()
    return {"latency": {"thresholds": thresholds["latency"]["burst_thresholds"]}}


def slow_down(rows, shift, probe_share, rng):
    """Copy of rows where the latencies of a random `probe_share` of the probes are scaled by 1 + shift."""
    probes = sorted({r["probe_id"] for r in rows})
    hit = set(rng.sample(probes, round(probe_share * len(probes))))
    return [
        {**r, "latency": r["latency"] * (1 + shift)}
        if r["probe_id"] in hit
        else dict(r)
        for r in rows
    ]


def alarms_fast(model, category, a, b, thresholds):
    """The detector's alarm rule without its p-value and bootstrap interval (those cost ~1 s a call).

    Same steps as detect_windows: common probes, effect, limit scaled to the window size,
    alarm when |effect| > limit. A test checks it against detect_windows.
    """
    a, b = restrict_to_common_probes(a, b)
    effect = compare_window_latencies(a, b)["effect"]
    limit12 = known_threshold(thresholds, f"{model}|{category}")
    if effect is None or limit12 is None:
        return False
    return abs(effect) > scaled_threshold(limit12, WINDOW_RUNS, WINDOW_RUNS)


def power(
    model,
    category,
    rows,
    shift,
    probe_share,
    n_trials,
    seed,
    thresholds,
    check=alarms_fast,
):
    """Share of random splits where the detector alarms after the slowdown."""
    rng = random.Random(seed)
    alarms = 0
    for _ in range(n_trials):
        a, b = split_burst(rows, rng)
        b = slow_down(b, shift, probe_share, rng)
        alarms += check(model, category, a, b, thresholds)
    return alarms / n_trials


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    th = burst_thresholds()
    keys = [
        (m, c)
        for m in (
            "claude-haiku-5-5",
            "claude-sonnet-5-5",
            "gpt-6-luna",
            "gpt-6-sol",
            "gpt-6.1-terra",
        )
        for c in ("all",)
    ]
    for share in (1.0, 0.25):
        print(
            f"\n=== slowdown on {share:.0%} of probes ({n_trials} trials per cell; alarm rate %) ==="
        )
        print(
            f"{'key':28}{'limit %':>8}" + "".join(f"{s * 100:>7.0f}%" for s in SHIFTS)
        )
        for m, c in keys:
            rows = burst_rows(m, c)
            lim = th["latency"]["thresholds"][f"{m}|{c}"]
            cells = "".join(
                f"{power(m, c, rows, s, share, n_trials, 0, th) * 100:>8.1f}"
                for s in SHIFTS
            )
            print(f"{m + '|' + c:28}{lim:>8.2f}{cells}")
