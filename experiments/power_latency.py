"""Synthetic power test of the latency detector on burst splits (the limits are the BURST ones).

Take a burst key, split its replies at random into two no-change windows, then make the current
window slower by a known percentage on a known share of probes, and see whether the detector
alarms. The final limits in thresholds.json include the real-run inflation factor and are far too
wide for burst data, so this uses the burst thresholds (thresholds["latency"]["burst_thresholds"]).
Burst windows hold 10 repeats, passed as 10/3 runs so the limit is the burst-calibrated one.

Second part, at the FINAL limits (real_power): the limits in thresholds.json include the real
run-to-run noise, so the question "what slowdown does it catch?" is answered on the real windows:
multiply the current window's latencies by 1 + s and count the share of the real later windows that
alarm. Real windows come from mpt.db, which is assumed to contain no real change, so the share at
s = 0 is the false-alarm rate and the share at s > 0 is power. The windows overlap and share one
reference window, so these shares are rough (11 mixed windows, 6 per slot).

Run:  python -m experiments.power_latency [n_trials]   (prints tables, writes nothing)
"""

import random
import sys

from experiments.calibrate_length import CATEGORIES, MODELS, burst_rows, split_burst
from experiments.latency_inflation import (
    LATER_STARTS,
    REFERENCE_RUNS,
    SLOT_WINDOW,
    WIDTH,
    slot_runs,
)
from src.db.connection import connect
from src.detectors.data import load_window
from src.detectors.output_latency import (
    compare_window_latencies,
    known_threshold,
    latency_limit,
)
from src.detectors.pass_rate import (
    load_thresholds,
    restrict_to_common_probes,
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
    return abs(effect) > latency_limit(limit12, WINDOW_RUNS, WINDOW_RUNS)


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


SLOWDOWNS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5)


def real_power(model, category, thresholds, conn, slot=None, slowdowns=SLOWDOWNS):
    """{slowdown: share of the real later windows that alarm}, or None if the key has no limit.

    Mixed (slot None): the reference runs 4-7 against each later 4-run window, the same windows the
    inflation factor is estimated on. Slot: that slot's first 2-run window against each later 2-run
    window. The limit is the final one for the key, scaled to the window sizes. The alarm rule is the
    detector's: |effect| > limit.
    """
    key = f"{model}|{category}" + (f"|{slot}" if slot else "")
    limit12 = known_threshold(thresholds, key)
    if limit12 is None:
        return None
    if slot is None:
        ref_runs = list(REFERENCE_RUNS)
        later = [list(range(a, a + WIDTH)) for a in LATER_STARTS]
        n = WIDTH
    else:
        runs = slot_runs(conn)[slot]
        ref_runs = runs[:SLOT_WINDOW]
        later = [
            runs[i : i + SLOT_WINDOW]
            for i in range(SLOT_WINDOW, len(runs) - SLOT_WINDOW + 1)
        ]
        n = SLOT_WINDOW
    limit = latency_limit(limit12, n, n)
    ref = load_window(model, category, ref_runs, conn)
    windows = [load_window(model, category, r, conn) for r in later]
    out = {}
    for s in slowdowns:
        alarms = 0
        for cur in windows:
            shifted = [{**r, "latency": r["latency"] * (1 + s)} for r in cur]
            a, b = restrict_to_common_probes(ref, shifted)
            effect = compare_window_latencies(a, b)["effect"]
            alarms += effect is not None and abs(effect) > limit
        out[s] = alarms / len(windows)
    return out


def smallest_caught(curve, level=0.8):
    """Smallest slowdown from which the share of alarming windows stays at or above `level`, or None."""
    shifts = sorted(curve)
    for i, s in enumerate(shifts):
        if s > 0 and all(curve[t] >= level for t in shifts[i:]):
            return s
    return None


def _fmt(curve):
    if curve is None:
        return "null"
    best = smallest_caught(curve)
    return ">150%" if best is None else f"{best * 100:.0f}%"


def final_limit_report():
    """Print, per model, the smallest slowdown each limit catches (>=80% of real windows alarm)."""
    th, conn = load_thresholds(), connect()
    print(
        "\nSmallest slowdown caught at the FINAL limits (>= 80% of the real windows alarm), category 'all':"
    )
    print(
        f"{'model':18}{'mixed limit':>12}{'mixed':>9}{'night limit':>13}{'night':>8}{'day limit':>11}{'day':>8}"
    )
    for m in MODELS:
        cells = []
        for slot in (None, "night", "day"):
            key = f"{m}|all" + (f"|{slot}" if slot else "")
            lim = th["latency"]["thresholds"].get(key)
            curve = real_power(m, "all", th, conn, slot)
            cells.append((lim, curve))
        print(
            f"{m:18}"
            + "".join(
                f"{'null' if lim is None else f'{lim:.1f}%':>{w1}}{_fmt(curve):>{w2}}"
                for (lim, curve), w1, w2 in zip(cells, (12, 13, 11), (9, 8, 8))
            )
        )
    print("\nBy category, mixed windows (limit % -> smallest slowdown caught):")
    print(f"{'model':18}" + "".join(f"{c:>20}" for c in CATEGORIES))
    for m in MODELS:
        row = ""
        for c in CATEGORIES:
            lim = th["latency"]["thresholds"].get(f"{m}|{c}")
            row += f"{('null' if lim is None else f'{lim:.0f}%') + ' -> ' + _fmt(real_power(m, c, th, conn)):>20}"
        print(f"{m:18}{row}")
    print("\nFalse-alarm share at 0% slowdown, all mixed + slot keys with a limit:")
    worst = 0.0
    for m in MODELS:
        for c in CATEGORIES:
            for slot in (None, "night", "day"):
                curve = real_power(m, c, th, conn, slot, slowdowns=(0.0,))
                if curve is not None:
                    worst = max(worst, curve[0.0])
    print(
        f"  highest share of real windows alarming with no slowdown: {worst * 100:.0f}%"
    )


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    final_limit_report()
    print("\n---- burst limits (not what runs in production) ----")
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
