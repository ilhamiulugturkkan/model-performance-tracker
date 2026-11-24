"""Synthetic power test of the pass-rate and length detectors, on burst splits.

Same design as the text and latency power tests: split a burst key into two no-change windows, make the
current window worse in a known way on a share of its probes, and count how often the real detect_windows
alarms. Share 0 changes nothing, so that column is the false-alarm rate (about 1 in 500).

  pass rate:  on the chosen probes each passing call fails with probability (drop / that probe's pass
              rate in the window), so the probe's pass rate falls by `drop` points in expectation
              (by all of it if it is already below `drop`). The window-level drop is about share x drop.
  length:     on the chosen probes every reply's output_tokens grows by `change` percent. The
              window-level change is share x change.

Burst windows hold 10 repeats and the thresholds are for 12, so windows are passed as 10/3 runs, which
makes scaled_threshold give the burst-calibrated limit (the same trick as experiments.false_alarms).
Resamples for the p-value and interval are set to 1: they do not change the alarm.

Run:  python -m experiments.power_pass_length [n_trials]   (default 200; prints tables)
"""

import random
import sys
from multiprocessing import Pool

from experiments.calibrate_length import CATEGORIES, MODELS, burst_rows, split_burst
from src.detectors import output_length, pass_rate

WINDOW_RUNS = 10 / 3
DROPS = (3.0, 5.0, 8.0)  # pass-rate drop, points
CHANGES = (5.0, 10.0, 20.0)  # length change, percent
SHARES = (0.0, 0.1, 0.25, 0.5, 1.0)  # share of probes that get the shift
DETECTORS = {"pass_rate": pass_rate, "length": output_length}
SIZES = {"pass_rate": DROPS, "length": CHANGES}


def _choose_probes(rows, share, rng):
    probes = sorted({r["probe_id"] for r in rows})
    return set(rng.sample(probes, round(share * len(probes))))


def drop_pass_rate(rows, drop_points, share, rng):
    """Copy of rows where the chosen probes lose `drop_points` of pass rate in expectation."""
    chosen = _choose_probes(rows, share, rng)
    scores = {}
    for r in rows:
        scores.setdefault(r["probe_id"], []).append(r["score"])
    rate = {p: sum(v) / len(v) for p, v in scores.items()}
    out = []
    for r in rows:
        r = dict(r)
        if r["probe_id"] in chosen and r["score"] > 0:
            # chance that this passing call fails: drop / that probe's pass rate (at most 1)
            fail_p = min(1.0, drop_points / 100 / rate[r["probe_id"]])
            if rng.random() < fail_p:
                r["score"] = 0.0
        out.append(r)
    return out


def lengthen(rows, change_percent, share, rng):
    """Copy of rows where the chosen probes' output_tokens grow by `change_percent`."""
    chosen = _choose_probes(rows, share, rng)
    return [
        {**r, "output_tokens": r["output_tokens"] * (1 + change_percent / 100)}
        if r["probe_id"] in chosen and r["output_tokens"] is not None
        else dict(r)
        for r in rows
    ]


INJECT = {"pass_rate": drop_pass_rate, "length": lengthen}


def power(name, model, category, rows, size, share, n_trials, seed, thresholds):
    """Share of random splits where the detector alarms after the shift (real detect_windows)."""
    detector, inject = DETECTORS[name], INJECT[name]
    rng = random.Random(seed)
    alarms = 0
    for _ in range(n_trials):
        a, b = split_burst(rows, rng)
        b = inject(b, size, share, rng)
        result = detector.detect_windows(
            model,
            category,
            a,
            b,
            WINDOW_RUNS,
            WINDOW_RUNS,
            thresholds,
            n_perm=1,
            n_boot=1,
        )
        alarms += result.status == "alarm"
    return alarms / n_trials


def _limit(name, thresholds, key):
    section = thresholds if name == "pass_rate" else thresholds["length"]
    return section["thresholds"].get(key)


def _job(args):
    name, model, category, n_trials, thresholds = args
    key = f"{model}|{category}"
    if _limit(name, thresholds, key) is None:
        return name, key, None
    rows = burst_rows(model, category)
    grid = {}
    for size in SIZES[name]:
        for share in SHARES:
            if share == 0.0 and size != SIZES[name][0]:
                grid[(size, share)] = grid[
                    (SIZES[name][0], 0.0)
                ]  # nothing changed: same cell
                continue
            grid[(size, share)] = power(
                name, model, category, rows, size, share, n_trials, 0, thresholds
            )
    return name, key, grid


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    thresholds = pass_rate.load_thresholds()
    jobs = [
        (n, m, c, n_trials, thresholds)
        for n in DETECTORS
        for m in MODELS
        for c in CATEGORIES
    ]
    with Pool(5) as pool:
        results = pool.map(_job, jobs)
    unit = {"pass_rate": "points", "length": "%"}
    for name in DETECTORS:
        print(
            f"\n=== {name}: alarm rate %, {n_trials} trials per cell (columns: share of probes shifted) ==="
        )
        print(
            f"{'key':30}{'limit':>7}{'size':>7}"
            + "".join(f"{s * 100:>7.0f}%" for s in SHARES)
        )
        for n, key, grid in results:
            if n != name:
                continue
            if grid is None:
                print(f"{key:30}  no threshold")
                continue
            lim = _limit(name, thresholds, key)
            for i, size in enumerate(SIZES[name]):
                label = f"{key}" if i == 0 else ""
                limit = f"{lim:.2f}" if i == 0 else ""
                print(
                    f"{label:30}{limit:>7}{size:>5.0f}{'':>2}"
                    + "".join(f"{grid[(size, s)] * 100:>8.1f}" for s in SHARES)
                )
