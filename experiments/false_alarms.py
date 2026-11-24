"""False-alarm rate of the pass-rate and length detectors on fresh no-change burst splits.

Each split is a stratified 10+10 burst split (experiments.calibrate_length.split_burst), pushed through
the real detect_windows with the stored thresholds, the way the detector runs in production. The
calibrated rate is about 1 in 500 (0.2%). A much higher rate on a key means ties at the limit: the
no-change effect often lands exactly on the 99.8th-percentile value, and the stored threshold is
rounded, which can turn a tie into an alarm (this happened for the text detector).

Burst windows hold 10 repeats and the thresholds are for 12, so the windows are passed as 10/3 runs
(3 repeats per run), which makes scaled_threshold give back the burst-calibrated limit.
Resamples for the p-value and interval are set to 1: they do not change the alarm.

Run:  python -m experiments.false_alarms [n_splits]   (default 3000; prints only)
"""

import random
import sys
from multiprocessing import Pool

from experiments.calibrate_length import CATEGORIES, MODELS, burst_rows, split_burst
from src.detectors import output_length, pass_rate

WINDOW_RUNS = 10 / 3
DETECTORS = {"pass_rate": pass_rate, "length": output_length}


def thresholds_for(name, thresholds):
    """The thresholds dict in the shape detector `name` reads, and its section for key lookup."""
    return thresholds, (thresholds if name == "pass_rate" else thresholds["length"])


def false_alarm_rate(name, model, category, rows, n_splits, seed, thresholds):
    """Share of no-change splits where detector `name` alarms through its real detect_windows."""
    detector = DETECTORS[name]
    rng = random.Random(seed)
    alarms = 0
    for _ in range(n_splits):
        a, b = split_burst(rows, rng)
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
    return alarms / n_splits


def _job(args):
    name, model, category, n_splits, thresholds = args
    _, section = thresholds_for(name, thresholds)
    key = f"{model}|{category}"
    if section["thresholds"].get(key) is None:
        return name, key, None
    rows = burst_rows(model, category)
    return (
        name,
        key,
        false_alarm_rate(name, model, category, rows, n_splits, 0, thresholds),
    )


if __name__ == "__main__":
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    thresholds = pass_rate.load_thresholds()
    jobs = [
        (name, m, c, n_splits, thresholds)
        for name in DETECTORS
        for m in MODELS
        for c in CATEGORIES
    ]
    with Pool(5) as pool:
        results = pool.map(_job, jobs)
    for name in DETECTORS:
        rates = {k: r for n, k, r in results if n == name and r is not None}
        print(
            f"\n=== {name}: false-alarm rate on {n_splits} no-change splits (target 0.2%) ==="
        )
        for key, rate in rates.items():
            flag = "  <== high" if rate > 0.01 else ""
            print(f"  {key:32}{rate * 100:>7.2f}%{flag}")
        print(
            f"  keys checked {len(rates)}, highest {max(rates.values()) * 100:.2f}%, mean {sum(rates.values()) / len(rates) * 100:.2f}%"
        )
