"""Synthetic power test of the output-text detector.

Take a burst key, split its replies at random into a reference and a current
window (a true no-change pair, the same kind the thresholds were calibrated on),
then change a fraction of the current replies in a known way and see whether the
detector alarms. Repeating that over many random splits gives the alarm rate
(power) at each fraction. At fraction 0 nothing is changed, so that row is the
false-alarm rate (about 1 in 500 by construction).

Two shifts, each matching a real concern:
  preamble: put "Sure! Here's the answer:" on its own line before the reply
  bullets:  put two "- point" lines after the reply

Window size: the burst windows hold 10 repeats per probe. The stored thresholds
are scaled to 12, so these windows are passed as 10/3 runs (3 repeats per run),
which makes scaled_threshold undo that scaling and give the burst-calibrated limit.

Run:  python -m experiments.power_text [n_trials]   (default 200; prints tables, writes nothing)
"""

import random
import sys
from multiprocessing import Pool

from experiments.calibrate_length import CATEGORIES, MODELS, burst_rows, split_burst
from src.detectors.output_dist import (
    compare_window_features,
    detect_windows,
    feature_ratios,
    is_beyond,
    known_thresholds,
)
from src.detectors.pass_rate import load_thresholds

PREAMBLE = "Sure! Here's the answer:\n"
N_BULLETS = 2
WINDOW_RUNS = 10 / 3  # 10 repeats per probe, 3 repeats per run
FRACTIONS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)


def add_preamble(text):
    return PREAMBLE + text


def add_bullets(text):
    return text.rstrip("\n") + "\n" + "\n".join(["- point"] * N_BULLETS)


SHIFTS = {"preamble": add_preamble, "bullets": add_bullets}


def inject(rows, shift, fraction, rng):
    """Copy of `rows` where round(fraction * n) randomly chosen replies are changed by `shift`.

    `shift` is a function text -> text. Rows with no reply text are never chosen.
    The input rows are not modified.
    """
    out = [dict(r) for r in rows]
    eligible = [i for i, r in enumerate(out) if r.get("raw_output") is not None]
    k = round(fraction * len(eligible))
    for i in rng.sample(eligible, k):
        out[i]["raw_output"] = shift(out[i]["raw_output"])
    return out


def run_trials(model, category, rows, shift, fraction, n_trials, seed, thresholds):
    """Alarm rate and per-feature "beyond its limit" rates over n_trials random splits.

    Returns {"power": share of trials with status alarm,
             "by_feature": {feature: share of trials where it was beyond its limit}}.
    """
    key = f"{model}|{category}"
    known = known_thresholds(thresholds, key)
    rng = random.Random(seed)
    alarms = 0
    beyond = {f: 0 for f in known}
    for _ in range(n_trials):
        window_a, window_b = split_burst(rows, rng)
        window_b = inject(window_b, SHIFTS[shift], fraction, rng)
        result = detect_windows(
            model, category, window_a, window_b, WINDOW_RUNS, WINDOW_RUNS, thresholds
        )
        alarms += result.status == "alarm"
        effects = compare_window_features(window_a, window_b)["effects"]
        for name, (ratio, _) in feature_ratios(
            effects, known, WINDOW_RUNS, WINDOW_RUNS
        ).items():
            beyond[name] += is_beyond(ratio)
    return {
        "power": alarms / n_trials,
        "by_feature": {f: n / n_trials for f, n in beyond.items()},
    }


def power_curve(model, category, shift, n_trials=200, seed=0, thresholds=None):
    """{fraction: run_trials(...)} for FRACTIONS. Thresholds default to thresholds.json."""
    thresholds = thresholds if thresholds is not None else load_thresholds()
    rows = burst_rows(model, category)
    return {
        fraction: run_trials(
            model, category, rows, shift, fraction, n_trials, seed, thresholds
        )
        for fraction in FRACTIONS
    }


def smallest_reliable_fraction(curve, level=0.8):
    """Smallest tested fraction from which power stays at or above `level`, or None."""
    fractions = sorted(curve)
    for i, fraction in enumerate(fractions):
        if fraction > 0 and all(curve[g]["power"] >= level for g in fractions[i:]):
            return fraction
    return None


def _job(args):
    model, category, shift, n_trials = args
    key = f"{model}|{category}"
    if not known_thresholds(load_thresholds(), key):
        return key, shift, None
    return key, shift, power_curve(model, category, shift, n_trials)


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    jobs = [
        (m, c, shift, n_trials) for shift in SHIFTS for m in MODELS for c in CATEGORIES
    ]
    with Pool(5) as pool:
        results = pool.map(_job, jobs)
    for shift in SHIFTS:
        print(f"\n=== {shift} ({n_trials} trials per cell; alarm rate %) ===")
        print(
            f"{'key':32}"
            + "".join(f"{f * 100:>6.0f}%" for f in FRACTIONS)
            + "   >=80% from   beyond at 100%"
        )
        for key, s, curve in results:
            if s != shift:
                continue
            if curve is None:
                print(f"{key:32}  no threshold")
                continue
            row = "".join(f"{curve[f]['power'] * 100:>7.1f}" for f in FRACTIONS)
            best = smallest_reliable_fraction(curve)
            moved = [n for n, v in curve[1.0]["by_feature"].items() if v >= 0.5]
            print(
                f"{key:32}{row}   {'-' if best is None else f'{best * 100:.0f}%':>10}   {', '.join(moved) or '-'}"
            )
