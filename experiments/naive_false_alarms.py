"""False-alarm rate of the naive baseline ("alarm on a fixed drop in pass rate") when nothing changed.

The comparison number for the evaluation: how often the naive rule fires on pairs of
windows with no real change, against the calibrated detector's 1 in 500.

The no-change pairs are the same stratified 10+10 burst splits the calibrations use
(experiments.calibrate_length.split_burst). Both naive modes see the very same splits.

Caveat: burst windows hold 10 repeats per probe and production windows 12, so the
pass-rate noise here is a little higher than in production. The naive rule has no
scaling, so its rate here is slightly pessimistic.

Run:  python -m experiments.naive_false_alarms [drop_points] [n_splits]   (prints only; default 5.0 and 20000)
"""

import random
import statistics
import sys
from multiprocessing import Pool

from experiments.calibrate_length import (
    BURST_DB,
    CATEGORIES,
    MODELS,
    burst_rows,
    split_burst,
)
from src.baselines.naive import detect_naive
from src.db.connection import connect
from src.detectors.pass_rate import load_thresholds

DROP_POINTS = 5.0  # D20: chosen before looking at any comparison result
REAL_CATEGORIES = [
    c for c in CATEGORIES if c != "all"
]  # "all" is the pooled key, not a category


def _naive_pair(model, category, window_a, window_b, drop_points):
    """(effect, two_sided_alarm, drops_only_alarm) for one pair of windows.

    A pair with no data gives effect None and no alarm.
    """
    two = detect_naive(
        model, category, window_a, window_b, drop_points, drops_only=False
    )
    drops = detect_naive(
        model, category, window_a, window_b, drop_points, drops_only=True
    )
    return two.effect_size, two.status == "alarm", drops.status == "alarm"


def naive_null_alarms(model, category, rows, drop_points, n_splits, seed):
    """Per no-change split: (effect in points, two_sided_alarm, drops_only_alarm), as a list.

    One random.Random(seed) makes all splits, and both modes see each split.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n_splits):
        window_a, window_b = split_burst(rows, rng)
        out.append(_naive_pair(model, category, window_a, window_b, drop_points))
    return out


def any_category_alarms(model, rows_all, probe_category, drop_points, n_splits, seed):
    """Per split: (any_two_sided, any_drops_only), alarm if ANY real category alarms.

    The model's rows are split once per split (probes split independently, so this is the same
    kind of split as per category) and that one split is partitioned by the probes' category,
    so every category sees the same random draw. This is how a naive monitor would be used.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n_splits):
        window_a, window_b = split_burst(rows_all, rng)
        any_two = any_drops = False
        for category in REAL_CATEGORIES:
            part_a = [r for r in window_a if probe_category[r["probe_id"]] == category]
            part_b = [r for r in window_b if probe_category[r["probe_id"]] == category]
            _, two, drops = _naive_pair(model, category, part_a, part_b, drop_points)
            any_two |= two
            any_drops |= drops
        out.append((any_two, any_drops))
    return out


def rate(flags):
    return sum(flags) / len(flags) if flags else None


def _key_job(args):
    model, category, drop_points, n_splits = args
    rows = burst_rows(model, category)
    res = naive_null_alarms(model, category, rows, drop_points, n_splits, seed=0)
    effects = [e for e, _, _ in res if e is not None]
    return {
        "key": f"{model}|{category}",
        "two": rate([t for _, t, _ in res]),
        "drops": rate([d for _, _, d in res]),
        "sd": statistics.pstdev(effects),
        "max": max(abs(e) for e in effects),
    }


def _any_job(args):
    model, drop_points, n_splits = args
    conn = connect(str(BURST_DB))
    probe_category = dict(conn.execute("SELECT probe_id, category FROM probes"))
    res = any_category_alarms(
        model,
        burst_rows(model, "all", conn),
        probe_category,
        drop_points,
        n_splits,
        seed=0,
    )
    return {
        "model": model,
        "two": rate([t for t, _ in res]),
        "drops": rate([d for _, d in res]),
    }


if __name__ == "__main__":
    drop_points = float(sys.argv[1]) if len(sys.argv) > 1 else DROP_POINTS
    n_splits = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    cal = load_thresholds()[
        "thresholds"
    ]  # calibrated pass-rate limits, points at 12 repeats
    with Pool(5) as pool:
        keys = pool.map(
            _key_job,
            [(m, c, drop_points, n_splits) for m in MODELS for c in CATEGORIES],
        )
        anys = pool.map(_any_job, [(m, drop_points, n_splits) for m in MODELS])

    print(
        f"Naive {drop_points} points, {n_splits} no-change burst splits per row (seed 0)"
    )
    print(
        f"{'key':32}{'two-sided %':>12}{'drops-only %':>14}{'effect sd':>11}{'max |eff|':>11}{'calibrated limit':>18}"
    )
    for r in keys:
        limit = cal.get(r["key"])
        print(
            f"{r['key']:32}{r['two'] * 100:>12.3f}{r['drops'] * 100:>14.3f}"
            f"{r['sd']:>11.3f}{r['max']:>11.3f}{('null' if limit is None else f'{limit:.2f}'):>18}"
        )
    print(
        f"\nAny of {len(REAL_CATEGORIES)} categories ({', '.join(REAL_CATEGORIES)}) alarms, per model:"
    )
    print(f"{'model':32}{'two-sided %':>12}{'drops-only %':>14}")
    for r in anys:
        print(f"{r['model']:32}{r['two'] * 100:>12.3f}{r['drops'] * 100:>14.3f}")
