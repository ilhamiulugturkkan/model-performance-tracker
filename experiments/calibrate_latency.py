"""Calibration of the latency detector threshold, in two parts.

Part 1 (burst): the burst run holds 20 repeats per probe, all taken back to back. Splitting them
at random into two windows of 10 gives pairs where nothing changed, the same stratified splits as
experiments.calibrate_length. The 99.8th percentile of |effect| over many such splits, scaled from
10 to 12 repeats by sqrt(10/12), is the BURST threshold. The no-change sd of the effect (same scaling)
is kept too.

Part 2 (real runs): the burst is a calm moment, so real run-to-run noise is larger (provider load
changes over hours). experiments.latency_inflation estimates by how much, as a factor. The limit
in thresholds.json is burst threshold x applied factor, and the file records all three numbers.

Run:  python -m experiments.calibrate_latency [n_splits]   (default 20000; rewrites the "latency" key)
"""

import json
import random
import statistics
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

from experiments.calibrate_length import (
    CATEGORIES,
    MIN_VARYING,
    MODELS,
    PROD_REPEATS,
    SCALE,
    THRESHOLD_PERCENTILE,
    THRESHOLDS_PATH,
    _git_commit,
    _round_or_none,
    burst_rows,
    length_threshold,  # 99.8th percentile of |effect| x sqrt(10/12); None if all effects are 0
    split_burst,
)
from src.detectors.output_latency import compare_window_latencies, window_latency


def null_effects(rows, n_splits, seed):
    """Latency effect (mean per-probe % change in median latency) of n_splits no-change splits."""
    rng = random.Random(seed)
    effects = []
    for _ in range(n_splits):
        window_a, window_b = split_burst(rows, rng)
        effect = compare_window_latencies(window_a, window_b)["effect"]
        if effect is not None:
            effects.append(effect)
    return effects


def burst_calibration_for_key(args):
    """{"threshold": burst threshold at 12 repeats or None, "null_sd": no-change sd at 12 repeats, "varying": n}."""
    model, category, n_splits, seed = args
    rows = burst_rows(model, category)
    effects = null_effects(rows, n_splits, seed)
    # Latency is continuous, so "varying" counts probes whose latency is not constant;
    # kept for the approximate flag exactly as in the length calibration.
    lat = window_latency(rows)
    varying = sum(len(set(v)) > 1 for v in lat.values())
    return f"{model}|{category}", {
        "threshold": length_threshold(effects),
        "null_sd": statistics.pstdev(effects) * SCALE if effects else None,
        "varying": varying,
    }


def burst_calibration(n_splits=20000, seed=0, models=MODELS, categories=CATEGORIES):
    """{key: burst_calibration_for_key(...)} for all keys, computed in parallel."""
    jobs = [(m, c, n_splits, seed) for m in models for c in categories]
    with Pool(5) as pool:
        return dict(pool.map(burst_calibration_for_key, jobs))


def build_latency_section(burst, factors, n_splits=20000, seed=0, slot_factors=None):
    """The "latency" section of thresholds.json.

    `burst` is burst_calibration(...); `factors` is {key: {"estimated": f, "applied": f, ...}}
    from experiments.latency_inflation (applied is 1.0 where the estimate is not clearly above 1).
    thresholds[key] = burst threshold x applied factor, which is what the detector reads.

    `slot_factors` ({"model|category|slot": same fields}) adds a "model|category|slot" limit for
    each slot: the same burst threshold (the burst cannot tell day from night) times that slot's
    own factor. The slot keys are always approximate.
    """
    thresholds, burst_thresholds, inflation, approximate = {}, {}, {}, []
    for key, entry in burst.items():
        burst_thresholds[key] = _round_or_none(entry["threshold"])
        applied = factors.get(key, {}).get("applied", 1.0)
        thresholds[key] = (
            None
            if entry["threshold"] is None
            else _round_or_none(entry["threshold"] * applied)
        )
        if key in factors:
            inflation[key] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in factors[key].items()
            }
        # approximate: few probes vary, or the factor could not be estimated cleanly
        if thresholds[key] is not None and (
            entry["varying"] < MIN_VARYING
            or factors.get(key, {}).get("approximate", True)
        ):
            approximate.append(key)
    for slot_key, est in (slot_factors or {}).items():
        base = burst[slot_key.rsplit("|", 1)[0]]
        thresholds[slot_key] = _round_or_none(base["threshold"] * est["applied"])
        inflation[slot_key] = {
            k: (round(v, 4) if isinstance(v, float) else v) for k, v in est.items()
        }
        approximate.append(slot_key)
    return {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_commit(),
        "metric": "latency",
        "unit": "percent",
        "repeats_per_probe": PROD_REPEATS,
        "method": (
            f"limit = burst threshold x applied inflation factor. Burst threshold: empirical "
            f"{THRESHOLD_PERCENTILE}th percentile of |effect| over {n_splits} stratified burst splits, "
            "effect = mean per-probe % change in the per-probe median latency "
            "(compare_window_latencies), scaled from 10 to 12 repeats by sqrt(10/12). "
            "Inflation: spread of real window-to-window effects (reference runs 4-7 against later "
            "4-run windows) over the burst no-change spread at the same window size; applied only "
            "when above the chi-square cutoff, else 1. Keys 'model|category|night' and "
            "'model|category|day' are for one time-of-day slot (03:00 or 15:00 UTC runs) compared "
            "with itself: same burst threshold, factor from 2-run same-slot windows. One threshold "
            "applied to |effect|; null = no variation in the burst"
        ),
        "n_splits": n_splits,
        "seed": seed,
        "approximate": sorted(approximate),
        "thresholds": thresholds,
        "burst_thresholds": burst_thresholds,
        "inflation": inflation,
    }


def write_latency_section(section, path=THRESHOLDS_PATH):
    """Set the "latency" key of the thresholds file, leaving every other key's text untouched.

    The file must be in the canonical format (json.dumps(indent=2) + newline); otherwise this
    raises instead of reformatting the file.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if json.dumps(data, indent=2) + "\n" != raw:
        raise ValueError(f"{path} is not in canonical format; refusing to rewrite it")
    data["latency"] = section
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys

    from experiments.latency_inflation import estimate_factors

    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    burst = burst_calibration(n_splits)
    factors = estimate_factors(burst)
    section = build_latency_section(burst, factors, n_splits)
    write_latency_section(section)
    print(f"{'key':32}{'burst':>9}{'factor':>8}{'limit':>9}")
    for key, limit in section["thresholds"].items():
        f = factors.get(key, {}).get("applied", 1.0)
        b = section["burst_thresholds"][key]
        print(
            f"{key:32}{'null' if b is None else f'{b:.2f}':>9}{f:>8.2f}{'null' if limit is None else f'{limit:.2f}':>9}"
        )
    print(f"wrote 'latency' to {THRESHOLDS_PATH} ({section['commit']})")
