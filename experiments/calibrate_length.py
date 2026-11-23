"""Calibration of the output-length detector threshold, built in small pieces.

Piece 1: burst_rows       - the burst rows for one model and category.
Piece 2: split_burst      - one random no-change pair of windows made from them.
Piece 3: null_effects     - repeat piece 2 to get the effect distribution when nothing changed.
Piece 4: length_threshold - turn that into a threshold (99.8th percentile, 12-repeat scale).
Piece 5: build_length_section / write_length_section - all models x categories into thresholds.json.

Run:  python -m experiments.calibrate_length [n_splits]   (default 20000; rewrites the "length" key)
"""

import json
import math
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.db.connection import connect
from src.detectors.data import load_window
from src.detectors.output_length import compare_window_output_lengths
from src.detectors.pass_rate import _percentile

BURST_DB = Path(__file__).resolve().parent.parent / "mpt_burst.db"
BURST_RUN_ID = 1
K = 10  # repeats per probe per window in the burst (20 -> 10 + 10)
PROD_REPEATS = 12  # production window: 4 runs x 3 repeats
SCALE = math.sqrt(K / PROD_REPEATS)
THRESHOLD_PERCENTILE = 99.8  # 1 false alarm in 500
# Fewer varying probes than this = lumpy null distribution, threshold only approximate
# (the same rule as the pass-rate "approximate" list in noise_report.py).
MIN_VARYING = 3
THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "detectors" / "thresholds.json"
)
MODELS = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-6.1-terra",
]
CATEGORIES = ["coding", "instructions", "math", "structured", "all"]


def burst_rows(model, category, conn=None):
    """Rows of the burst run for one model and category ("all" = every category).

    Same row format as the detector takes: probe_id, repeat_no, output_tokens, ...
    """
    if conn is None:
        # connect() would silently create an empty DB if the path were wrong.
        if not BURST_DB.exists():
            raise FileNotFoundError(f"burst database not found: {BURST_DB}")
        conn = connect(str(BURST_DB))
    return load_window(model, category, [BURST_RUN_ID], conn=conn)


def split_burst(rows, rng):
    """One fake no-change pair of windows: (window_a, window_b), lists of rows.

    For each probe, shuffle its rows and deal n // 2 to each window (20 repeats ->
    10 and 10). With fewer usable rows (failed calls) the sides stay equal and the
    odd leftover row is dropped. `rng` is a random.Random; the input is not modified.
    """
    by_probe = {}
    for row in rows:
        by_probe.setdefault(row["probe_id"], []).append(row)

    window_a, window_b = [], []
    for probe_rows in by_probe.values():
        shuffled = probe_rows[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        window_a.extend(shuffled[:half])
        window_b.extend(shuffled[half : 2 * half])
    return window_a, window_b


def null_effects(rows, n_splits, seed):
    """Length effect (mean per-probe % change) of n_splits random no-change splits.

    Uses one random.Random(seed) for all splits. Splits whose effect is None
    (no usable probes) are skipped, so the result can be shorter than n_splits.
    """
    rng = random.Random(seed)
    effects = []
    for _ in range(n_splits):
        window_a, window_b = split_burst(rows, rng)
        effect = compare_window_output_lengths(window_a, window_b)["effect"]
        if effect is not None:
            effects.append(effect)
    return effects


def length_threshold(effects):
    """Alert threshold (percent) for a 12-repeat window, from no-change effects.

    99.8th percentile of |effect|, scaled by sqrt(10/12) from the 10-repeat burst
    windows. Returns None for a deterministic category: every |effect| < 1e-9
    (or no effects at all).
    """
    magnitudes = sorted(abs(e) for e in effects)
    if not magnitudes or magnitudes[-1] < 1e-9:
        return None
    return _percentile(magnitudes, THRESHOLD_PERCENTILE) * SCALE


def _git_commit():
    """Short HEAD hash, with "-dirty" if the working tree has uncommitted changes."""
    root = Path(__file__).resolve().parent.parent
    run = lambda *args: subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    commit = run("rev-parse", "--short", "HEAD")
    return commit + "-dirty" if run("status", "--porcelain") else commit


def count_varying_probes(rows):
    """How many probes have more than one distinct output length in `rows`."""
    lengths = {}
    for row in rows:
        lengths.setdefault(row["probe_id"], set()).add(row["output_tokens"])
    return sum(len(values) > 1 for values in lengths.values())


def build_length_section(n_splits=20000, seed=0, models=MODELS, categories=CATEGORIES):
    """The "length" section of thresholds.json: a threshold per "model|category".

    A deterministic category (every no-change effect is 0) gets None (null).
    "approximate" lists the keys with a threshold that rests on fewer than
    MIN_VARYING probes whose length ever varies in the burst.
    """
    thresholds = {}
    approximate = []
    for model in models:
        for category in categories:
            rows = burst_rows(model, category)
            effects = null_effects(rows, n_splits, seed)
            key = f"{model}|{category}"
            thresholds[key] = _round_or_none(length_threshold(effects))
            if thresholds[key] is not None and count_varying_probes(rows) < MIN_VARYING:
                approximate.append(key)
    return {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_commit(),
        "metric": "output length",
        "unit": "percent",
        "repeats_per_probe": PROD_REPEATS,
        "method": (
            f"empirical {THRESHOLD_PERCENTILE}th percentile of |effect| over {n_splits} stratified "
            "burst splits, where effect = mean per-probe % change in mean output tokens "
            "(compare_window_output_lengths), scaled from 10 to 12 repeats by sqrt(10/12); "
            "one threshold applied to |effect|; null = deterministic category"
        ),
        "n_splits": n_splits,
        "seed": seed,
        "approximate": approximate,
        "thresholds": thresholds,
    }


def _round_or_none(value):
    return None if value is None else round(value, 4)


def write_length_section(section, path=THRESHOLDS_PATH):
    """Set the "length" key of the thresholds file, leaving every other key's text untouched.

    The file must be in the canonical format (json.dumps(indent=2) + newline), so
    that re-serialising cannot change anything else. Otherwise this raises instead
    of reformatting the file.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if json.dumps(data, indent=2) + "\n" != raw:
        raise ValueError(f"{path} is not in canonical format; refusing to rewrite it")
    data["length"] = section  # replaces in place, or appends as the last key
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys

    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    section = build_length_section(n_splits=n_splits)
    write_length_section(section)
    for key, value in section["thresholds"].items():
        print(f"{key:32} {value}")
    print(f"wrote 'length' to {THRESHOLDS_PATH} ({section['commit']})")
