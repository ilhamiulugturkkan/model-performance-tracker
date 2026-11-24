"""Calibration of the output-text detector thresholds (six reply features).

The burst run holds 20 repeats per probe. Splitting them at random into two
windows of 10 gives pairs of windows where nothing changed. On each pair the
detector computes one effect per feature, so the spread of those effects is the
noise.

Six features checked separately would raise the false-alarm rate up to six
times. So the alarm rule is calibrated as a whole:
  sd_f      = spread (sd) of feature f's no-change effects
  M         = per split, the largest |effect_f| / sd_f over the features
  M_thr     = 99.8th percentile of M (1 false alarm in 500 for the whole detector)
  t_f       = M_thr * sd_f, scaled from 10 to 12 repeats by sqrt(10/12)
The detector alarms when any |effect_f| > t_f. A feature whose effect never varies
(sd 0) gets null and is not checked; a key where no feature varies is null.

Run:  python -m experiments.calibrate_text [n_splits]   (default 20000; rewrites the "text" key)
"""

import json
import random
import statistics
from datetime import UTC, datetime
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
    burst_rows,
    split_burst,
)
from src.detectors.output_dist import FEATURES, compare_window_features
from src.detectors.pass_rate import _percentile

DEAD_SD = 1e-9  # a feature whose no-change effects spread less than this never moves


def null_feature_effects(rows, n_splits, seed):
    """No-change effects of n_splits random splits, as a list of {feature: effect or None}.

    Uses one random.Random(seed) for all splits.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n_splits):
        window_a, window_b = split_burst(rows, rng)
        out.append(compare_window_features(window_a, window_b)["effects"])
    return out


def text_thresholds(effects):
    """{feature: threshold at 12 repeats, or None} from no-change effects, or None for the whole key.

    See the module docstring for the rule. Returns None when no feature varies
    (a deterministic key).
    """
    sd = {}
    for name in FEATURES:
        values = [e[name] for e in effects if e[name] is not None]
        sd[name] = statistics.pstdev(values) if len(values) > 1 else 0.0
    live = [name for name in FEATURES if sd[name] > DEAD_SD]
    if not live:
        return None

    worst = sorted(
        max(abs(e[name]) / sd[name] for name in live if e[name] is not None)
        for e in effects
        if any(e[name] is not None for name in live)
    )
    m_thr = _percentile(worst, THRESHOLD_PERCENTILE)
    return {
        name: (m_thr * sd[name] * SCALE if name in live else None) for name in FEATURES
    }


def count_varying_probes(rows):
    """How many probes have more than one distinct reply text in `rows`."""
    texts = {}
    for row in rows:
        texts.setdefault(row["probe_id"], set()).add(row["raw_output"])
    return sum(len(values) > 1 for values in texts.values())


def _rounded(entry):
    if entry is None:
        return None
    return {k: (None if v is None else float(f"{v:.6g}")) for k, v in entry.items()}


def build_text_section(n_splits=20000, seed=0, models=MODELS, categories=CATEGORIES):
    """The "text" section of thresholds.json: a {feature: threshold} entry per "model|category".

    "approximate" lists the keys where fewer than MIN_VARYING probes have replies
    that ever differ in the burst.
    """
    thresholds = {}
    approximate = []
    for model in models:
        for category in categories:
            rows = burst_rows(model, category)
            key = f"{model}|{category}"
            thresholds[key] = _rounded(
                text_thresholds(null_feature_effects(rows, n_splits, seed))
            )
            if thresholds[key] is not None and count_varying_probes(rows) < MIN_VARYING:
                approximate.append(key)
    return {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_commit(),
        "metric": "output text features",
        "features": list(FEATURES),
        "unit": "each feature's own unit (lines, markers, share, characters, fraction, ratio)",
        "repeats_per_probe": PROD_REPEATS,
        "method": (
            f"{n_splits} stratified burst splits; per feature, effect = mean over probes of "
            "(window 2 mean - window 1 mean) of the reply feature; t_f = M * sd_f, where sd_f is "
            "the spread of feature f's no-change effects and M is the "
            f"{THRESHOLD_PERCENTILE}th percentile of the per-split max over features of "
            "|effect_f| / sd_f; scaled from 10 to 12 repeats by sqrt(10/12); the detector alarms "
            "when any |effect_f| > t_f; null = feature (or whole key) never varies"
        ),
        "n_splits": n_splits,
        "seed": seed,
        "approximate": approximate,
        "thresholds": thresholds,
    }


def write_text_section(section, path=THRESHOLDS_PATH):
    """Set the "text" key of the thresholds file, leaving every other key's text untouched.

    The file must be in the canonical format (json.dumps(indent=2) + newline), so
    that re-serialising cannot change anything else. Otherwise this raises instead
    of reformatting the file.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if json.dumps(data, indent=2) + "\n" != raw:
        raise ValueError(f"{path} is not in canonical format; refusing to rewrite it")
    data["text"] = section
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys

    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    section = build_text_section(n_splits=n_splits)
    write_text_section(section)
    for key, entry in section["thresholds"].items():
        print(f"{key:32} {entry}")
    print(f"wrote 'text' to {THRESHOLDS_PATH} ({section['commit']})")
