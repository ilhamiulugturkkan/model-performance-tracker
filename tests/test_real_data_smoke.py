"""Smoke test of the pass-rate and length detectors on real scheduled windows.

Needs the real database (mpt.db, git-ignored), so it skips when the data is not there.
Run ids are pinned so the data cannot change under the test: runs 1-4 against runs 8-11.

Not every key comes out "ok" on this data, and that is expected: some models really
drifted between those windows (e.g. Haiku replies got longer, Sonnet's instruction
pass rate rose) and the burst noise is a lower bound on real run-to-run noise. So the
test checks what must always hold, not that nothing alarms:
  - a window compared with itself never alarms,
  - every key gives a well-formed result,
  - a key with no threshold never alarms,
  - alarms are a minority (a unit or scaling bug would make nearly everything alarm).
"""

import math
from pathlib import Path

import pytest

from src.db.connection import DB_PATH, connect
from src.detectors import output_length, pass_rate

REFERENCE_RUNS = [1, 2, 3, 4]
LATEST_RUNS = [8, 9, 10, 11]
MODELS = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-6.1-terra",
]
CATEGORIES = ["coding", "instructions", "math", "structured", "all"]
DETECTORS = {"pass_rate": pass_rate, "length": output_length}


@pytest.fixture(scope="module")
def conn():
    if not Path(DB_PATH).exists():
        pytest.skip("mpt.db not available")
    connection = connect(DB_PATH)
    have = {r[0] for r in connection.execute("SELECT DISTINCT run_id FROM calls")}
    if not set(REFERENCE_RUNS + LATEST_RUNS) <= have:
        pytest.skip("scheduled runs 1-4 and 8-11 not in mpt.db")
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def thresholds():
    data = pass_rate.load_thresholds()
    if "length" not in data:
        pytest.skip("thresholds.json has no length section")
    return data


def run_all(conn, thresholds, ref_runs, cur_runs, categories=CATEGORIES):
    """{(detector, key): result} for every model and the given categories."""
    results = {}
    for name, module in DETECTORS.items():
        for model in MODELS:
            for category in categories:
                results[(name, f"{model}|{category}")] = module.detect(
                    model,
                    category,
                    ref_runs,
                    cur_runs,
                    conn=conn,
                    thresholds=thresholds,
                )
    return results


@pytest.fixture(scope="module")
def reference_vs_latest(conn, thresholds):
    # Computed once: the p-values and intervals make each detection take a moment.
    return run_all(conn, thresholds, REFERENCE_RUNS, LATEST_RUNS)


def test_a_window_compared_with_itself_never_alarms(conn, thresholds):
    same = run_all(conn, thresholds, REFERENCE_RUNS, REFERENCE_RUNS, ["math", "all"])
    for (name, key), r in same.items():
        assert r.status != "alarm", f"{name} {key}: {r.reason}"
        if r.status == "ok":
            assert r.effect_size == pytest.approx(0, abs=1e-9), f"{name} {key}"


def test_reference_vs_latest_gives_well_formed_results(reference_vs_latest):
    results = reference_vs_latest
    for (name, key), r in results.items():
        label = f"{name} {key}"
        assert r.detector == name and r.context.window_runs == LATEST_RUNS, label
        assert r.status in ("ok", "alarm", "no_threshold", "no_data"), label
        if r.status in ("ok", "alarm"):
            assert r.context.threshold > 0 and math.isfinite(r.effect_size), label
            assert r.ci_low <= r.ci_high, label
            # The length detector has no p-value when every probe is unchanged.
            assert (
                (0 <= r.p_value <= 1) if r.p_value is not None else r.effect_size == 0
            ), label
            assert (abs(r.effect_size) > r.context.threshold) == (
                r.status == "alarm"
            ), label
        else:
            assert r.context.threshold is None and r.p_value is None, label

    # Both detectors must actually have checked most keys, and alarms must be the exception.
    for name in DETECTORS:
        statuses = [r.status for (n, _), r in results.items() if n == name]
        checked = sum(s in ("ok", "alarm") for s in statuses)
        assert checked >= 12, f"{name} checked only {checked} keys"
        assert statuses.count("alarm") < checked / 2, f"{name} alarms on most keys"


def test_keys_without_a_threshold_are_skipped_not_alarmed(
    reference_vs_latest, thresholds
):
    results = reference_vs_latest
    for name, section in (("pass_rate", thresholds), ("length", thresholds["length"])):
        for key, value in section["thresholds"].items():
            if value is None:
                assert results[(name, key)].status == "no_threshold", f"{name} {key}"
