"""Smoke test of the four detectors on real scheduled windows, through their database paths.

Needs the real database (mpt.db, git-ignored), so it skips when the data is not there.
Run ids are pinned so the data cannot change under the test: runs 4-7 against runs 18-21. Both
windows are on the same code commit (runs 1-2 are not, which is why the old 1-4 pair was dropped).

Detectors: pass rate, length, text (reply shape) and latency (mixed windows, plus night and day
separately). The project assumes nothing really changed in these runs, so an alarm here counts as a
false alarm. A few weak ones are expected (the burst noise is a lower bound on real noise; for example
Haiku's hidden reasoning grew, which the length detector counts on purpose). So the test checks what
must always hold, not that nothing alarms:
  - the pinned runs share one commit,
  - a window compared with itself never alarms,
  - every key gives a well-formed result,
  - a key with no threshold never alarms,
  - alarms are a small minority (a unit or scaling bug would make many keys alarm).

The p-value and interval resamples are set low (N_RESAMPLES): they do not change the alarm, and the
whole file stays at a few seconds.
"""

import math
from pathlib import Path

import pytest

from src.db.connection import DB_PATH, connect
from src.detectors import output_dist, output_latency, output_length, pass_rate

REFERENCE_RUNS = [4, 5, 6, 7]
LATEST_RUNS = [18, 19, 20, 21]
N_RESAMPLES = 200
FAST = {"n_perm": N_RESAMPLES, "n_boot": N_RESAMPLES}
MODELS = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-6.1-terra",
]
CATEGORIES = ["coding", "instructions", "math", "structured", "all"]
# name -> (module, extra arguments for detect, section of thresholds.json holding its keys)
DETECTORS = {
    "pass_rate": (pass_rate, FAST, None),
    "length": (output_length, FAST, "length"),
    "text": (output_dist, {}, "text"),
    "latency": (output_latency, FAST, "latency"),
}
SECTION_REQUIRED = ("length", "text", "latency")
STATUSES = ("ok", "alarm", "no_threshold", "no_data")


@pytest.fixture(scope="module")
def conn():
    if not Path(DB_PATH).exists():
        pytest.skip("mpt.db not available")
    connection = connect(DB_PATH)
    have = {r[0] for r in connection.execute("SELECT DISTINCT run_id FROM calls")}
    if not set(REFERENCE_RUNS + LATEST_RUNS) <= have:
        pytest.skip("scheduled runs 4-7 and 18-21 not in mpt.db")
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def thresholds():
    data = pass_rate.load_thresholds()
    missing = [s for s in SECTION_REQUIRED if s not in data]
    if missing:
        pytest.skip(f"thresholds.json has no {missing} section")
    return data


def section(thresholds, name):
    section_name = DETECTORS[name][2]
    return thresholds if section_name is None else thresholds[section_name]


def run_all(conn, thresholds, ref_runs, cur_runs, categories=CATEGORIES):
    """{(detector, key): result} for every model and the given categories, through detect()."""
    results = {}
    for name, (module, kwargs, _) in DETECTORS.items():
        for model in MODELS:
            for category in categories:
                results[(name, f"{model}|{category}")] = module.detect(
                    model,
                    category,
                    ref_runs,
                    cur_runs,
                    conn=conn,
                    thresholds=thresholds,
                    **kwargs,
                )
    return results


def run_slots(conn, thresholds, ref_runs, cur_runs, categories=CATEGORIES):
    """{(slot, key): result} for the latency detector run on night and day separately."""
    results = {}
    for model in MODELS:
        for category in categories:
            slots = output_latency.detect_slots(
                model,
                category,
                ref_runs,
                cur_runs,
                conn=conn,
                thresholds=thresholds,
                **FAST,
            )
            for slot, result in slots.items():
                results[(slot, f"{model}|{category}")] = result
    return results


@pytest.fixture(scope="module")
def reference_vs_latest(conn, thresholds):
    return run_all(conn, thresholds, REFERENCE_RUNS, LATEST_RUNS)


@pytest.fixture(scope="module")
def slots_reference_vs_latest(conn, thresholds):
    return run_slots(conn, thresholds, REFERENCE_RUNS, LATEST_RUNS)


def test_the_pinned_runs_share_one_code_commit(conn):
    commits = {
        row[0]
        for row in conn.execute(
            f"SELECT DISTINCT git_commit FROM runs WHERE run_id IN ({','.join('?' * 8)})",
            REFERENCE_RUNS + LATEST_RUNS,
        )
    }
    assert len(commits) == 1, f"the pinned windows span several commits: {commits}"


def test_a_window_compared_with_itself_never_alarms(conn, thresholds):
    same = run_all(conn, thresholds, REFERENCE_RUNS, REFERENCE_RUNS, ["math", "all"])
    for (name, key), r in same.items():
        assert r.status != "alarm", f"{name} {key}: {r.reason}"
        if r.status == "ok":
            zero = r.statistic if name == "text" else r.effect_size
            assert zero == pytest.approx(0, abs=1e-9), f"{name} {key}"
    slots = run_slots(conn, thresholds, REFERENCE_RUNS, REFERENCE_RUNS, ["math", "all"])
    for (slot, key), r in slots.items():
        assert r.status != "alarm", f"latency {slot} {key}: {r.reason}"


def check_result(name, r, label):
    assert r.detector == name and r.context.window_runs == LATEST_RUNS, label
    assert r.status in STATUSES, label
    if r.status not in ("ok", "alarm"):
        assert r.context.threshold is None and r.p_value is None, label
        return
    if name == "text":
        # statistic is the largest |effect| / limit over the checked features; no p-value or interval
        assert r.metric_unit == "score" and r.context.threshold == 1.0, label
        assert math.isfinite(r.statistic) and r.statistic >= 0, label
        assert r.p_value is None and r.ci_low is None and r.ci_high is None, label
        assert (r.statistic > 1 + output_dist.TIE_TOLERANCE) == (r.status == "alarm"), (
            label
        )
        return
    assert r.context.threshold > 0 and math.isfinite(r.effect_size), label
    assert r.ci_low <= r.ci_high, label
    # No p-value is possible when every probe is unchanged (the effect is then exactly 0).
    assert (0 <= r.p_value <= 1) if r.p_value is not None else r.effect_size == 0, label
    tolerance = 1 + pass_rate.TIE_TOLERANCE if name == "pass_rate" else 1
    assert (abs(r.effect_size) > r.context.threshold * tolerance) == (
        r.status == "alarm"
    ), label


def test_reference_vs_latest_gives_well_formed_results(reference_vs_latest):
    for (name, key), r in reference_vs_latest.items():
        check_result(name, r, f"{name} {key}")

    # Every detector must actually have checked most of its keys, and alarms must be a small minority.
    for name in DETECTORS:
        statuses = [r.status for (n, _), r in reference_vs_latest.items() if n == name]
        checked = sum(s in ("ok", "alarm") for s in statuses)
        assert checked >= 12, f"{name} checked only {checked} keys"
        assert statuses.count("alarm") <= checked / 4, f"{name} alarms on too many keys"


def test_latency_night_and_day_results_are_well_formed(slots_reference_vs_latest):
    for (slot, key), r in slots_reference_vs_latest.items():
        label = f"latency {slot} {key}"
        check_result("latency", r, label)
        assert f"[{slot}]" in r.reason, label
    statuses = [r.status for r in slots_reference_vs_latest.values()]
    checked = sum(s in ("ok", "alarm") for s in statuses)
    assert checked >= 24, f"slot latency checked only {checked} results"
    assert statuses.count("alarm") <= checked / 4


def test_keys_without_a_threshold_are_skipped_not_alarmed(
    reference_vs_latest, slots_reference_vs_latest, thresholds
):
    for name in DETECTORS:
        for key, value in section(thresholds, name)["thresholds"].items():
            if value is None and key.count("|") == 1:  # slot keys are checked below
                assert reference_vs_latest[(name, key)].status == "no_threshold", (
                    f"{name} {key}"
                )
    for key, value in thresholds["latency"]["thresholds"].items():
        if value is None and key.count("|") == 2:  # a "model|category|slot" key
            model_category, slot = key.rsplit("|", 1)
            assert (
                slots_reference_vs_latest[(slot, model_category)].status
                == "no_threshold"
            ), key
