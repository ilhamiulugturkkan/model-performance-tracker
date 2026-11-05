"""Tests for the coding scorer (unit_tests).

Two layers:

1. Logic tests. No Docker: a stand-in runner replaces the sandbox.
2. Sandbox tests. Real Docker, skipped automatically when Docker or the image is missing.
   Build the image once (and again whenever child.py changes):

       docker build -t watchdog-sandbox src/scorer/unit_tests

The probes come from probes/coding.jsonl. The reference solutions come from
tests/coding_reference.py (one correct solution per probe).
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.scorer import get_scorer
from src.scorer.unit_tests import unit_tests
from src.scorer.unit_tests.code_scorer import IMAGE
from src.scorer.unit_tests.scoring import extract_code

HERE = Path(__file__).resolve().parent
PROBES_FILE = HERE.parent / "probes" / "coding.jsonl"  # <repo>/probes/coding.jsonl

PROBES = [
    json.loads(line)
    for line in PROBES_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
_parts = re.split(
    r"# --- (\w+) ---\n", (HERE / "coding_reference.py").read_text(encoding="utf-8")
)
REFERENCE = {_parts[i]: _parts[i + 1].strip() for i in range(1, len(_parts), 2)}


def probe_for(entry_point):
    for probe in PROBES:
        if probe["entry_point"] == entry_point:
            return probe
    pytest.skip(f"no probe with entry point {entry_point!r} in {PROBES_FILE.name}")


# ----------------------------------------------------------------------------
# Layer 1: logic, no Docker
# ----------------------------------------------------------------------------
PASS = {"loaded": True, "failed": [], "error": None, "load_error": None}
PROBE = {"entry_point": "f", "tests": ["assert f() == 1"], "timeout_seconds": 5}


def fake_runner(result):
    """A stand-in for the sandbox that returns `result` and remembers what it was given."""
    calls = []

    def runner(probe, code):
        calls.append(code)
        return result

    runner.calls = calls
    return runner


def test_the_scorer_is_registered():
    assert get_scorer("unit_tests") is not None


@pytest.mark.parametrize(
    "sandbox_result,want_passed,want_reason",
    [
        (PASS, True, "passed"),
        ({**PASS, "failed": ["assert f() == 1"]}, False, "tests_failed"),
        (
            {"loaded": False, "failed": [], "error": "TIMEOUT", "load_error": None},
            False,
            "timeout",
        ),
        (
            {
                "loaded": False,
                "failed": [],
                "error": "NO_RESULT",
                "load_error": "returncode=1: boom",
            },
            False,
            "no_result",
        ),
        # wrong function name: the code loaded, the function just isn't there
        (
            {
                "loaded": False,
                "failed": [],
                "error": "Different entry points",
                "load_error": None,
            },
            False,
            "missing_function",
        ),
        # syntax error: the code never loaded, and the missing function is only a consequence
        (
            {
                "loaded": False,
                "failed": [],
                "error": "Different entry points",
                "load_error": "SyntaxError: bad",
            },
            False,
            "did_not_load",
        ),
        # a stray sys.exit(0) must never count as a pass
        (
            {
                "loaded": False,
                "failed": [],
                "error": "SystemExit: 0",
                "load_error": "SystemExit: 0",
            },
            False,
            "did_not_load",
        ),
    ],
)
def test_every_sandbox_outcome_maps_to_the_right_verdict(
    sandbox_result, want_passed, want_reason
):
    passed, details = unit_tests(
        PROBE, "def f():\n    return 1", runner=fake_runner(sandbox_result)
    )
    assert passed is want_passed
    assert details["reason"] == want_reason


def test_failed_tests_and_load_error_are_kept_in_details():
    result = {
        "loaded": False,
        "failed": [],
        "error": "Different entry points",
        "load_error": "SyntaxError: bad",
    }
    _, details = unit_tests(PROBE, "x", runner=fake_runner(result))
    assert details["load_error"] == "SyntaxError: bad"

    result = {**PASS, "failed": ["assert f() == 1"]}
    _, details = unit_tests(PROBE, "x", runner=fake_runner(result))
    assert details["failed_tests"] == ["assert f() == 1"]


def test_missing_reply_fails_without_calling_the_sandbox():
    runner = fake_runner(PASS)
    assert unit_tests(PROBE, None, runner=runner) == (False, {"reason": "no_reply"})
    assert runner.calls == []


def test_the_sandbox_gets_the_code_without_the_fence():
    runner = fake_runner(PASS)
    passed, details = unit_tests(
        PROBE,
        "Here you go:\n```python\ndef f():\n    return 1\n```\nEnjoy!",
        runner=runner,
    )
    assert passed is True
    assert details["fenced"] is True
    assert runner.calls == ["def f():\n    return 1\n"]


def test_an_unfenced_reply_is_passed_through_unchanged():
    runner = fake_runner(PASS)
    _, details = unit_tests(PROBE, "def f():\n    return 1", runner=runner)
    assert details["fenced"] is False
    assert runner.calls == ["def f():\n    return 1"]


def test_the_block_that_defines_the_function_is_preferred():
    reply = "```python\nprint('example')\n```\n```python\ndef f():\n    return 1\n```"
    code, fenced = extract_code(reply, "f")
    assert fenced is True
    assert "def f()" in code


def test_a_broken_probe_is_a_loud_error_not_a_model_failure():
    # Checked before any container starts, so this needs no Docker.
    with pytest.raises(ValueError):
        unit_tests(
            {"entry_point": "f", "tests": "assert f() == 1"}, "def f():\n    return 1"
        )
    with pytest.raises(ValueError):
        unit_tests({"tests": ["assert 1"]}, "x")


# ----------------------------------------------------------------------------
# Layer 2: the real sandbox
# ----------------------------------------------------------------------------
def docker_ready():
    if shutil.which("docker") is None:
        return False
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return done.returncode == 0


requires_docker = pytest.mark.skipif(
    not docker_ready(),
    reason=f"Docker or the image {IMAGE!r} is not available. Build it: docker build -t {IMAGE} src/scorer/unit_tests",
)


def probe_containers():
    out = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=probe-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return set(out.stdout.split())


@requires_docker
@pytest.mark.parametrize("probe", PROBES, ids=[p["id"] for p in PROBES])
def test_the_reference_solution_passes_every_real_probe(probe):
    reference = REFERENCE.get(probe["entry_point"])
    if reference is None:
        pytest.skip(f"no reference solution for {probe['entry_point']}")
    passed, details = unit_tests(probe, reference)
    assert passed, details


@requires_docker
def test_a_boundary_bug_fails_on_exactly_the_edge_case_test():
    probe = probe_for("count_divisible_above")
    wrong = REFERENCE["count_divisible_above"].replace("n > m", "n >= m")
    passed, details = unit_tests(probe, wrong)
    assert passed is False
    assert details["reason"] == "tests_failed"
    assert details["failed_tests"] == [
        "assert count_divisible_above([3, 6], 3, 3) == 1"
    ]


@requires_docker
def test_a_fenced_reply_passes_and_is_logged():
    probe = probe_for("count_divisible_above")
    passed, details = unit_tests(
        probe, "```python\n" + REFERENCE["count_divisible_above"] + "\n```"
    )
    assert passed is True
    assert details["fenced"] is True


@requires_docker
def test_a_wrong_function_name_is_reported_as_a_missing_function():
    probe = probe_for("count_divisible_above")
    passed, details = unit_tests(
        probe,
        REFERENCE["count_divisible_above"].replace("count_divisible_above", "count_it"),
    )
    assert (passed, details["reason"]) == (False, "missing_function")


@requires_docker
def test_a_syntax_error_is_reported_as_not_loading():
    probe = probe_for("count_divisible_above")
    passed, details = unit_tests(
        probe, "def count_divisible_above(nums, k, m)\n    return 0"
    )
    assert (passed, details["reason"]) == (False, "did_not_load")
    assert "SyntaxError" in details["load_error"]


@requires_docker
def test_a_stray_exit_in_the_reply_is_not_a_false_pass():
    probe = probe_for("count_divisible_above")
    passed, _details = unit_tests(
        probe,
        "def count_divisible_above(nums, k, m):\n    return 0\n\nimport sys\nsys.exit(0)",
    )
    assert passed is False


@requires_docker
def test_a_reply_with_a_nul_character_does_not_crash_the_scorer():
    probe = probe_for("count_divisible_above")
    passed, _ = unit_tests(probe, REFERENCE["count_divisible_above"] + "\n# \x00")
    assert passed is False


@requires_docker
def test_an_infinite_loop_times_out_and_leaves_no_container_behind():
    probe = dict(probe_for("count_divisible_above"), timeout_seconds=2)
    before = probe_containers()
    passed, details = unit_tests(probe, "while True:\n    pass")
    assert (passed, details["reason"]) == (False, "timeout")
    assert probe_containers() - before == set()
