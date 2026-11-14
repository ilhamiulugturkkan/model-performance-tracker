"""Tests for the runner: collect, score, report. A fake caller stands in for the APIs (it costs nothing)."""

import json

import pytest

from src.db.connection import connect
from src.db.migrate import migrate
from src.probes.loader import load_probes
from src.runner import collect, is_truncated, report, score, select_probes, start_run

M1 = {
    "id": "math_001", "version": "v1", "category": "math", "topic": "integers", "split": None,
    "prompt": "What is 2 + 2? Reply with only the number.", "scorer": "numeric_match", "expected": 4, "tolerance": 0,
}  # fmt: skip
M2 = dict(
    M1, id="math_002", prompt="What is 3 * 3? Reply with only the number.", expected=9
)
I1 = {
    "id": "instr_001", "version": "v1", "category": "instructions", "topic": "word_shape", "split": None,
    "prompt": "Write exactly 3 words and nothing else.", "scorer": "rule_check", "rules": {"words": 3},
}  # fmt: skip
ANSWERS = {
    M1["prompt"]: "4",
    M2["prompt"]: "9",
    I1["prompt"]: "red green blue",
}  # every reply correct


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "test.db")
    migrate(connection)
    for name, probes in (("math", [M1, M2]), ("instructions", [I1])):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(p) + "\n" for p in probes), encoding="utf-8")
        load_probes(connection, [path])
    yield connection
    connection.close()


def result(text="4", **over):
    base = {
        "provider": "anthropic", "requested_model": "claude-haiku-5-5", "returned_model": "claude-haiku-5-5",
        "text": text, "input_tokens": 20, "output_tokens": 3, "reasoning_tokens": 0, "finish_reason": "end_turn",
        "request_id": "req_1", "http_status": 200, "error": None, "latency": 0.5, "attempts": 1,
        "raw_response": '{"x":1}',
    }  # fmt: skip
    return {**base, **over}


class Caller:
    """A fake for call_model. `plan` decides each reply; it counts every call."""

    def __init__(self, plan=None):
        self.plan = plan or (
            lambda provider, model, prompt, n: result(
                ANSWERS[prompt],
                provider=provider,
                requested_model=model,
                returned_model=model,
            )
        )
        self.calls = []

    def __call__(self, client, provider, model, prompt, settings):
        self.calls.append((provider, model, prompt))
        return self.plan(provider, model, prompt, len(self.calls))


CLIENTS = {
    "anthropic": object(),
    "openai": object(),
}  # never used, but keeps make_client out of the test
BOTH = ["haiku", "luna"]


def run_collect(conn, caller, models=BOTH, repeats=3, run_id=None, **kwargs):
    run_id = run_id or start_run(conn, {"models": models})
    summary = collect(
        conn,
        run_id,
        models,
        repeats,
        caller=caller,
        clients=CLIENTS,
        notify=kwargs.pop("notify", lambda m: None),
        **kwargs,
    )
    return run_id, summary


def count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


# ------------------------------------------------------------------ collect
def test_one_row_per_probe_model_and_repeat_with_every_field_filled(conn):
    caller = Caller()
    run_id, summary = run_collect(conn, caller)
    assert summary == {"ok": 18}  # 3 probes x 2 models x 3 repeats
    assert count(conn, "SELECT COUNT(*) FROM calls WHERE run_id = ?", run_id) == 18
    row = conn.execute(
        "SELECT provider, request_string, received_string, parameters, raw_prompt, raw_output, input_tokens, output_tokens, "
        "finish_reason, repeat_no, attempts, request_id, http_status, error, time_stamp, cost, fingerprint "
        "FROM calls WHERE provider = 'openai' AND repeat_no = 2 LIMIT 1"
    ).fetchone()
    assert row[:3] == ("openai", "gpt-6-luna", "gpt-6-luna")
    assert json.loads(row[3]) == {
        "max_tokens": 4096,
        "temperature": 0,
        "reasoning_effort": "none",
    }
    assert row[4] in ANSWERS and row[5] == ANSWERS[row[4]]
    assert (
        row[6:12] == (20, 3, "end_turn", 2, 1, "req_1")
        and row[12] == 200
        and row[13] is None
    )
    assert row[14].endswith("+00:00")  # a UTC timestamp
    assert (
        row[15] is None and row[16] is None
    )  # cost is computed later; no provider gave a fingerprint


def test_claude_parameters_say_the_temperature_was_not_settable(conn):
    run_collect(conn, Caller(), models=["haiku"], repeats=1)
    parameters = json.loads(
        conn.execute("SELECT parameters FROM calls LIMIT 1").fetchone()[0]
    )
    assert parameters == {
        "max_tokens": 4096,
        "temperature": "provider default (not settable)",
    }


def test_the_run_row_records_the_config_and_a_commit(conn):
    run_id = start_run(conn, {"models": BOTH, "repeats": 3})
    started_at, commit, config = conn.execute(
        "SELECT started_at, git_commit, config FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert json.loads(config) == {"models": BOTH, "repeats": 3}
    assert commit and started_at.endswith("+00:00")


def test_running_again_calls_nothing_and_changes_nothing(conn):
    caller = Caller()
    run_id, _ = run_collect(conn, caller)
    made = len(caller.calls)
    _, summary = run_collect(conn, caller, run_id=run_id)
    assert summary == {"already_done": 18}
    assert len(caller.calls) == made
    assert count(conn, "SELECT COUNT(*) FROM calls") == 18


def test_a_failed_call_is_saved_then_retried_on_resume_without_touching_good_rows(conn):
    failing = Caller(
        lambda p, m, prompt, n: (
            result(
                None,
                error="server_error: boom",
                http_status=500,
                attempts=4,
                finish_reason=None,
            )
            if n % 2 == 0
            else result(ANSWERS[prompt])
        )
    )
    run_id, summary = run_collect(conn, failing, models=["haiku"], repeats=1)
    assert summary == {"ok": 2, "failed": 1}
    failed = conn.execute(
        "SELECT error, http_status, attempts, raw_output FROM calls WHERE error IS NOT NULL"
    ).fetchone()
    assert failed == ("server_error: boom", 500, 4, None)

    healthy = Caller()
    _, summary = run_collect(conn, healthy, models=["haiku"], repeats=1, run_id=run_id)
    assert summary == {
        "already_done": 2,
        "ok": 1,
    }  # only the failed one was called again
    assert len(healthy.calls) == 1
    assert count(conn, "SELECT COUNT(*) FROM calls WHERE error IS NOT NULL") == 0
    assert count(conn, "SELECT COUNT(*) FROM calls") == 3  # replaced, not duplicated


def test_an_empty_balance_stops_that_provider_only_and_alerts_once(conn):
    def plan(provider, model, prompt, n):
        if provider == "openai":
            return result(
                None,
                error="quota_exhausted: You exceeded your current quota",
                http_status=429,
                finish_reason=None,
            )
        return result(ANSWERS[prompt])

    alerts = []
    caller = Caller(plan)
    _, summary = run_collect(conn, caller, repeats=3, notify=alerts.append)
    assert [c[0] for c in caller.calls].count(
        "openai"
    ) == 1  # one attempt, then it stopped asking
    assert [c[0] for c in caller.calls].count(
        "anthropic"
    ) == 9  # the other provider carried on
    assert summary["skipped_provider_stopped"] == 8
    assert len(alerts) == 1 and "openai" in alerts[0] and "resume" in alerts[0]


def test_a_failing_model_is_stopped_alone_and_its_sibling_keeps_running(conn):
    def plan(provider, model, prompt, n):
        if model == "claude-sonnet-5-5":
            return result(
                None, error="server_error: boom", http_status=500, finish_reason=None
            )
        return result(
            ANSWERS[prompt],
            provider=provider,
            requested_model=model,
            returned_model=model,
        )

    alerts = []
    caller = Caller(plan)
    _, summary = run_collect(
        conn, caller, models=["haiku", "sonnet"], repeats=3, notify=alerts.append
    )
    models_called = [c[1] for c in caller.calls]
    assert (
        models_called.count("claude-sonnet-5-5") == 3
    )  # three failures, then it stopped asking
    assert models_called.count("claude-haiku-5-5") == 9  # same provider, carried on
    assert summary["skipped_model_stopped"] == 6
    assert "skipped_provider_stopped" not in summary
    assert (
        len(alerts) == 1 and "claude-sonnet-5-5" in alerts[0] and "resume" in alerts[0]
    )


# ------------------------------------------------------------------ score
def test_replies_are_scored_with_the_real_scorers(conn):
    def plan(provider, model, prompt, n):
        wrong = {M1["prompt"]: "5", M2["prompt"]: "9", I1["prompt"]: "only two"}
        return result(wrong[prompt])

    run_id, _ = run_collect(conn, Caller(plan), models=["haiku"], repeats=1)
    assert score(conn, run_id) == {"scored": 3}
    by_probe = dict(
        conn.execute(
            "SELECT p.slug, s.score FROM scores s JOIN calls c ON c.call_id = s.call_id JOIN probes p ON p.probe_id = c.probe_id"
        ).fetchall()
    )
    assert by_probe == {"math_001": 0.0, "math_002": 1.0, "instr_001": 0.0}
    assert (
        json.loads(conn.execute("SELECT details FROM scores LIMIT 1").fetchone()[0])
        is not None
    )


def test_failed_and_cut_off_replies_are_not_scored(conn):
    def plan(provider, model, prompt, n):
        if n == 1:
            return result(
                None,
                error="timeout: slow",
                http_status=None,
                finish_reason=None,
            )
        if n == 2:
            return result("4", finish_reason="max_tokens")
        return result(ANSWERS[prompt])

    run_id, _ = run_collect(conn, Caller(plan), models=["haiku"], repeats=1)
    assert score(conn, run_id) == {"skipped_truncated": 1, "scored": 1}
    assert count(conn, "SELECT COUNT(*) FROM scores") == 1


def test_openai_style_cut_off_is_also_recognised():
    assert is_truncated("max_tokens") and is_truncated("incomplete:max_output_tokens")
    assert (
        not is_truncated("end_turn")
        and not is_truncated("completed")
        and not is_truncated(None)
    )


def test_scoring_twice_adds_nothing_and_rescoring_replaces(conn):
    run_id, _ = run_collect(conn, Caller(), models=["haiku"], repeats=1)
    assert score(conn, run_id) == {"scored": 3}
    assert score(conn, run_id) == {}  # nothing left to score
    always_fail = lambda name: lambda probe, text: (False, {"reason": "forced"})
    assert score(conn, run_id, rescore=True, get_scorer_fn=always_fail) == {"scored": 3}
    assert count(conn, "SELECT COUNT(*) FROM scores") == 3  # replaced, not doubled
    assert count(conn, "SELECT COUNT(*) FROM scores WHERE score = 0") == 3


def test_an_unavailable_scorer_leaves_its_replies_unscored_and_the_rest_carry_on(conn):
    from src.scorer import get_scorer

    def sandbox_down(name):
        if name == "rule_check":
            raise_down = RuntimeError("docker failed (exit 125): Cannot connect")

            def broken(probe, text):
                raise raise_down

            return broken
        return get_scorer(name)

    alerts = []
    run_id, _ = run_collect(conn, Caller(), models=["haiku"], repeats=3)
    summary = score(conn, run_id, get_scorer_fn=sandbox_down, notify=alerts.append)
    assert summary == {"scored": 6, "skipped_scorer_unavailable": 3}
    assert len(alerts) == 1 and "rule_check" in alerts[0]
    assert score(conn, run_id) == {
        "scored": 3
    }  # later, with the sandbox back, they get scored


# ------------------------------------------------------------------ report and selection
def test_report_flags_probes_that_every_model_always_passes_or_always_fails(conn):
    answers = {
        M1["prompt"]: "4",
        M2["prompt"]: "0",
        I1["prompt"]: "red green blue",
    }  # math_002 is always wrong
    run_id, _ = run_collect(
        conn, Caller(lambda p, m, prompt, n: result(answers[prompt])), repeats=3
    )
    score(conn, run_id)
    lines = []
    flags = report(conn, run_id, out=lines.append)
    assert flags["always_pass"] == ["instr_001", "math_001"]
    assert flags["always_fail"] == ["math_002"]
    table = "\n".join(lines)
    assert (
        "claude-haiku-5-5" in table
        and "gpt-6-luna" in table
        and "100%" in table
        and "0%" in table
    )
    assert "largest single reply 3 output tokens" in table


def test_a_probe_that_only_one_model_gets_right_is_not_flagged(conn):
    def plan(provider, model, prompt, n):
        return result(ANSWERS[prompt] if provider == "anthropic" else "wrong")

    run_id, _ = run_collect(conn, Caller(plan), repeats=1)
    score(conn, run_id)
    flags = report(conn, run_id, out=lambda line: None)
    assert flags == {"always_pass": [], "always_fail": []}


def test_selection_filters(conn):
    assert [r[1] for r in select_probes(conn)] == ["math_001", "math_002", "instr_001"]
    assert [r[1] for r in select_probes(conn, category="math")] == [
        "math_001",
        "math_002",
    ]
    assert [r[1] for r in select_probes(conn, per_category=1)] == [
        "math_001",
        "instr_001",
    ]
    assert select_probes(conn, split="scheduled") == []  # nothing has a split yet


# ------------------------------------------------------------------ the whole chain, through the real SDKs
def test_end_to_end_through_the_real_sdks_with_a_fake_network(conn, tmp_path):
    import json as _json
    from pathlib import Path

    import anthropic
    import httpx2
    import openai

    from src.providers.clients import call_model

    fixtures = Path(__file__).parent / "fixtures"
    anthropic_body = _json.loads((fixtures / "anthropic.json").read_text())
    openai_body = _json.loads((fixtures / "openai.json").read_text())
    probe = dict(
        M1,
        id="math_391",
        prompt="What is 17 * 23? Reply with only the number.",
        expected=391,
    )
    path = tmp_path / "math.jsonl"
    path.write_text(_json.dumps(probe) + "\n", encoding="utf-8")
    load_probes(conn, [path])

    clients = {
        "anthropic": anthropic.Anthropic(
            api_key="test", max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=anthropic_body))),
        ),
        "openai": openai.OpenAI(
            api_key="test", max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=openai_body))),
        ),
    }  # fmt: skip
    probes = [row for row in select_probes(conn) if row[1] == "math_391"]
    run_id = start_run(conn, {"models": BOTH})
    summary = collect(
        conn,
        run_id,
        BOTH,
        repeats=1,
        probes=probes,
        caller=call_model,
        clients=clients,
        notify=lambda m: None,
    )
    assert summary == {"ok": 2}

    rows = conn.execute(
        "SELECT provider, received_string, raw_output, input_tokens, output_tokens, reasoning_tokens, finish_reason, raw_response FROM calls ORDER BY provider"
    ).fetchall()
    assert [r[:7] for r in rows] == [
        ("anthropic", "claude-haiku-5-5", "391", 25, 3, 0, "end_turn"),
        ("openai", "gpt-6-luna", "391", 20, 24, 17, "completed"),
    ]
    assert (
        "<removed" in rows[1][7]
    )  # the opaque reasoning blob never reaches the database
    assert score(conn, run_id) == {"scored": 2}
    assert count(conn, "SELECT COUNT(*) FROM scores WHERE score = 1.0") == 2
