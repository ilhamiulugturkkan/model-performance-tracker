"""Collect model replies, score them, and report pass rates.

    python3 -m src.runner run --db mpt_test.db --models haiku luna etc. --repeats 3
    python3 -m src.runner resume --db mpt_test.db --run-id 1     # after a crash or an empty balance
    python3 -m src.runner score --db mpt_test.db --run-id 1       # score whatever is not scored yet
    python3 -m src.runner report --db mpt_test.db --run-id 1

Two phases, kept apart on purpose: COLLECT saves the raw replies (it costs money), SCORE reads them
from the database (it is free and can be redone whenever a scorer changes).
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime

from src.alerts import send_telegram
from src.db.connection import DB_PATH, connect
from src.providers.clients import call_model, make_client
from src.providers.models import MODELS, parameters_record
from src.scorer import get_scorer

NOTIFIERS = {
    "print": print,
    "telegram": send_telegram,
}  # --notify takes one of these names
FAILURE_LIMIT = 3  # failed calls in a row (each after every retry) before a provider is given up on for this run


def utc_now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_commit():
    """The commit this run was made from, with +dirty when there are uncommitted changes."""
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        sha = subprocess.run(
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [git, "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def start_run(conn, config):
    cursor = conn.execute(
        "INSERT INTO runs (started_at, git_commit, config) VALUES (?, ?, ?)",
        (utc_now(), git_commit(), json.dumps(config)),
    )
    conn.commit()
    return cursor.lastrowid


def select_probes(conn, category=None, split=None, per_category=None):
    rows = conn.execute(
        "SELECT probe_id, slug, category, body FROM probes "
        "WHERE body IS NOT NULL AND (? IS NULL OR category = ?) AND (? IS NULL OR split = ?) "
        "ORDER BY probe_id",
        (category, category, split, split),
    ).fetchall()
    if per_category:
        taken, kept = Counter(), []
        for row in rows:
            if taken[row[2]] < per_category:
                kept.append(row)
                taken[row[2]] += 1
        rows = kept
    return rows


def _already_done(conn, run_id, probe_id, provider, model, repeat):
    row = conn.execute(
        "SELECT error FROM calls WHERE run_id = ? AND probe_id = ? AND provider = ? AND request_string = ? AND repeat_no = ?",
        (run_id, probe_id, provider, model, repeat),
    ).fetchone()
    return (
        row is not None and row[0] is None
    )  # a failed call is tried again; a good one never is


def _save_call(conn, run_id, probe_id, prompt, config, repeat, started, result):
    conn.execute(
        """
        INSERT INTO calls (run_id, probe_id, provider, request_string, received_string, parameters, raw_prompt,
                           raw_response, raw_output, input_tokens, output_tokens, latency, http_status, error,
                           time_stamp, repeat_no, attempts, finish_reason, reasoning_tokens, request_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id, probe_id, provider, request_string, repeat_no) DO UPDATE SET
            received_string = excluded.received_string, parameters = excluded.parameters,
            raw_prompt = excluded.raw_prompt, raw_response = excluded.raw_response,
            raw_output = excluded.raw_output, input_tokens = excluded.input_tokens,
            output_tokens = excluded.output_tokens, latency = excluded.latency,
            http_status = excluded.http_status, error = excluded.error, time_stamp = excluded.time_stamp,
            attempts = excluded.attempts, finish_reason = excluded.finish_reason,
            reasoning_tokens = excluded.reasoning_tokens, request_id = excluded.request_id
        WHERE calls.error IS NOT NULL
        """,
        (
            run_id, probe_id, config["provider"], config["model"], result["returned_model"],
            json.dumps(parameters_record(config)), prompt, result["raw_response"], result["text"],
            result["input_tokens"], result["output_tokens"], result["latency"], result["http_status"],
            result["error"], started, repeat, result["attempts"], result["finish_reason"],
            result["reasoning_tokens"], result["request_id"],
        ),
    )  # fmt: skip
    conn.commit()


class FailureTracker:
    """Counts failed calls in a row, per provider. A success starts the count again."""

    def __init__(self, limit=FAILURE_LIMIT):
        self.limit = limit
        self.streak = Counter()

    def failed(self, provider):
        """Record a failure. True exactly once: when the streak reaches the limit."""
        self.streak[provider] += 1
        return self.streak[provider] == self.limit

    def succeeded(self, provider):
        self.streak[provider] = 0


def collect(
    conn,
    run_id,
    model_keys,
    repeats=3,
    probes=None,
    caller=call_model,
    clients=None,
    notify=print,
    failure_limit=FAILURE_LIMIT,
):
    """Send every probe to every model, `repeats` times, and save each reply. Safe to run again."""
    clients = {} if clients is None else clients
    probes = select_probes(conn) if probes is None else probes
    stopped = (
        set()
    )  # providers we stop calling: empty balance, or too many failures in a row
    tracker = FailureTracker(failure_limit)
    summary = Counter()
    for probe_id, _slug, _category, body in probes:
        probe = json.loads(body)
        for key in model_keys:
            config = MODELS[key]
            provider, model = config["provider"], config["model"]
            for repeat in range(1, repeats + 1):
                if provider in stopped:
                    summary["skipped_provider_stopped"] += 1
                    continue
                if _already_done(conn, run_id, probe_id, provider, model, repeat):
                    summary["already_done"] += 1
                    continue
                if provider not in clients:
                    clients[provider] = make_client(provider)
                started = utc_now()
                result = caller(
                    clients[provider],
                    provider,
                    model,
                    probe["prompt"],
                    config["settings"],
                )
                _save_call(
                    conn,
                    run_id,
                    probe_id,
                    probe["prompt"],
                    config,
                    repeat,
                    started,
                    result,
                )
                summary["failed" if result["error"] else "ok"] += 1
                if not result["error"]:
                    tracker.succeeded(provider)
                elif result["error"].startswith("quota_exhausted"):
                    stopped.add(provider)
                    notify(
                        f"STOP: {provider} balance is empty ({result['error'][:120]}). Top up, then: resume --run-id {run_id}"
                    )
                elif tracker.failed(provider):
                    stopped.add(provider)
                    notify(
                        f"STOP: {provider} failed {failure_limit} calls in a row, each after every retry "
                        f"(last: {result['error'][:120]}). Fix it, then: resume --run-id {run_id}"
                    )
    if (
        summary["failed"] and not stopped
    ):  # a stopped provider has already sent its own alert
        notify(
            f"run {run_id}: {summary['failed']} of {summary['failed'] + summary['ok']} calls failed after every retry"
        )
    return dict(summary)


def is_truncated(finish_reason):
    """The reply was cut off by the token limit. That is our limit's fault, not a wrong answer."""
    return finish_reason == "max_tokens" or (finish_reason or "").startswith(
        "incomplete"
    )


def score(conn, run_id=None, rescore=False, get_scorer_fn=get_scorer, notify=print):
    """Score every reply that has no score yet. A re-score REPLACES a score, it never adds one."""
    rows = conn.execute(
        """
        SELECT c.call_id, c.raw_output, c.finish_reason, p.body
        FROM calls c
        JOIN probes p ON p.probe_id = c.probe_id
        LEFT JOIN scores s ON s.call_id = c.call_id
        WHERE c.error IS NULL AND c.raw_output IS NOT NULL
          AND (? IS NULL OR c.run_id = ?)
          AND (? = 1 OR s.score_id IS NULL)
        ORDER BY c.call_id
        """,
        (run_id, run_id, int(rescore)),
    ).fetchall()
    summary = Counter()
    unavailable = set()
    for call_id, text, finish_reason, body in rows:
        probe = json.loads(body)
        name = probe["scorer"]
        if is_truncated(finish_reason):
            summary["skipped_truncated"] += 1
            continue
        if name in unavailable:
            summary["skipped_scorer_unavailable"] += 1
            continue
        try:
            passed, details = get_scorer_fn(name)(probe, text)
        except RuntimeError as error:  # the sandbox is down: our setup, not the model
            unavailable.add(name)
            notify(
                f"SCORER UNAVAILABLE: {name} ({str(error)[:150]}). Its replies stay unscored; run `score` again later."
            )
            summary["skipped_scorer_unavailable"] += 1
            continue
        conn.execute(
            "INSERT INTO scores (call_id, scorer, score, details) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (call_id, scorer) DO UPDATE SET score = excluded.score, details = excluded.details",
            (call_id, name, 1.0 if passed else 0.0, json.dumps(details)),
        )
        conn.commit()
        summary["scored"] += 1
    return dict(summary)


def report(conn, run_id, out=print):
    rows = conn.execute(
        """
        SELECT p.slug, p.category, c.request_string, c.error, c.finish_reason, s.score
        FROM calls c
        JOIN probes p ON p.probe_id = c.probe_id
        LEFT JOIN scores s ON s.call_id = c.call_id
        WHERE c.run_id = ?
        ORDER BY p.category, p.slug, c.request_string
        """,
        (run_id,),
    ).fetchall()
    cells = defaultdict(Counter)
    for slug, category, model, error, finish_reason, scored in rows:
        cell = cells[(category, slug, model)]
        cell["calls"] += 1
        if error:
            cell["errors"] += 1
        elif is_truncated(finish_reason):
            cell["cut"] += 1
        elif scored is not None:
            cell["scored"] += 1
            cell["passed"] += int(scored)

    out(
        f"{'probe':14} {'model':18} {'scored':>6} {'passed':>6} {'rate':>5} {'err':>4} {'cut':>4}"
    )
    verdicts = defaultdict(list)
    for (category, slug, model), cell in sorted(cells.items()):
        rate = f"{100 * cell['passed'] // cell['scored']}%" if cell["scored"] else "-"
        out(
            f"{slug:14} {model:18} {cell['scored']:>6} {cell['passed']:>6} {rate:>5} {cell['errors']:>4} {cell['cut']:>4}"
        )
        if cell["scored"]:
            verdicts[slug].append(cell["passed"] / cell["scored"])

    always_pass = sorted(
        s for s, rates in verdicts.items() if all(r == 1.0 for r in rates)
    )
    always_fail = sorted(
        s for s, rates in verdicts.items() if all(r == 0.0 for r in rates)
    )
    out(
        f"\nalways PASS on every model (no signal, candidates to drop): {len(always_pass)}  {' '.join(always_pass)}"
    )
    out(
        f"always FAIL on every model (check the answer key first):    {len(always_fail)}  {' '.join(always_fail)}"
    )

    out("\nper model:")
    for model, calls, errors, in_tokens, out_tokens, max_out in conn.execute(
        "SELECT request_string, COUNT(*), SUM(error IS NOT NULL), SUM(input_tokens), SUM(output_tokens), MAX(output_tokens) "
        "FROM calls WHERE run_id = ? GROUP BY request_string",
        (run_id,),
    ):
        out(
            f"  {model}: {calls} calls, {errors} failed, tokens in {in_tokens or 0} / out {out_tokens or 0}, largest single reply {max_out or 0} output tokens"
        )
    return {"always_pass": always_pass, "always_fail": always_fail}


def _connect_and_check(path):
    conn = connect(path)
    if (
        conn.execute("SELECT COUNT(*) FROM probes WHERE body IS NOT NULL").fetchone()[0]
        == 0
    ):
        sys.exit("no probes in this database: run the loader first")
    return conn


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["run", "resume", "score", "report"])
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS)
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--category")
    parser.add_argument("--split", choices=["scheduled", "heldout"])
    parser.add_argument(
        "--per-category",
        type=int,
        help="only the first N probes of each category (a cheap trial)",
    )
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument(
        "--notify",
        choices=sorted(NOTIFIERS),
        default="print",
        help="where alerts go: the terminal, or Telegram (needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)",
    )
    return parser


def _main():
    args = build_parser().parse_args()
    notify = NOTIFIERS[args.notify]

    conn = _connect_and_check(args.db)  # a database that exists and has probes in it
    if args.command in ("resume", "score", "report") and args.run_id is None:
        sys.exit(
            "--run-id is required"
        )  # these commands work on a run that already exists

    if args.command == "run":
        # a new run gets its own id; its config says which models, how many repeats, which probes
        config = {"models": args.models, "repeats": args.repeats, "category": args.category,
                  "split": args.split, "per_category": args.per_category}  # fmt: skip
        run_id = start_run(conn, config)
        print(f"run {run_id}")
    elif args.command == "resume":
        # a resumed run uses the config it started with
        config = json.loads(
            conn.execute(
                "SELECT config FROM runs WHERE run_id = ?", (args.run_id,)
            ).fetchone()[0]
        )
        run_id = args.run_id
    else:
        run_id = args.run_id

    if args.command in ("run", "resume"):
        probes = select_probes(
            conn, config["category"], config["split"], config["per_category"]
        )
        print(
            "collect:",
            collect(
                conn, run_id, config["models"], config["repeats"], probes, notify=notify
            ),
        )
    if args.command in ("run", "resume", "score"):
        print("score:", score(conn, run_id, rescore=args.rescore, notify=notify))
    report(conn, run_id)


if __name__ == "__main__":
    _main()
