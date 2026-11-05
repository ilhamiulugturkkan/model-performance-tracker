import json
import subprocess
import uuid

IMAGE = "watchdog-sandbox"
MARKER = "@@RESULT@@"
STDERR_TAIL_CHARS = 500
DEFAULT_TIMEOUT_SECONDS = 5
DOCKER_FAILURE_EXIT_CODES = (125, 126, 127)


def _error_result(error, load_error=None):
    # Fresh dict on every call so results are never shared between runs.
    return {"loaded": False, "failed": [], "error": error, "load_error": load_error}


def _check_probe(probe):
    entry_point = probe.get("entry_point")
    if not isinstance(entry_point, str) or not entry_point:
        raise ValueError(
            "Invalid probe configuration: entry_point must be a non-empty string"
        )
    tests = probe.get("tests")
    if (
        not isinstance(tests, list)
        or not tests
        or not all(isinstance(t, str) for t in tests)
    ):
        raise ValueError(
            "Invalid probe configuration: tests must be a non-empty list of strings"
        )


def _docker_command(name, timeout):
    return [
        "docker",
        "run",
        "-i",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--memory",
        "256m",
        "--cpus",
        "1",
        "--pids-limit",
        "64",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65534:65534",
        IMAGE,
        "timeout",
        "-k",
        "1",
        str(timeout),
        "python",
        "-I",
        "/child.py",
    ]


def run_child(probe, reply):
    _check_probe(probe)
    run_name = "probe-" + uuid.uuid4().hex[:12]
    timeout = probe.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)

    # Send only what the child needs. json.dumps escapes lone surrogates (\ud800),
    # so the payload is plain ASCII and the stdin encoding never matters.
    payload = {
        "entry_point": probe["entry_point"],
        "tests": probe["tests"],
        "reply": reply,
    }
    try:
        result = subprocess.run(
            _docker_command(run_name, timeout),
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout + 3,
            check=False,  # we read the exit code ourselves
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", run_name], capture_output=True, check=False)
        return _error_result("TIMEOUT")

    # No usable verdict: the child crashed, ran out of memory, or exited early.
    stderr_tail = result.stderr[-STDERR_TAIL_CHARS:].strip()

    if result.returncode in DOCKER_FAILURE_EXIT_CODES:
        raise RuntimeError(f"docker failed (exit {result.returncode}): {stderr_tail}")

    # The child prints MARKER + json on its own line. Take the last such line,
    # ignoring anything the candidate code printed before or after it.
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(MARKER):
            try:
                return json.loads(line[len(MARKER) :])
            except json.JSONDecodeError:
                break

    if result.returncode == 124:
        return _error_result("TIMEOUT")

    return _error_result(
        "NO_RESULT",
        load_error=f"returncode={result.returncode}: {stderr_tail}",
    )


def _main():
    """Developer tool: run one reply against one probe, for trying things by hand.

    python3 src/scorer/unit_tests/code_scorer.py coding_001 reply.txt
    echo 'while True: pass' | python3 src/scorer/unit_tests/code_scorer.py coding_001
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=_main.__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("probe_id", help="for example coding_001")
    parser.add_argument(
        "reply_file", nargs="?", help="file holding the reply (default: read stdin)"
    )
    args = parser.parse_args()

    probes_file = Path(__file__).resolve().parents[3] / "probes" / "coding.jsonl"
    probes = [
        json.loads(line)
        for line in probes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    probe = next((p for p in probes if p["id"] == args.probe_id), None)
    if probe is None:
        sys.exit(f"no probe {args.probe_id!r} in {probes_file}")

    reply = (
        Path(args.reply_file).read_text(encoding="utf-8")
        if args.reply_file
        else sys.stdin.read()
    )
    print(run_child(probe, reply))


if __name__ == "__main__":
    _main()
