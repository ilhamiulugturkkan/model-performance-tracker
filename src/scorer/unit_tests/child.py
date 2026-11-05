import json
import sys

MARKER = "@@RESULT@@"

# The parent sends one JSON object on stdin: {"entry_point", "tests", "reply"}.
# The parent has already checked entry_point and tests, so no config checks here:
# a crash in this process always means the candidate code did it.
payload = json.load(sys.stdin)
CANDIDATE = payload["reply"]
TESTS = payload["tests"]
ENTRY_POINT = payload["entry_point"]

result = {"loaded": False, "failed": [], "error": None, "load_error": None}
ns = {}

try:
    exec(CANDIDATE, ns)  # noqa: S102 - running the candidate is this file's whole job
    result["loaded"] = True
except BaseException as e:  # noqa: BLE001 - candidate code can raise SystemExit
    result["error"] = f"{type(e).__name__}: {e}"
    result["load_error"] = f"{type(e).__name__}: {e}"

if not callable(ns.get(ENTRY_POINT)):
    result["loaded"] = False
    result["error"] = "Different entry points"

if result["loaded"]:
    for line in TESTS:
        if line.strip():
            try:
                exec(line, ns)  # noqa: S102 - running the candidate is this file's whole job
            except BaseException:  # noqa: BLE001 - any failure, even SystemExit, fails the test
                result["failed"].append(line)

# Leading newline: if the candidate printed without a trailing newline, the marker
# still starts its own line, so the parent's startswith check holds.
print("\n" + MARKER + json.dumps(result))
