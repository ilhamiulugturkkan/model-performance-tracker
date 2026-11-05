import re

from src.scorer.unit_tests.code_scorer import _check_probe, run_child

# A model usually wraps code in a markdown fence. We accept that and log it.
FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(reply, entry_point):
    """Return (code, fenced). Prefer the block that mentions the entry point."""
    blocks = FENCE.findall(reply)
    if not blocks:
        return reply, False
    for block in blocks:
        if entry_point in block:
            return block, True
    return blocks[0], True


def _reason(result):
    """Turn the sandbox result into one fixed word, so causes can be counted later."""
    error = result.get("error")
    if error == "TIMEOUT":
        return "timeout"
    if error == "NO_RESULT":
        return "no_result"
    if not result.get("loaded"):
        # The code failed to load (syntax error, stray exit) or the function is missing.
        return "did_not_load" if result.get("load_error") else "missing_function"
    if result.get("failed"):
        return "tests_failed"
    return "passed"


def unit_tests(probe, reply, runner=run_child):
    """Score a coding reply. `runner` lets tests replace the sandbox with a stand-in."""
    # A broken probe is our mistake, not the model's, so it raises before anything else.
    _check_probe(probe)
    if not isinstance(reply, str):
        return False, {"reason": "no_reply"}

    code, fenced = extract_code(reply, probe["entry_point"])
    result = runner(probe, code)

    reason = _reason(result)
    # Every failure path above sets loaded=False, so this one rule is enough.
    passed = bool(result.get("loaded")) and not result.get("failed")
    details = {"reason": reason, "fenced": fenced}
    if result.get("failed"):
        details["failed_tests"] = result["failed"]
    if result.get("load_error"):
        details["load_error"] = result["load_error"]
    return passed, details
