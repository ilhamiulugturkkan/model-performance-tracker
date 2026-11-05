import json
import re

from src.scorer.common import no_reply

FENCE = re.compile(r"\s*```(?:json)?\s*(.*?)\s*```\s*", re.DOTALL)


def strip_fence(reply):
    """Return (text, fenced). Only a reply that is entirely one fenced block is unwrapped."""
    m = FENCE.fullmatch(reply)
    return (m.group(1), True) if m else (reply, False)


def json_match(probe, reply):
    expected_json = probe.get("expected")
    if reply is None:
        return no_reply()
    # first transform reply to a parsable json
    text, fenced = strip_fence(reply)

    # Design choice: a code fence is accepted, and logged as the queryable field details["fenced"].
    def result(passed, reason):
        return passed, {"reason": reason, "fenced": fenced}

    try:
        reply_json = json.loads(text)
    except json.JSONDecodeError:
        return result(False, "Reply is not a valid JSON string.")

    if not isinstance(reply_json, dict):
        return result(False, "Reply JSON is not a dictionary.")

    if not probe.get("extra_keys_allowed") and set(reply_json.keys()) - set(
        expected_json.keys()
    ):
        # if there's extra field immediately reject
        return result(False, "Reply JSON contains extra keys that are not allowed.")

    tolerance = probe.get("tolerance")
    if tolerance == 0:
        # in both keys, we compare the keys that are in expected_json
        for key in expected_json:
            if key not in reply_json:
                return result(False, f"Reply JSON is missing the key '{key}'.")
            if expected_json[key] != reply_json.get(key):
                return result(
                    False,
                    f"Reply JSON does not match the expected JSON for key '{key}'.",
                )
    else:
        # numeric values may differ within the tolerance
        for key in expected_json:
            if key not in reply_json:
                return result(False, f"Reply JSON is missing the key '{key}'.")
            if isinstance(expected_json[key], (int, float)) and isinstance(
                reply_json[key], (int, float)
            ):
                # care for int - float case
                expected_value = float(expected_json[key])
                reply_value = float(reply_json[key])
                if abs(expected_value - reply_value) > tolerance:
                    return result(
                        False,
                        f"Reply JSON does not match the expected JSON within tolerance for key '{key}'.",
                    )
            else:
                if expected_json[key] != reply_json[key]:
                    return result(
                        False,
                        f"Reply JSON does not match the expected JSON for key '{key}'.",
                    )
    return result(True, "Reply JSON matches the expected JSON.")
