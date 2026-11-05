import math

from src.scorer.common import no_reply


def numeric_match(probe, reply):
    expected = probe.get("expected")
    tolerance = probe.get("tolerance")
    # Throw error if one of em is None
    if expected is None or tolerance is None:
        raise ValueError("Expected value or tolerance is None.")

    if reply is None:
        return no_reply()

    upper = expected + tolerance
    lower = expected - tolerance

    # convert the reply to a numeric value, care for the minus sign and other edge cases
    text = reply.strip().replace("−", "-")
    try:
        value = float(text)
    except ValueError:
        return False, {"reason": "Reply is not a valid numeric value.", "value": None}
    # "nan" / "inf" parse as floats but are not answers (and are not valid JSON for the details column)
    if not math.isfinite(value):
        return False, {"reason": "Reply is not a valid numeric value.", "value": None}

    passed = lower <= value <= upper
    where = "within" if passed else "outside"
    return passed, {
        "reason": f"Reply value {value} is {where} the range {lower} to {upper}.",
        "value": value,
    }
