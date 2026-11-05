def no_reply():
    """Result for a call that returned nothing (None): NOT scored, so passed is None, not False.

    The caller records the failure in calls.error and writes no scores row.
    Pass rates are computed over calls that returned a reply; the error rate is tracked separately.
    Every scorer's details is a dict, so the JSON column has one shape (see docs/decisions.md D11).
    """
    return None, {"reason": "no reply"}
