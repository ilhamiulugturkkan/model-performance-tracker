from src.scorer.json_match import json_match
from src.scorer.numeric_match import numeric_match
from src.scorer.rule_check import rule_check

SCORERS = {
    "numeric_match": numeric_match,
    "json_match": json_match,
    "rule_check": rule_check,
}


def get_scorer(name):
    check = SCORERS.get(name)
    if check is None:
        raise ValueError(f"unknown rule key {name!r}")
    return check
