import ast
import collections
import json
import re
import sys
from pathlib import Path

CATEGORY_PREFIX = {
    "math": "math_",
    "instructions": "instr_",
    "structured": "struct_",
    "coding": "coding_",
}
# Which scorers each category may use (extend this when you add e.g. SQL probes).
CATEGORY_SCORERS = {
    "math": {"numeric_match"},
    "instructions": {"rule_check"},
    "structured": {"json_match"},
    "coding": {"unit_tests"},
}
SPLITS = {None, "scheduled", "heldout"}
MIN_PROBES_PER_CATEGORY = 14  # warning only

COMMON_REQUIRED = {"id", "version", "category", "split", "prompt", "scorer"}
COMMON_OPTIONAL = {"topic"}

# scorer name -> (required extra fields, optional extra fields)
SCORER_FIELDS = {
    "numeric_match": ({"expected", "tolerance"}, set()),
    "json_match": ({"expected", "tolerance", "extra_keys_allowed"}, set()),
    "rule_check": ({"rules"}, set()),
    "unit_tests": ({"entry_point", "tests", "timeout_seconds"}, set()),
}


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_letter(v):
    return isinstance(v, str) and len(v) == 1 and v.isalpha()


def is_text(v):
    return isinstance(v, str) and v.strip() != ""


def list_of(check):
    return lambda v: isinstance(v, list) and len(v) > 0 and all(check(x) for x in v)


# rule key -> function that returns True when the value is valid
RULE_KEYS = {
    "sentences": lambda v: is_int(v) and v > 0,
    "words": lambda v: is_int(v) and v > 0,
    "unique_words": lambda v: isinstance(v, bool),
    "letters_only": lambda v: isinstance(v, bool),
    "each_word_contains": is_letter,
    "each_word_starts_with": is_letter,
    "min_word_length": lambda v: is_int(v) and v > 0,
    "forbidden_letters": list_of(is_letter),
    "forbidden_words": list_of(is_text),
    "required_words": list_of(is_text),
    "words_from_list": list_of(is_text),
    "uppercase_words": list_of(is_text),
    "list_items": lambda v: is_int(v) and v > 0,
    "digit_count": lambda v: is_int(v) and v >= 0,
    "starts_with": is_text,
    "ends_with": is_text,
}


def _reject_duplicate_keys(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key '{key}'")
        seen[key] = value
    return seen


def load_lines(path):
    """Yield (line_number, object, error). Exactly one of object/error is None."""
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield (
                    n,
                    json.loads(line, object_pairs_hook=_reject_duplicate_keys),
                    None,
                )
            except (json.JSONDecodeError, ValueError) as e:
                yield n, None, f"invalid JSON: {e}"


def check_common(obj, filename):
    """Layer 1: fields every probe must have. Returns a list of error strings."""
    if not isinstance(obj, dict):
        return ["line is not a JSON object"]
    errors = []

    missing = COMMON_REQUIRED - obj.keys()
    if missing:
        errors.append(f"missing field(s): {sorted(missing)}")

    scorer = obj.get("scorer")
    extra_req, extra_opt = SCORER_FIELDS.get(scorer, (set(), set()))
    allowed = COMMON_REQUIRED | COMMON_OPTIONAL | extra_req | extra_opt
    unknown = obj.keys() - allowed
    if unknown:
        errors.append(f"unknown field(s): {sorted(unknown)}")

    category = obj.get("category")
    file_category = Path(filename).stem
    if category not in CATEGORY_PREFIX:
        errors.append(
            f"category must be one of {sorted(CATEGORY_PREFIX)}, got {category!r}"
        )
    elif file_category in CATEGORY_PREFIX and category != file_category:
        errors.append(
            f"category {category!r} does not match file name {file_category!r}"
        )

    probe_id = obj.get("id")
    if not isinstance(probe_id, str) or not re.fullmatch(r"[a-z]+_\d{3}", probe_id):
        errors.append(f"id must look like 'math_001', got {probe_id!r}")
    elif category in CATEGORY_PREFIX and not probe_id.startswith(
        CATEGORY_PREFIX[category]
    ):
        errors.append(f"id {probe_id!r} must start with {CATEGORY_PREFIX[category]!r}")

    version = obj.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"v\d+", version):
        errors.append(f"version must look like 'v1', got {version!r}")

    if "split" in obj and obj["split"] not in SPLITS:
        errors.append(
            f"split must be null, 'scheduled' or 'heldout', got {obj['split']!r}"
        )

    if "prompt" in obj and not is_text(obj["prompt"]):
        errors.append("prompt must be a non-empty string")

    if "topic" in obj and not (
        isinstance(obj["topic"], str) and re.fullmatch(r"[a-z][a-z_]*", obj["topic"])
    ):
        errors.append(f"topic must be lowercase snake_case, got {obj['topic']!r}")

    if scorer is not None and scorer not in SCORER_FIELDS:
        errors.append(f"unknown scorer {scorer!r}; known: {sorted(SCORER_FIELDS)}")
    elif category in CATEGORY_SCORERS and scorer not in CATEGORY_SCORERS[category]:
        errors.append(f"category {category!r} cannot use scorer {scorer!r}")

    return errors


def check_scorer(obj):
    """Layer 2: fields that depend on the scorer. Returns a list of error strings."""
    if not isinstance(obj, dict):
        return []
    scorer = obj.get("scorer")
    if scorer not in SCORER_FIELDS:
        return []  # already reported by check_common
    errors = []
    required, _ = SCORER_FIELDS[scorer]
    missing = required - obj.keys()
    if missing:
        errors.append(f"{scorer}: missing field(s) {sorted(missing)}")

    if scorer in ("numeric_match", "json_match"):
        tol = obj.get("tolerance")
        if "tolerance" in obj and not (is_number(tol) and tol >= 0):
            errors.append("tolerance must be a number >= 0")

    if (
        scorer == "numeric_match"
        and "expected" in obj
        and not is_number(obj["expected"])
    ):
        errors.append("numeric_match: expected must be a number")

    if scorer == "json_match":
        if "expected" in obj and not isinstance(obj["expected"], dict):
            errors.append("json_match: expected must be a JSON object")
        if "extra_keys_allowed" in obj and not isinstance(
            obj["extra_keys_allowed"], bool
        ):
            errors.append("json_match: extra_keys_allowed must be true or false")

    if scorer == "rule_check" and "rules" in obj:
        rules = obj["rules"]
        if not isinstance(rules, dict) or not rules:
            errors.append("rule_check: rules must be a non-empty object")
        else:
            for key, value in rules.items():
                if key not in RULE_KEYS:
                    errors.append(f"rule_check: unknown rule key {key!r}")
                elif not RULE_KEYS[key](value):
                    errors.append(f"rule_check: bad value for rule {key!r}: {value!r}")

    if scorer == "unit_tests":
        entry = obj.get("entry_point")
        tests = obj.get("tests")
        if "entry_point" in obj and not (
            isinstance(entry, str) and entry.isidentifier()
        ):
            errors.append("unit_tests: entry_point must be a valid Python name")
        if "tests" in obj:
            if not list_of(is_text)(tests):
                errors.append(
                    "unit_tests: tests must be a non-empty list of non-empty strings"
                )
            else:
                for i, test in enumerate(tests):
                    try:
                        ast.parse(test)
                    except SyntaxError as e:
                        errors.append(
                            f"unit_tests: tests[{i}] is not valid Python: {e.msg}"
                        )
                if isinstance(entry, str) and not any(entry in t for t in tests):
                    errors.append("unit_tests: entry_point does not appear in tests")
        if (
            isinstance(entry, str)
            and isinstance(obj.get("prompt"), str)
            and entry not in obj["prompt"]
        ):
            errors.append("unit_tests: entry_point does not appear in the prompt")
        timeout = obj.get("timeout_seconds")
        if "timeout_seconds" in obj and not (is_int(timeout) and 1 <= timeout <= 30):
            errors.append("unit_tests: timeout_seconds must be an integer from 1 to 30")

    return errors


def check_set(entries):
    """Layer 3: checks across all files.

    entries is a list of (path, line_number, object). Returns (errors, warnings, summary_lines).
    """
    errors, warnings, summary = [], [], []
    seen = {}
    for path, n, obj in entries:
        key = (obj.get("id"), obj.get("version"))
        if key in seen:
            errors.append(
                f"{path}:{n}: duplicate (id, version) {key}, first seen at {seen[key]}"
            )
        else:
            seen[key] = f"{path}:{n}"

    by_category = collections.Counter(o.get("category") for _, _, o in entries)
    for category, count in sorted(by_category.items(), key=lambda kv: str(kv[0])):
        summary.append(f"{category}: {count} probes")
        if count < MIN_PROBES_PER_CATEGORY:
            warnings.append(
                f"category {category!r} has {count} probes (fewer than {MIN_PROBES_PER_CATEGORY}); "
                "its detection floor will be coarse"
            )
        topics = collections.Counter(
            o.get("topic") for _, _, o in entries if o.get("category") == category
        )
        top_topic, top_count = topics.most_common(1)[0]
        if count >= 5 and top_count / count > 0.4:
            warnings.append(
                f"category {category!r}: topic {top_topic!r} is {top_count} of {count} probes"
            )
    return errors, warnings, summary


def main(paths):
    errors, entries = [], []
    if not paths:
        print("usage: python3 -m src.probes.validate probes/*.jsonl")
        return 2
    for path in paths:
        try:
            for n, obj, err in load_lines(path):
                if err:
                    errors.append(f"{path}:{n}: {err}")
                    continue
                problems = check_common(obj, path) + check_scorer(obj)
                errors += [f"{path}:{n}: {p}" for p in problems]
                if isinstance(obj, dict):
                    entries.append((path, n, obj))
        except OSError as e:
            errors.append(f"{path}: cannot read file: {e}")

    set_errors, warnings, summary = check_set(entries)
    errors += set_errors

    for e in errors:
        print(e)
    for w in warnings:
        print(f"warning: {w}")
    print(
        f"{len(entries)} probes checked: {len(errors)} error(s), {len(warnings)} warning(s)"
    )
    for line in summary:
        print(f"  {line}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
