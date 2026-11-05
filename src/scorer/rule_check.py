import re

from src.scorer.common import no_reply

KNOWN = {
    "sentences",
    "words",
    "unique_words",
    "letters_only",
    "each_word_contains",
    "each_word_starts_with",
    "min_word_length",
    "forbidden_letters",
    "forbidden_words",
    "required_words",
    "words_from_list",
    "uppercase_words",
    "list_items",
    "digit_count",
    "starts_with",
    "ends_with",
}

WORD = re.compile(r"[A-Za-z']+")


def words_of(text):
    return WORD.findall(text.lower())


def rule_check(probe, reply):
    rules = probe.get("rules")
    # A typo'd rule must never be silently ignored.
    unknown = set(rules) - KNOWN
    if unknown:
        raise ValueError(f"unknown rule key(s): {sorted(unknown)}")
    if reply is None:
        return no_reply()
    all_passed = True
    all_details = {}
    if rules.get("sentences"):
        passed, details = check_sentences(rules, reply)
        all_passed &= passed
        all_details["sentences"] = details
    if rules.get("words"):
        passed, details = check_words(rules, reply)
        all_passed &= passed
        all_details["words"] = details
    if rules.get("unique_words"):
        passed, details = check_unique_words(reply)
        all_passed &= passed
        all_details["unique_words"] = details
    if rules.get("letters_only"):
        passed, details = check_letters_only(reply)
        all_passed &= passed
        all_details["letters_only"] = details
    if rules.get("each_word_contains"):
        passed, details = check_each_word_contains(rules, reply)
        all_passed &= passed
        all_details["each_word_contains"] = details
    if rules.get("each_word_starts_with"):
        passed, details = check_each_word_starts_with(rules, reply)
        all_passed &= passed
        all_details["each_word_starts_with"] = details
    if rules.get("min_word_length"):
        passed, details = check_min_word_length(rules, reply)
        all_passed &= passed
        all_details["min_word_length"] = details
    if rules.get("forbidden_letters"):
        passed, details = check_forbidden_letters(rules, reply)
        all_passed &= passed
        all_details["forbidden_letters"] = details
    if rules.get("forbidden_words"):
        passed, details = check_forbidden_words(rules, reply)
        all_passed &= passed
        all_details["forbidden_words"] = details
    if rules.get("required_words"):
        passed, details = check_required_words(rules, reply)
        all_passed &= passed
        all_details["required_words"] = details
    if rules.get("words_from_list"):
        passed, details = check_words_from_list(rules, reply)
        all_passed &= passed
        all_details["words_from_list"] = details
    if rules.get("uppercase_words"):
        passed, details = check_uppercase_words(rules, reply)
        all_passed &= passed
        all_details["uppercase_words"] = details
    if rules.get("list_items"):
        passed, details = check_list_items(rules, reply)
        all_passed &= passed
        all_details["list_items"] = details
    if rules.get("digit_count") is not None:
        passed, details = check_digit_count(rules, reply)
        all_passed &= passed
        all_details["digit_count"] = details
    if rules.get("starts_with"):
        passed, details = check_starts_with(rules, reply)
        all_passed &= passed
        all_details["starts_with"] = details
    if rules.get("ends_with"):
        passed, details = check_ends_with(rules, reply)
        all_passed &= passed
        all_details["ends_with"] = details

    return all_passed, all_details


# Return (passed, details)
def check_sentences(rules, reply):
    reply = reply.strip()
    sent_count = rules.get("sentences")

    # Check the sentence count in the reply against the rule. Sentences can end with ., ?, !, etc.
    reply_sentences = re.split(r"(?<=[.?!])\s+", reply)
    if len(reply_sentences) != sent_count:
        return (
            False,
            f"Reply contains {len(reply_sentences)} sentences, but the rule requires {sent_count} sentences.",
        )
    return True, "Sentence count matches the rule."


def check_words(rules, reply):
    reply = reply.strip()
    word_count = rules.get("words")

    # Check the word count in the reply against the rule
    reply_words = words_of(reply)
    if len(reply_words) != word_count:
        return (
            False,
            f"Reply contains {len(reply_words)} words, but the rule requires {word_count} words.",
        )
    return True, "Word count matches the rule."


def check_unique_words(reply):
    # Check if each word is unique or not. Compare after lowercasing each word
    reply_words = [word.lower() for word in words_of(reply)]
    if len(set(reply_words)) != len(reply_words):
        return (
            False,
            "Reply contains duplicate words, but the rule requires all words to be unique.",
        )
    return True, "All words are unique."


def check_letters_only(reply):
    if not re.fullmatch(r"[A-Za-z\s]+", reply):
        return (
            False,
            "Reply contains non-letter characters, but the rule requires only letters.",
        )
    return True, "Reply contains only letters."


def check_each_word_contains(rules, reply):
    required_letter = rules.get("each_word_contains").lower()
    reply_words = [word.lower() for word in words_of(reply)]
    for word in reply_words:
        if required_letter not in word:
            return (
                False,
                f"Word '{word}' does not contain the required letter '{required_letter}'.",
            )
    return True, "All words contain the required letter."


def check_each_word_starts_with(rules, reply):
    # Lower or uppercase the reply and required letter to ensure case-insensitive matching
    required_letter = rules.get("each_word_starts_with").lower()
    reply_words = [word.lower() for word in words_of(reply)]
    for word in reply_words:
        if not word.startswith(required_letter):
            return (
                False,
                f"Word '{word}' does not start with the required letter '{required_letter}'.",
            )
    return True, "All words start with the required letter."


def check_min_word_length(rules, reply):
    min_length = rules.get("min_word_length")
    reply_words = words_of(reply)
    for word in reply_words:
        if len(word) < min_length:
            return (
                False,
                f"Word '{word}' is shorter than the required minimum length of {min_length}.",
            )
    return True, "All words meet the minimum length requirement."


def check_forbidden_letters(rules, reply):
    forbidden_letters = rules.get("forbidden_letters")
    for letter in forbidden_letters:
        if letter.lower() in reply.lower():
            return False, f"Reply contains the forbidden letter '{letter}'."
    return True, "Reply does not contain any forbidden letters."


def check_forbidden_words(rules, reply):
    # compare lowercased words to ensure case-insensitive matching re.findall(r"[A-Za-z']+", reply.lower())
    forbidden_words = rules.get("forbidden_words")
    reply_words = re.findall(r"[A-Za-z']+", reply.lower())
    for word in reply_words:
        if word.lower() in [fw.lower() for fw in forbidden_words]:
            return False, f"Reply contains the forbidden word '{word}'."
    return True, "Reply does not contain any forbidden words."


def check_required_words(rules, reply):
    required_words = rules.get("required_words")
    reply_words = [word.lower() for word in words_of(reply)]
    for word in required_words:
        if word.lower() not in reply_words:
            return False, f"Reply does not contain the required word '{word}'."
    return True, "Reply contains all required words."


def check_words_from_list(rules, reply):
    # Every word in the reply must come from the list (the reply may use a subset).
    allowed = {w.lower() for w in rules.get("words_from_list")}
    for word in words_of(reply):
        if word not in allowed:
            return False, f"Word '{word}' is not in the allowed list."
    return True, "All reply words are in the allowed list."


def check_uppercase_words(rules, reply):
    # Each listed word must appear at least once, and every occurrence must be all caps.
    for target in rules.get("uppercase_words"):
        found = re.findall(rf"\b{re.escape(target)}\b", reply, flags=re.IGNORECASE)
        if not found:
            return False, f"Reply does not contain the word '{target.upper()}'."
        for occurrence in found:
            if occurrence != target.upper():
                return (
                    False,
                    f"Word '{occurrence}' must be written as '{target.upper()}'.",
                )
    return True, "All required words appear in uppercase."


def check_list_items(rules, reply):
    expected = rules.get("list_items")
    items = [item.strip() for item in reply.split(",") if item.strip()]
    if len(items) != expected:
        return (
            False,
            f"Reply contains {len(items)} list items, but the rule requires {expected}.",
        )
    return True, "List item count matches the rule."


def check_digit_count(rules, reply):
    digit_count = rules.get("digit_count")
    reply_digits = [c for c in reply if c.isdigit()]
    if len(reply_digits) != digit_count:
        return (
            False,
            f"Reply does not contain the required number of digits ({digit_count}).",
        )
    return True, "Reply contains the required number of digits."


def check_starts_with(rules, reply):
    # Compare whole words, case-insensitively (design choice), so "Weirdly" != "Weird".
    wanted = words_of(rules.get("starts_with"))
    got = words_of(reply)[: len(wanted)]
    if got != wanted:
        return (
            False,
            f"Reply does not start with the required text '{rules.get('starts_with')}'.",
        )
    return True, "Reply starts with the required text."


def check_ends_with(rules, reply):
    # Compare whole words, case-insensitively (design choice), so "untomorrow" != "tomorrow".
    wanted = words_of(rules.get("ends_with"))
    got = words_of(reply)[-len(wanted) :]
    if got != wanted:
        return (
            False,
            f"Reply does not end with the required text '{rules.get('ends_with')}'.",
        )
    return True, "Reply ends with the required text."
