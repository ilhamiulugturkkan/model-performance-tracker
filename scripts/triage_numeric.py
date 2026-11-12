"""For every FAILED numeric probe call, say WHY it failed. Reads saved replies only: free, no API calls.

    python3 scripts/triage_numeric.py mpt_test.db 1        # database, run id

wrong format : the right answer is the LAST number in the reply, but the reply is not a bare number
buried       : the right answer appears in the reply, but not as its last number (ambiguous)
wrong answer : the right answer does not appear at all
"""

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text):
    text = text.replace("\u2212", "-")  # the unicode minus sign
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,000 -> 1000
    return [float(n) for n in NUMBER.findall(text)]


def classify(text, expected, tolerance):
    found = numbers_in(text)
    close = [abs(n - expected) <= tolerance + 1e-9 for n in found]
    if found and close[-1]:
        return "wrong format"
    if any(close):
        return "buried"
    return "wrong answer"


def main(db, run_id):
    conn = sqlite3.connect(db)
    rows = conn.execute(
        """
        SELECT p.slug, c.request_string, c.raw_output, p.body
        FROM calls c
        JOIN probes p ON p.probe_id = c.probe_id
        JOIN scores s ON s.call_id = c.call_id
        WHERE c.run_id = ? AND s.score = 0 AND c.raw_output IS NOT NULL
        """,
        (run_id,),
    ).fetchall()
    table, totals = defaultdict(Counter), defaultdict(Counter)
    for slug, model, text, body in rows:
        probe = json.loads(body)
        if probe.get("scorer") != "numeric_match":
            continue
        kind = classify(text, probe["expected"], probe["tolerance"])
        table[(slug, model)][kind] += 1
        totals[model][kind] += 1

    print(
        f"{'probe':10} {'model':18} {'wrong format':>12} {'buried':>7} {'wrong answer':>13}"
    )
    for (slug, model), counts in sorted(table.items()):
        print(
            f"{slug:10} {model:18} {counts['wrong format']:>12} {counts['buried']:>7} {counts['wrong answer']:>13}"
        )
    print("\nfailed numeric calls, by model:")
    for model, counts in sorted(totals.items()):
        total = sum(counts.values())
        print(
            f"  {model}: {total} failed = {counts['wrong format']} wrong format, {counts['buried']} buried, {counts['wrong answer']} wrong answer"
        )


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "mpt_test.db",
        int(sys.argv[2]) if len(sys.argv) > 2 else 1,
    )
