# Reference solutions used to verify the coding probes. Keep for your own checks.

# --- count_divisible_above ---
def count_divisible_above(nums, k, m):
    return sum(1 for n in nums if n % k == 0 and n > m)


# --- total_covered_length ---
def total_covered_length(intervals):
    total = 0
    cur_start = cur_end = None
    for s, e in sorted(intervals):
        if cur_end is None or s > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


# --- second_largest_distinct ---
def second_largest_distinct(nums):
    vals = sorted(set(nums))
    return vals[-2] if len(vals) >= 2 else None


# --- run_length_decode ---
def run_length_decode(s):
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        i += 1
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        out.append(ch * (int(s[i:j]) if j > i else 1))
        i = j
    return "".join(out)


# --- is_balanced ---
def is_balanced(s):
    pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
    stack = []
    for ch in s:
        if ch in "([{<":
            stack.append(ch)
        elif ch in pairs:  # noqa: SIM102
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


# --- rotate_clockwise ---
def rotate_clockwise(matrix):
    return [list(row) for row in zip(*matrix[::-1])]


# --- business_days_between ---
import datetime


def business_days_between(d1, d2):
    a = datetime.date.fromisoformat(d1)
    b = datetime.date.fromisoformat(d2)
    n, cur = 0, a
    while cur < b:
        if cur.weekday() < 5:
            n += 1
        cur += datetime.timedelta(days=1)
    return n


# --- average_of_positives ---
def average_of_positives(nums):
    pos = [n for n in nums if n > 0]
    return sum(pos) / len(pos) if pos else 0.0


# --- last_k ---
def last_k(items, k):
    return items[max(len(items) - k, 0) :] if k > 0 else []


# --- parse_duration ---
import re


def parse_duration(s):
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s)
    h, mi, se = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + se


# --- longest_run ---
def longest_run(s):
    best = cur = 0
    prev = None
    for ch in s:
        cur = cur + 1 if ch == prev else 1
        prev = ch
        best = max(best, cur)
    return best


# --- top_k_words ---
import re  # noqa: F811
from collections import Counter


def top_k_words(text, k):
    words = re.findall(r"[a-z]+", text.lower())
    ranked = sorted(Counter(words).items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:k]]
