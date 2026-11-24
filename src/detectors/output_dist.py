"""Output-text detector: did the shape of the replies change?

Six cheap features are read from every reply (see FEATURES). For each feature the
detector takes the per-probe mean in each window, the per-probe change in that
mean, and the average of those changes over the probes seen in both windows.
That average is the feature's effect, in the feature's own unit (lines, markers,
share, characters, fraction, ratio).

Each feature has its own noise threshold per "model|category" (thresholds.json,
section "text"). The detector alarms when any feature's |effect| exceeds its
threshold, and names the features that moved. A key without thresholds is never
checked ("no_threshold").
"""

import math
import re
import statistics
from functools import lru_cache

from .data import load_window, model_provider
from .pass_rate import (
    load_thresholds,
    restrict_to_common_probes,
    scaled_threshold,
)
from .result import AlertContext, DetectorResult

FEATURES = (
    "lines",  # number of lines in the reply
    "markdown",  # headers + bold spans + bullets (- * or 1. 1)) + code fences, per reply
    "digit_share",  # digits / characters
    "word_len",  # average word length, in characters
    "short_start",  # 1 if the first line is under SHORT_START_CHARS characters, else 0
    "distinct",  # unique words / total words
)
FEATURE_LABELS = {
    "lines": "line count",
    "markdown": "markdown markers",
    "digit_share": "digit share",
    "word_len": "average word length",
    "short_start": "short first line",
    "distinct": "distinct-word ratio",
}
SHORT_START_CHARS = 60
# A ratio this close to 1 counts as AT the limit, not beyond it. Needed because the
# stored thresholds are rounded to 6 significant digits, and a lumpy feature's no-change
# effect often sits exactly on its limit: without this, rounding turns ties into alarms.
TIE_TOLERANCE = 1e-4

_WORD = re.compile(r"\w+(?:['’]\w+)*")
_HEADER = re.compile(r"^ {0,3}#{1,6}[ \t]", re.MULTILINE)
# A bullet is "- " or "* "; a numbered item is up to 3 digits then "." or ")" then a space
# ("3.14" and "2024. " are not).
_BULLET = re.compile(r"^[ \t]*(?:[-*]|\d{1,3}[.)])[ \t]+\S", re.MULTILINE)
_FENCE = re.compile(r"^[ \t]*```", re.MULTILINE)
_BOLD = re.compile(r"\*\*[^*\n]+\*\*")


def extract_features(text):
    """The six features of one reply, as {name: float or None}.

    Cached by text: the calibration re-reads the same replies in every split.

    None means the feature is undefined for this reply (no words, or no first line);
    such replies are left out of that feature's averages. A reply that is empty or
    only whitespace has no first line and no words, but it still has 0 lines,
    0 markers and a 0 digit share.
    """
    return dict(zip(FEATURES, _extract(text)))


@lru_cache(maxsize=50_000)
def _extract(text):
    stripped = text.strip()
    words = _WORD.findall(stripped.lower())
    lines = stripped.splitlines()

    markers = (
        len(_HEADER.findall(stripped))
        + len(_BOLD.findall(stripped))
        + len(_BULLET.findall(stripped))
        + len(_FENCE.findall(stripped))
    )
    # Same order as FEATURES.
    return (
        float(len(lines)),
        float(markers),
        sum(ch.isdigit() for ch in stripped) / len(stripped) if stripped else 0.0,
        sum(map(len, words)) / len(words) if words else None,
        float(len(lines[0].strip()) < SHORT_START_CHARS) if lines else None,
        len(set(words)) / len(words) if words else None,
    )


def window_features(window):
    """{probe_id: {feature: [value per reply]}}; replies with no text, or undefined values, are left out.

    A row with raw_output None (no reply at all) is skipped. An empty string is a
    reply (as in D11) and counts for the features that are defined for it.
    """
    probes = {}
    for row in window:
        text = row.get("raw_output")
        if text is None:
            continue
        per_feature = probes.setdefault(row["probe_id"], {f: [] for f in FEATURES})
        for name, value in extract_features(text).items():
            if value is not None:
                per_feature[name].append(value)
    return probes


def compare_window_features(reference_window, current_window, top_n=3):
    """Compare the reply features of two windows.

    Only probes in both windows count, and for a feature only probes where both
    windows have a defined value for it.

    Returns a dict:
      effects:  {feature: mean over probes of (current mean - reference mean), None if no probe counts}
      n_probes: {feature: how many probes went into the effect}
      top:      {feature: [(probe_id, change)]}, the `top_n` biggest per-probe changes by absolute size
    """
    ref = window_features(reference_window)
    cur = window_features(current_window)
    effects, n_probes, top = {}, {}, {}
    for name in FEATURES:
        changes = {}
        for probe_id in ref.keys() & cur.keys():
            r, c = ref[probe_id][name], cur[probe_id][name]
            if r and c:
                changes[probe_id] = statistics.fmean(c) - statistics.fmean(r)
        effects[name] = statistics.fmean(changes.values()) if changes else None
        n_probes[name] = len(changes)
        top[name] = sorted(changes.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:top_n]
    return {"effects": effects, "n_probes": n_probes, "top": top}


def known_thresholds(thresholds, key):
    """{feature: 12-repeat threshold} for `key`, keeping only usable numbers.

    A feature is usable if its threshold is a positive finite number. Missing
    section, missing key, null (deterministic category) or junk gives {} and the
    detector then does nothing for that key. A feature without a usable threshold
    is never checked.
    """
    entry = thresholds.get("text", {}).get("thresholds", {}).get(key)
    if not isinstance(entry, dict):
        return {}
    usable = {}
    for name in FEATURES:
        value = entry.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(value) and value > 0:
            usable[name] = value
    return usable


def _fmt(name, value):
    # Shares and fractions are shown in percentage points, 2 decimals (a digit-share
    # limit is often a fraction of a point). Others get 3 significant digits, so a
    # limit like 0.0012 is not shown as 0.00.
    if name == "digit_share" or name == "short_start":
        return f"{value * 100:+.2f} points"
    return f"{value:+.3g}"


def is_beyond(ratio):
    """True if |effect| / limit is over 1, ignoring ties at the limit (see TIE_TOLERANCE)."""
    return ratio > 1 + TIE_TOLERANCE


def feature_ratios(effects, known, n_ref_runs, n_cur_runs):
    """{feature: (|effect| / limit, limit)} for the checked features that have an effect.

    `known` is known_thresholds(...) (12-repeat thresholds); each limit is scaled
    to the window sizes. A ratio above 1 is beyond the noise threshold.
    """
    ratios = {}
    for name, threshold12 in known.items():
        if effects[name] is not None:
            limit = scaled_threshold(threshold12, n_ref_runs, n_cur_runs)
            ratios[name] = (abs(effects[name]) / limit, limit)
    return ratios


def detect_windows(
    model,
    category,
    reference_window,
    current_window,
    n_ref_runs,
    n_cur_runs,
    thresholds,
    provider="unknown",
    window_runs=(),
):
    """Output-text detector: rows in, DetectorResult out (no database).

    The statistic is the largest ratio |effect| / threshold over the checked
    features, so >1 means at least one feature moved beyond its noise threshold
    (thresholds scaled to the window sizes). The reason names every feature beyond
    its threshold, or the closest one if none is.
    """
    key = f"{model}|{category}"

    def result(status, reason, statistic=None):
        return DetectorResult(
            detector="text",
            metric_unit="score",
            statistic=statistic,
            reason=reason,
            status=status,
            context=AlertContext(
                provider=provider,
                model=model,
                category=category,
                window_runs=list(window_runs),
                threshold=1.0 if statistic is not None else None,
            ),
        )

    if n_ref_runs == 0 or n_cur_runs == 0:
        return result("no_data", "A window has no runs.")

    reference_window, current_window = restrict_to_common_probes(
        reference_window, current_window
    )
    compared = compare_window_features(reference_window, current_window)
    effects = compared["effects"]
    if all(v is None for v in effects.values()):
        return result(
            "no_data",
            "No probes with reply text appear in both windows.",
        )

    known = known_thresholds(thresholds, key)
    if not known:
        if key in thresholds.get("text", {}).get("thresholds", {}):
            why = (
                "calibrated, but its replies never varied in the burst, so a "
                "noise threshold cannot be determined"
            )
        else:
            why = "never calibrated"
        return result(
            "no_threshold",
            f"No text thresholds for {key} ({why}), so no alarm rule exists "
            "and nothing is checked.",
        )

    ratios = feature_ratios(effects, known, n_ref_runs, n_cur_runs)
    if not ratios:
        return result("no_data", "No checked feature has a value in both windows.")

    beyond = sorted(
        (n for n, (ratio, _) in ratios.items() if is_beyond(ratio)),
        key=lambda n: -ratios[n][0],
    )
    shown = beyond or [max(ratios, key=lambda n: ratios[n][0])]
    parts = "; ".join(
        f"{FEATURE_LABELS[n]} {_fmt(n, effects[n])} (limit {_fmt(n, ratios[n][1])[1:]})"
        for n in shown
    )
    unchecked = [FEATURE_LABELS[n] for n in FEATURES if n not in known]
    note = f" Not checked (no threshold): {', '.join(unchecked)}." if unchecked else ""
    statistic = max(r for r, _ in ratios.values())
    if beyond:
        reason = f"Reply text changed ({n_cur_runs} runs): {parts}.{note}"
    else:
        reason = f"Reply text within noise ({n_cur_runs} runs). Closest: {parts}.{note}"
    return result("alarm" if beyond else "ok", reason, statistic)


def detect(model, category, reference_runs, current_runs, conn=None, thresholds=None):
    """Load both windows from the database and run detect_windows on them."""
    return detect_windows(
        model,
        category,
        load_window(model, category, reference_runs, conn),
        load_window(model, category, current_runs, conn),
        len(reference_runs),
        len(current_runs),
        thresholds if thresholds is not None else load_thresholds(),
        provider=model_provider(model, conn) or "unknown",
        window_runs=current_runs,
    )
