"""How much noisier is real latency than the burst says? (the "inflation factor")

The burst threshold comes from windows taken back to back, a calm moment. Real windows are
hours apart, and provider load changes. This module measures the extra spread on the real runs:

  effects:  the reference window (runs 4-7) against each later 4-run window (starting at runs
            8, 9, ..., 18), through the same compare_window_latencies the detector uses.
  spread:   root-mean-square of those effects (about 0, since with no change they should be 0).
  factor:   spread / burst no-change sd at the same window size (12 repeats).

The factor is applied only when it is clearly above 1: above sqrt(chi2_{0.95, n} / n) for n
effects (about 1.35 for 11), which is what a pure-noise factor stays under 95% of the time.
Caveat: the windows overlap and share one reference window, so the effects are correlated and
the cutoff is more lenient than it looks. Every key is marked approximate for that reason.

Diagnostic, lag 1 against lag 2: effects of 4-run windows 4 runs apart (lag 1) against 8 runs
apart (lag 2). With no drift both spreads match; a lag-2 spread clearly above lag 1 means the
latency drifts over time, and the factor then measures drift, not just noise.
"""

import math
import statistics

from src.db.connection import connect
from src.detectors.data import load_window
from src.detectors.output_latency import NIGHT_END_HOUR, SLOTS, compare_window_latencies

REFERENCE_RUNS = [4, 5, 6, 7]
LATER_STARTS = range(8, 19)  # windows 8-11, 9-12, ..., 18-21
LAST_RUN = 21
WIDTH = 4
# Runs to leave out per model, from that run on. Empty: the plan was to cut Haiku at its
# reasoning start, but reasoning ramps up on 5 probes over runs 5-13 and does not move latency;
# see the notes in the report. Kept as a parameter so the cut can be tried.
EXCLUDE_FROM = {}
# Known real drift inside the estimation runs. The estimate keeps it (the assumption is that nothing
# changed, so drift counts as noise), but the factor is recorded with this note.
DRIFT_NOTES = {
    "claude-haiku-5-5": (
        "gradual speed-up of about 1.3% per run over runs 3-21 (a straight line fits better than a "
        "single step), so there is no clean cut-off; the factor includes it"
    ),
}


def _lower_gamma_regularized(a, x):
    """P(a, x), the regularized lower incomplete gamma function (series, then continued fraction)."""
    if x <= 0:
        return 0.0
    if x < a + 1:
        term = total = 1 / a
        n = a
        while abs(term) > abs(total) * 1e-15:
            n += 1
            term *= x / n
            total += term
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1 - a
    c, d = 1e300, 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_quantile(p, df):
    """The p-quantile of a chi-square distribution with df degrees of freedom (bisection)."""
    lo, hi = 0.0, max(10.0, 10.0 * df)
    for _ in range(200):
        mid = (lo + hi) / 2
        if _lower_gamma_regularized(df / 2, mid / 2) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cutoff(n_effects):
    """Factor above which the spread is clearly more than noise: sqrt(chi2_{0.95, n} / n)."""
    return math.sqrt(chi2_quantile(0.95, n_effects) / n_effects)


def _rms(values):
    return math.sqrt(statistics.fmean(v * v for v in values))


def _effect(model, category, ref_runs, cur_runs, conn):
    return compare_window_latencies(
        load_window(model, category, ref_runs, conn),
        load_window(model, category, cur_runs, conn),
    )["effect"]


def real_effects(model, category, conn=None):
    """Effects (%) of the reference window against each later 4-run window, as {start_run: effect}."""
    conn = conn or connect()
    last = EXCLUDE_FROM.get(model, LAST_RUN + 1) - 1
    out = {}
    for start in LATER_STARTS:
        runs = list(range(start, start + WIDTH))
        if runs[-1] > last:
            continue
        effect = _effect(model, category, REFERENCE_RUNS, runs, conn)
        if effect is not None:
            out[start] = effect
    return out


def lag_effects(model, category, lag_runs, conn=None):
    """Effects of 4-run windows `lag_runs` apart (every start that fits), as a list."""
    conn = conn or connect()
    last = EXCLUDE_FROM.get(model, LAST_RUN + 1) - 1
    out = []
    for start in range(1, last - lag_runs - WIDTH + 2):
        a = list(range(start, start + WIDTH))
        b = list(range(start + lag_runs, start + lag_runs + WIDTH))
        if b[-1] > last:
            continue
        effect = _effect(model, category, a, b, conn)
        if effect is not None:
            out.append(effect)
    return out


def estimate_factor(model, category, burst_sd, conn=None):
    """Everything about one key's inflation: spread, factor, cutoff, applied factor, lag diagnostic."""
    conn = conn or connect()
    effects = list(real_effects(model, category, conn).values())
    if not effects or not burst_sd:
        return None
    spread = _rms(effects)
    estimated = spread / burst_sd
    limit = cutoff(len(effects))
    lag1 = lag_effects(model, category, WIDTH, conn)
    lag2 = lag_effects(model, category, 2 * WIDTH, conn)
    return {
        "n_windows": len(effects),
        "mean_effect": statistics.fmean(effects),
        "spread": spread,
        "burst_sd": burst_sd,
        "estimated": estimated,
        "cutoff": limit,
        "applied": estimated if estimated > limit else 1.0,
        "lag1_spread": _rms(lag1) if lag1 else None,
        "lag2_spread": _rms(lag2) if lag2 else None,
        "approximate": True,  # overlapping windows, one shared reference window
        **({"note": DRIFT_NOTES[model]} if model in DRIFT_NOTES else {}),
    }


def estimate_factors(burst):
    """{key: estimate_factor(...)} for every key in burst_calibration(...) that has a threshold."""
    conn = connect()
    out = {}
    for key, entry in burst.items():
        if entry["threshold"] is None:
            continue
        model, category = key.split("|")
        est = estimate_factor(model, category, entry["null_sd"], conn)
        if est is not None:
            out[key] = est
    return out


# ---- day and night separately ----
#
# Same idea per slot, but a same-slot 4-run window spans 8 runs and only about 9 runs per slot
# exist, which leaves 2 later windows: too few to measure a spread. So the slot estimate uses
# 2-run same-slot windows (6 repeats), 6 later windows, and compares with the burst noise at that
# size (the 12-repeat burst sd times sqrt(12 / 6)). Applying that factor to the 12-repeat limit
# assumes the factor does not change with window size, which is only roughly true (latency
# drifts), so every slot key is approximate.

SLOT_WINDOW = 2  # runs per window within one slot
SLOT_FIRST_RUN = {
    "night": 4,
    "day": 5,
}  # regular schedule; the reference pair is runs (4, 6) / (5, 7)
REPEATS_PER_RUN = 3


def slot_runs(conn=None):
    """{"night": [run ids], "day": [run ids]} from each run's UTC start hour, regular schedule only."""
    conn = conn or connect()
    out = {slot: [] for slot in SLOTS}
    for run_id, started_at in conn.execute(
        "SELECT run_id, started_at FROM runs ORDER BY run_id"
    ):
        slot = "night" if int(started_at[11:13]) < NIGHT_END_HOUR else "day"
        if run_id >= SLOT_FIRST_RUN[slot] and run_id <= LAST_RUN:
            out[slot].append(run_id)
    return out


def slot_real_effects(model, category, slot, conn=None):
    """Effects (%) of the slot's first 2-run window against each later, non-overlapping-start 2-run window."""
    conn = conn or connect()
    runs = slot_runs(conn)[slot]
    ref = runs[:SLOT_WINDOW]
    out = []
    for i in range(SLOT_WINDOW, len(runs) - SLOT_WINDOW + 1):
        effect = _effect(model, category, ref, runs[i : i + SLOT_WINDOW], conn)
        if effect is not None:
            out.append(effect)
    return out


def slot_lag_effects(model, category, slot, lag, conn):
    """Effects of 2-run same-slot windows `lag` slot-runs apart (every start that fits)."""
    runs = slot_runs(conn)[slot]
    out = []
    for i in range(len(runs) - lag - SLOT_WINDOW + 1):
        effect = _effect(
            model,
            category,
            runs[i : i + SLOT_WINDOW],
            runs[i + lag : i + lag + SLOT_WINDOW],
            conn,
        )
        if effect is not None:
            out.append(effect)
    return out


def estimate_slot_factor(model, category, slot, burst_sd12, conn=None):
    """Like estimate_factor, for one slot, at 2-run (6 repeat) windows."""
    conn = conn or connect()
    effects = slot_real_effects(model, category, slot, conn)
    if not effects or not burst_sd12:
        return None
    burst_sd = burst_sd12 * math.sqrt(12 / (REPEATS_PER_RUN * SLOT_WINDOW))
    spread = _rms(effects)
    estimated = spread / burst_sd
    limit = cutoff(len(effects))
    lag1 = slot_lag_effects(model, category, slot, SLOT_WINDOW, conn)
    lag2 = slot_lag_effects(model, category, slot, 2 * SLOT_WINDOW, conn)
    return {
        "n_windows": len(effects),
        "mean_effect": statistics.fmean(effects),
        "spread": spread,
        "burst_sd": burst_sd,
        "estimated": estimated,
        "cutoff": limit,
        "applied": estimated if estimated > limit else 1.0,
        "lag1_spread": _rms(lag1) if lag1 else None,
        "lag2_spread": _rms(lag2) if lag2 else None,
        "approximate": True,
        **({"note": DRIFT_NOTES[model]} if model in DRIFT_NOTES else {}),
    }


def estimate_slot_factors(burst):
    """{"model|category|slot": estimate_slot_factor(...)} for every key with a burst threshold."""
    conn = connect()
    out = {}
    for key, entry in burst.items():
        if entry["threshold"] is None:
            continue
        model, category = key.split("|")
        for slot in SLOTS:
            est = estimate_slot_factor(model, category, slot, entry["null_sd"], conn)
            if est is not None:
                out[f"{key}|{slot}"] = est
    return out
