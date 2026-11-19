"""Pass-rate spread: burst "fake runs" vs real scheduled runs, per model.

Burst: each probe's 20 repeats are shuffled and cut into groups of 3. Group g (taking the
g-th 3 repeats of every probe) is a fake run = 68 probes x 3 repeats, same size as a real
run. sd of the suite pass rate across the fake runs, averaged over many shuffles.
Real: sd of the suite pass rate across the scheduled runs (no Opus).
"""

import json
import math
import random

from src.db.connection import connect

BURST_DB = "/Users/ilhamiulugturkkan/Desktop/model-performance-tracker/mpt_burst.db"
REAL_DB = "/Users/ilhamiulugturkkan/Desktop/model-performance-tracker/mpt.db"
GROUP = 3  # repeats per probe per fake run (= one real run)
N_SHUFFLES = 5000
random.seed(0)

RUN_ID, PROBE_ID, REQUEST, IS_PASSED = 1, 2, 4, -1
QUERY = """
    SELECT calls.*, scores.score = 1.0 AS is_passed
    FROM calls LEFT JOIN scores ON scores.call_id = calls.call_id
"""


def sd(vals):
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def pass_rate(rows):  # rows: (passed, scored); unscored (errored) calls are skipped
    return sum(r[0] for r in rows) / sum(r[1] for r in rows)


def to_row(c):
    p = c[IS_PASSED]
    return (1 if p else 0, 0 if p is None else 1)


burst = connect(BURST_DB).execute(QUERY).fetchall()
real_conn = connect(REAL_DB)
scheduled = {
    run_id
    for run_id, config in real_conn.execute("SELECT run_id, config FROM runs")
    if json.loads(config).get("split") == "scheduled"
}
real = [
    c
    for c in real_conn.execute(QUERY).fetchall()
    if c[RUN_ID] in scheduled and "opus" not in c[REQUEST]
]
runs_all = sorted(scheduled)
runs_late = [r for r in runs_all if r >= 3]
print(
    f"scheduled runs: {runs_all}  (no Opus); burst fake runs per shuffle: {20 // GROUP}"
)

print(
    f"\n{'model':18} {'burst sd':>9} {'real sd':>9} {'ratio':>6}   {'real sd':>9} {'ratio':>6}"
)
print(f"{'':18} {'(fake)':>9} {'runs 1-11':>9} {'':>6}   {'runs 3-11':>9}")
for model in sorted({c[REQUEST] for c in real}):
    # burst fake runs
    by_probe = {}
    for c in burst:
        if c[REQUEST] == model:
            by_probe.setdefault(c[PROBE_ID], []).append(to_row(c))
    probes = list(by_probe.values())
    n_groups = min(len(r) for r in probes) // GROUP
    var_sum = 0.0
    for _ in range(N_SHUFFLES):
        fake = [[] for _ in range(n_groups)]
        for rows in probes:
            random.shuffle(rows)
            for g in range(n_groups):
                fake[g].extend(rows[g * GROUP : (g + 1) * GROUP])
        var_sum += sd([pass_rate(f) for f in fake]) ** 2
    burst_sd = math.sqrt(var_sum / N_SHUFFLES)  # average the variances, then sqrt

    # real runs
    by_run = {}
    for c in real:
        if c[REQUEST] == model:
            by_run.setdefault(c[RUN_ID], []).append(to_row(c))
    sd_all = sd([pass_rate(by_run[r]) for r in runs_all])
    sd_late = sd([pass_rate(by_run[r]) for r in runs_late])
    print(
        f"{model:18} {burst_sd:9.4f} {sd_all:9.4f} {sd_all / burst_sd:6.2f}   "
        f"{sd_late:9.4f} {sd_late / burst_sd:6.2f}"
    )
print(
    "\nratio = real sd / burst sd. ~1.0 = runs vary no more than same-size burst groups;"
)
print(
    "above ~1.35 (11 runs) / ~1.39 (9 runs) = more run-to-run spread than burst noise explains (5%, normal)."
)
