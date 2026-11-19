"""Per provider / model / category pass-rate noise floor.

Same method as docs/noise_findings.md, restricted to one category's probes:
  - burst: shuffle each probe's 20 repeats, 10 vs 10 between two windows (stratified),
    20,000 splits; threshold = 99.8th percentile of |window2 - window1| (plan: 1 in 500);
    std / thresholds rescaled to a 12-repeat window by sqrt(10/12).
  - minimum detectable shift (MDS) = 99.8% threshold + 0.84 x std   (about 80% power)
  - run ratio: sd of the category pass rate over the real scheduled runs divided by the sd
    of same-size burst fake runs (3 repeats per probe). Above ~1.35 = extra run-to-run noise.
All numbers are in percentage points of pass rate.
"""

import json
import math
import random

from src.db.connection import connect

BURST_DB = "/Users/ilhamiulugturkkan/Desktop/model-performance-tracker/mpt_burst.db"
REAL_DB = "/Users/ilhamiulugturkkan/Desktop/model-performance-tracker/mpt.db"
N_SPLITS = 20_000
K = 10  # repeats per probe per window in the burst (20 -> 10 + 10)
PROD_REPEATS = 12  # production window: 4 runs x 3 repeats
SCALE = math.sqrt(K / PROD_REPEATS)
POWER_Z = 0.84
GROUP = 3
N_FAKE_SHUFFLES = 3000
random.seed(0)

RUN_ID, PROBE_ID, PROVIDER, REQUEST, IS_PASSED = 1, 2, 3, 4, -1
QUERY = """
    SELECT calls.*, scores.score = 1.0 AS is_passed
    FROM calls LEFT JOIN scores ON scores.call_id = calls.call_id
"""


def sd(vals):
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def pct(sorted_vals, q):
    return sorted_vals[int(q * (len(sorted_vals) - 1))]


def rate(rows):  # rows: (passed, scored)
    den = sum(r[1] for r in rows)
    return sum(r[0] for r in rows) / den if den else None


def to_row(c):
    p = c[IS_PASSED]
    return (1 if p else 0, 0 if p is None else 1)


burst = connect(BURST_DB).execute(QUERY).fetchall()
real_conn = connect(REAL_DB)
category_of = dict(connect(BURST_DB).execute("SELECT probe_id, category FROM probes"))
assert category_of == dict(
    real_conn.execute("SELECT probe_id, category FROM probes")
), "probe ids differ"
scheduled = {
    r
    for r, cfg in real_conn.execute("SELECT run_id, config FROM runs")
    if json.loads(cfg).get("split") == "scheduled"
}
real = [
    c
    for c in real_conn.execute(QUERY).fetchall()
    if c[RUN_ID] in scheduled and "opus" not in c[REQUEST]
]
runs = sorted(scheduled)

categories = sorted(set(category_of.values())) + ["ALL"]
members = lambda cat: [p for p, c in category_of.items() if cat == "ALL" or c == cat]

print(
    f"{'provider':10} {'model':18} {'category':12} {'probes':>6} {'pass%':>6} {'std':>5} "
    f"{'95%':>5} {'99.8%':>6} {'MDS':>5} {'run ratio':>9}  note"
)
for model in sorted({c[REQUEST] for c in real}):
    provider = next(c[PROVIDER] for c in burst if c[REQUEST] == model)
    by_probe = {}
    for c in burst:
        if c[REQUEST] == model:
            by_probe.setdefault(c[PROBE_ID], []).append(to_row(c))
    cat_probes = {cat: [by_probe[p] for p in members(cat)] for cat in categories}

    # null: one shuffle per split, every category read off the same shuffle
    diffs = {cat: [] for cat in categories}
    for _ in range(N_SPLITS):
        w1, w2 = {}, {}
        for p, rows in by_probe.items():
            random.shuffle(rows)
            cat = category_of[p]
            w1.setdefault(cat, []).extend(rows[:K])
            w2.setdefault(cat, []).extend(rows[K : 2 * K])
        for cat in categories[:-1]:
            diffs[cat].append(rate(w2[cat]) - rate(w1[cat]))
        a = [r for v in w1.values() for r in v]
        b = [r for v in w2.values() for r in v]
        diffs["ALL"].append(rate(b) - rate(a))

    # burst fake runs (3 repeats per probe) for the run-to-run comparison
    n_groups = 20 // GROUP
    fake_var = {cat: 0.0 for cat in categories}
    for _ in range(N_FAKE_SHUFFLES):
        groups = [{} for _ in range(n_groups)]
        for p, rows in by_probe.items():
            random.shuffle(rows)
            for g in range(n_groups):
                groups[g].setdefault(category_of[p], []).extend(
                    rows[g * GROUP : (g + 1) * GROUP]
                )
        for cat in categories:
            vals = [
                rate([r for v in g.values() for r in v] if cat == "ALL" else g[cat])
                for g in groups
            ]
            fake_var[cat] += sd(vals) ** 2

    by_run = {}
    for c in real:
        if c[REQUEST] == model:
            by_run.setdefault(c[RUN_ID], {}).setdefault(
                category_of[c[PROBE_ID]], []
            ).append(to_row(c))

    for cat in categories:
        d = diffs[cat]
        m = sum(d) / len(d)
        std = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1))
        absd = sorted(abs(v) for v in d)
        t95, t998 = pct(absd, 0.95) * SCALE, pct(absd, 0.998) * SCALE
        mds = t998 + POWER_Z * std * SCALE
        p_hat = rate([r for rows in cat_probes[cat] for r in rows])
        burst_sd = math.sqrt(fake_var[cat] / N_FAKE_SHUFFLES)
        run_rates = [
            rate(
                [r for v in by_run[x].values() for r in v]
                if cat == "ALL"
                else by_run[x][cat]
            )
            for x in runs
        ]
        run_sd = sd(run_rates)
        ratio = f"{run_sd / burst_sd:9.2f}" if burst_sd > 0 else f"{'n/a':>9}"
        note = ""
        if std == 0:
            note = "no variance: always the same result, floor undefined"
        elif p_hat > 0.97 or p_hat < 0.03:
            note = "near ceiling/floor: thresholds unreliable"
        print(
            f"{provider:10} {model:18} {cat:12} {len(members(cat)):6d} {100 * p_hat:6.1f} {100 * std * SCALE:5.2f} "
            f"{100 * t95:5.2f} {100 * t998:6.2f} {100 * mds:5.2f} {ratio}  {note}"
        )
    print()
print(
    "std/95%/99.8%/MDS: points of pass rate, 12-repeat window (4 runs). run ratio = real run sd / burst fake-run sd."
)
