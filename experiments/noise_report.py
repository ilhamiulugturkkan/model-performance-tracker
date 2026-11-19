"""Noise report: python -m experiments.noise_report  ->  docs/noise-report.md + 2 SVG charts.

Noise comes from the burst (68 probes x 20 repeats per model, no change between calls).
Canonical thresholds: empirical 99.8th percentile of random stratified burst splits (pass rate).
A closed form is used only for the predicted sd of one run, where no tail estimate is needed:
    Var(window mean) = S / (P^2 * r),   S = sum over probes of the within-probe variance, P = probes
"""

import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev, variance

from src.db.connection import connect

BURST_DB, REAL_DB, OUT = "mpt_burst.db", "mpt.db", Path("docs")
R = 12  # repeats per probe in a production window (4 runs x 3)
K, N_SPLITS = (
    10,
    20_000,
)  # burst split: 10 repeats per probe in each window, scaled to R by sqrt(K / R)
Z, ZB = 3.0902, 0.8416  # 1-in-500 two-sided false alarm, 80% power
DROPS = (5, 10)  # points of pass rate
QUERY = """SELECT calls.run_id, calls.provider, calls.request_string, calls.probe_id, scores.score = 1.0,
                  calls.latency, calls.output_tokens, calls.raw_output
           FROM calls LEFT JOIN scores ON scores.call_id = calls.call_id"""
METRICS = [("pass rate (pts)", 0, 100), ("latency (s)", 1, 1), ("output tokens", 2, 1)]


def load(path, scheduled_only=False):
    conn = connect(path)
    ok = {
        r
        for r, c in conn.execute("SELECT run_id, config FROM runs")
        if json.loads(c).get("split") == "scheduled"
    }
    rows = [
        (r, m, p, (s, lat, tok, (text or "").strip()), prov)
        for r, prov, m, p, s, lat, tok, text in conn.execute(QUERY)
        if "opus" not in m and (not scheduled_only or r in ok) and lat is not None
    ]
    starts = dict(conn.execute("SELECT run_id, started_at FROM runs"))
    return rows, starts, dict(conn.execute("SELECT probe_id, category FROM probes"))


burst_rows, _, cat_of = load(BURST_DB)
real_rows, starts, _ = load(REAL_DB, scheduled_only=True)
models = sorted({m for _, m, *_ in real_rows})
provider = {m: prov for _, m, _, _, prov in burst_rows}
burst = defaultdict(list)  # (model, probe) -> [(pass, latency, tokens, text)]
for _, m, p, v, _ in burst_rows:
    burst[(m, p)].append(v)
cats = sorted(set(cat_of.values())) + ["all"]
probes_of = {c: [p for p in cat_of if c == "all" or cat_of[p] == c] for c in cats}
runs = sorted({r for r, *_ in real_rows})
slot = {r: "AM" if int(starts[r][11:13]) < 12 else "PM" for r in runs}


def S(model, k, probes):
    """Sum of within-probe variances of metric k (0 pass, 1 latency, 2 tokens)."""
    total = 0.0
    for p in probes:
        v = [t[k] for t in burst[(model, p)] if t[k] is not None]
        total += variance(v) if len(v) > 1 else 0.0
    return total


def run_values(model, k, probes):
    """{run: mean of metric k over that run's scored calls}."""
    acc = defaultdict(list)
    for r, m, p, v, _ in real_rows:
        if m == model and p in probes and v[k] is not None:
            acc[r].append(v[k])
    return {r: mean(x) for r, x in acc.items()}


def perm_null(model):
    """(window 2 - window 1) pass-rate differences per category over random stratified burst splits:
    each probe's repeats are shuffled, K go to window 1 and the rest to window 2 (same probes in both)."""
    real_cats = cats[:-1]
    base = {
        c: [0, 0, 0, 0] for c in real_cats
    }  # sums for probes whose result never varies
    varying = []
    for p in cat_of:
        v = [t[0] for t in burst[(model, p)] if t[0] is not None]
        if len(set(v)) > 1:
            varying.append((cat_of[p], v))
        elif v:
            b = base[cat_of[p]]
            b[0] += K * v[0]
            b[1] += K
            b[2] += (len(v) - K) * v[0]
            b[3] += len(v) - K
    diffs = {c: [] for c in cats}
    for _ in range(N_SPLITS):
        acc = {c: base[c][:] for c in real_cats}
        for c, v in varying:
            random.shuffle(v)
            a = acc[c]
            a[0] += sum(v[:K])
            a[1] += K
            a[2] += sum(v[K:])
            a[3] += len(v) - K
        for c in real_cats:
            diffs[c].append(acc[c][2] / acc[c][3] - acc[c][0] / acc[c][1])
        t = [sum(acc[c][i] for c in real_cats) for i in range(4)]
        diffs["all"].append(t[2] / t[3] - t[0] / t[1])
    sc = math.sqrt(K / R)
    out = {}
    for c, d in diffs.items():
        std, thr = (
            stdev(d) * sc,
            sorted(abs(x) for x in d)[int(0.998 * (len(d) - 1))] * sc,
        )
        out[c] = {"std": std, "thr": thr, "mds": thr + ZB * std}
    return out


def span(path):
    lo, hi = (
        connect(path)
        .execute(
            "SELECT MIN(time_stamp), MAX(time_stamp) FROM calls WHERE request_string NOT LIKE '%opus%'"
        )
        .fetchone()
    )
    lo, hi = datetime.fromisoformat(lo), datetime.fromisoformat(hi)
    mins = int((hi - lo).total_seconds() // 60)
    return f"{lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M} UTC ({mins // 60} h {mins % 60} min)"


def perm_p(vals):
    """Exact permutation p for PM - AM over which runs are PM."""
    pm = [r for r in runs if slot[r] == "PM"]
    diff = lambda grp: (
        mean(vals[r] for r in grp) - mean(vals[r] for r in runs if r not in grp)
    )
    d = diff(pm)
    ps = [abs(diff(c)) >= abs(d) - 1e-12 for c in combinations(runs, len(pm))]
    return d, sum(ps) / len(ps)


def identical(groups):
    same = sum(c * (c - 1) // 2 for g in groups for c in Counter(g).values())
    return same / sum(len(g) * (len(g) - 1) // 2 for g in groups)


def table(head, rows):
    return "\n".join(
        ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def bars_svg(path, title, items, refs=(), legend=()):
    """Horizontal bars. items: (label, value, colour). refs: dashed reference values."""
    W, row, left = 780, 19, 230
    H = 70 + row * len(items)
    mx = max([v for _, v, _ in items] + list(refs)) * 1.08
    x = lambda v: left + (W - left - 90) * v / mx
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
        f'<text x="16" y="22" font-size="15" font-weight="600" fill="#0b0b0b">{title}</text>',
    ]
    lx = 16
    for name, col in legend:
        s.append(
            f'<rect x="{lx}" y="34" width="10" height="10" rx="2" fill="{col}"/><text x="{lx + 15}" y="43" fill="#52514e">{name}</text>'
        )
        lx += 15 + 7 * len(name) + 16
    for r in refs:
        s.append(
            f'<line x1="{x(r):.0f}" x2="{x(r):.0f}" y1="52" y2="{H - 8}" stroke="#52514e" stroke-dasharray="4 3"/>'
            f'<text x="{x(r) + 4:.0f}" y="{H - 12}" fill="#52514e">{r:g}</text>'
        )
    for i, (lab, v, col) in enumerate(items):
        y = 56 + i * row
        s.append(
            f'<text x="{left - 8}" y="{y + 10}" text-anchor="end" fill="#52514e">{lab}</text>'
            f'<rect x="{left}" y="{y}" width="{max(x(v) - left, 1):.0f}" height="13" rx="2" fill="{col}"><title>{lab}: {v:.2f}</title></rect>'
            f'<text x="{x(v) + 5:.0f}" y="{y + 10}" fill="#0b0b0b">{v:.1f}</text>'
        )
    path.write_text("".join(s) + "</svg>", encoding="utf-8")


COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
random.seed(0)
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True
).stdout.strip()
md = [
    "# Noise report (W3.1)\n",
    f"- Generated: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m scripts.noise_report`, repo commit `{commit}`.",
    f"- Burst (`{BURST_DB}`): {span(BURST_DB)}. {len(models)} models x {len(cat_of)} probes x 20 repeats, fired back to back.",
    f"- Scheduled (`{REAL_DB}`): runs {runs[0]}-{runs[-1]}, {span(REAL_DB)}. Scheduled split, no Opus, 3 repeats per probe per run.",
    f"- Window = {R} repeats per probe (4 runs). Pass-rate values are points. Plan target: 1 false alarm in 500 checks.",
    (
        f"- **Canonical method:** the false-alarm threshold is the empirical 99.8th percentile of |window 2 - window 1| over "
        f"{N_SPLITS:,} random stratified burst splits (each probe's repeats shuffled, {K} per window, so both windows hold the same probes), "
        f"scaled to {R} repeats by sqrt({K}/{R}). The normal approximation (3.09 x std) is shown beside it for comparison only; "
        "the two differ most for the 10-probe categories (higher for some, lower for others), where the tails are heavy or lumpy; "
        "where the normal figure is lower, using it would give more than 1 false alarm in 500. "
        "Output-length thresholds in section 4 are still the normal approximation (provisional). Latency is not thresholded (section 4).\n"
    ),
]

# 1. pass-rate floor per provider / model / category
floor = {m: perm_null(m) for m in models}
rows, mds = [], {}
for m in models:
    for c in cats:
        P, s = len(probes_of[c]), S(m, 0, probes_of[c])
        f = floor[m][c]
        p_hat = mean(
            t[0] for p in probes_of[c] for t in burst[(m, p)] if t[0] is not None
        )
        one_run = math.sqrt(s) / (P * math.sqrt(3))
        ratio = (
            f"{stdev(run_values(m, 0, probes_of[c]).values()) / one_run:.2f}"
            if s > 0
            else "n/a"
        )
        mds[(m, c)] = 100 * f["mds"]
        rows.append(
            [
                provider[m],
                m,
                c,
                P,
                f"{100 * p_hat:.1f}",
                f"{100 * f['std']:.2f}",
                f"{100 * f['thr']:.2f}",
                f"{100 * Z * f['std']:.2f}",
                f"{mds[(m, c)]:.2f}" if s > 0 else "no noise",
                ratio,
            ]
        )
md += [
    "## 1. Pass-rate noise floor per provider, model, category\n",
    (
        "`std` = noise of the difference between two windows; `99.8% empirical` = the false-alarm threshold (canonical); "
        "`3.09 x std` = normal approximation, for comparison; `MDS` = minimum detectable shift at 80% power (empirical threshold + 0.84 x std); "
        "`run ratio` = real run-to-run sd / burst-predicted sd (above about 1.35 = runs vary more than the burst predicts). "
        "`no noise` = every probe gives the same result each time.\n"
    ),
    table(
        [
            "provider",
            "model",
            "category",
            "probes",
            "pass %",
            "std",
            "99.8% empirical",
            "3.09 x std",
            "MDS",
            "run ratio",
        ],
        rows,
    ),
    "",
    "![MDS](noise-report-mds.svg)\n",
]
chart_cats = [c for c in cats if all(floor[m][c]["std"] > 0 for m in models)]
bars_svg(
    OUT / "noise-report-mds.svg",
    "Smallest detectable pass-rate drop (points)",
    [
        (f"{c} - {m}", mds[(m, c)], COL[models.index(m)])
        for c in chart_cats
        for m in models
    ],
    refs=DROPS,
    legend=[(m, COL[i]) for i, m in enumerate(models)],
)

# 2. sample sizes
md += [
    "## 2. Runs per window to detect a 5-point and a 10-point drop\n",
    "```text",
    "MDS(r) = MDS12 * sqrt(12 / r)        MDS12 = empirical 99.8% threshold + 0.84 * std for a 12-repeat window (section 1)",
    "r      = 12 * (MDS12 / d)^2          repeats per probe per window needed to detect a drop of d points",
    "runs   = ceil(r / 3)                 3 repeats per probe per run; 2 runs a day; same count for reference and current window",
    "```\n",
]
rows = []
for m in models:
    for c in cats:
        P = len(probes_of[c])
        p_hat = mean(
            t[0] for p in probes_of[c] for t in burst[(m, p)] if t[0] is not None
        )
        row = [m, c, P]
        for d in DROPS:
            if mds[(m, c)] == 0:
                row.append("no noise")
            elif p_hat < d / 100:
                row.append("impossible")
            else:
                n = math.ceil(R * (mds[(m, c)] / d) ** 2 / 3)
                row.append(f"{n} ({n / 2:.1f} d)")
        rows.append(row)
md += [
    table(
        ["model", "category", "probes"] + [f"{d} pt: runs (days)" for d in DROPS], rows
    ),
    "",
]

# 3. identical text
rows = []
for m in models:
    b = [[t[3] for t in burst[(m, p)]] for p in cat_of]
    sched = defaultdict(list)
    for r, mm, p, v, _ in real_rows:
        if mm == m:
            sched[(r, p)].append(v[3])
    rows.append(
        [m, f"{100 * identical(b):.0f}%", f"{100 * identical(sched.values()):.0f}%"]
    )
md += [
    "## 3. Identical-text rate\n",
    "Share of pairs of repeats (same model, same probe) with exactly the same text.\n",
    table(["model", "burst (20 repeats)", "scheduled, within a run (3 repeats)"], rows),
    "",
]

# 4. latency and length against the scheduled runs; 5. time of day
rows4, tests, lat_items, lat_ratio, run_lat = [], [], [], {}, {}
for m in models:
    for name, k, mult in METRICS:
        s = S(m, k, list(cat_of))
        vals = run_values(m, k, set(cat_of))
        std12, one_run = (
            math.sqrt(2 * s) / (len(cat_of) * math.sqrt(R)),
            math.sqrt(s) / (len(cat_of) * math.sqrt(3)),
        )
        lvl_b = mean(t[k] for p in cat_of for t in burst[(m, p)] if t[k] is not None)
        thr = [
            f"{100 * floor[m]['all']['thr']:.3f}",
            "not usable",
            f"{Z * std12:.3f} (provisional)",
        ][k]
        ratio = stdev(vals.values()) / one_run
        rows4.append(
            [
                m,
                name,
                f"{mult * lvl_b:.2f}",
                f"{mult * mean(vals.values()):.2f}",
                thr,
                f"{mult * one_run:.3f}",
                f"{mult * stdev(vals.values()):.3f}",
                f"{ratio:.2f}",
            ]
        )
        d, pv = perm_p(vals)
        am, pm = [mean(vals[r] for r in runs if slot[r] == g) for g in ("AM", "PM")]
        tests.append((m, name, mult * am, mult * pm, mult * d, pv))
        if k == 1:
            lat_items += [(f"{m} AM", am, "#2a78d6"), (f"{m} PM", pm, "#eb6834")]
            lat_ratio[m], run_lat[m] = ratio, vals
n_tests = len(tests)
rows5 = [
    [
        m,
        name,
        f"{am:.3f}",
        f"{pm:.3f}",
        f"{d:+.3f}",
        f"{pv:.3f}",
        f"{min(1.0, pv * n_tests):.3f}",
    ]
    for m, name, am, pm, d, pv in tests
]
flagged = [t for t in tests if t[5] < 0.05 and t[4] > 0]
slow = {
    m: [r for r in runs if run_lat[m][r] > 1.5 * median(run_lat[m].values())]
    for m in models
}
md += [
    "## 4. Latency and output length: burst vs scheduled runs\n",
    (
        "`burst sd` = predicted sd of one run's mean (3 repeats per probe, calls treated as independent); `run sd` = actual sd across the "
        "scheduled runs; `ratio` above about 1.35 = runs vary more than the burst predicts. `threshold` = false-alarm threshold for a "
        "12-repeat window: empirical for pass rate (section 1), normal approximation for output tokens (provisional).\n"
    ),
    (
        f"**Latency cannot be thresholded from the burst.** Real run-to-run latency spread is {min(lat_ratio.values()):.1f}x to "
        f"{max(lat_ratio.values()):.1f}x the burst prediction (largest: {max(lat_ratio, key=lat_ratio.get)}). Provider load drives "
        "day-to-day latency, and the burst's back-to-back calls flatter it. Latency needs its own baseline from the scheduled runs.\n"
    ),
    table(
        [
            "model",
            "metric",
            "burst mean",
            "scheduled mean",
            "threshold",
            "burst sd",
            "run sd",
            "ratio",
        ],
        rows4,
    ),
    "",
    "## 5. Time of day\n",
    (
        f"AM = runs started before 12:00 UTC ({', '.join(str(r) for r in runs if slot[r] == 'AM')}); PM = 15:00 UTC runs "
        f"({', '.join(str(r) for r in runs if slot[r] == 'PM')}). `p` = exact permutation test over which runs are PM; "
        f"`p x {n_tests}` = p multiplied by the number of tests in this table (capped at 1).\n"
    ),
    table(
        ["model", "metric", "AM mean", "PM mean", "PM - AM", "p", f"p x {n_tests}"],
        rows5,
    ),
    "",
    "**Suggestive, not confirmed.** "
    + (
        "PM is higher with raw p < 0.05 for: "
        + "; ".join(f"{m} {name} (p = {pv:.3f})" for m, name, _, _, _, pv in flagged)
        + f". With {n_tests} tests the smallest adjusted p is {min(min(1.0, t[5] * n_tests) for t in tests):.2f}. "
        f"Read it as a possible PM slowdown for {', '.join(sorted({t[0] for t in flagged}))}; it needs more runs."
        if flagged
        else f"No test has raw p < 0.05 with PM higher ({n_tests} tests)."
    ),
    "",
    "Mean latency of each run (s), to see which runs drive the slot differences:\n",
    table(
        ["run", "slot"] + models,
        [[r, slot[r]] + [f"{run_lat[m][r]:.2f}" for m in models] for r in runs],
    ),
    "",
    "Runs slower than 1.5x the model's median run: "
    + "; ".join(f"{m}: {', '.join(map(str, slow[m])) or 'none'}" for m in models)
    + ".\n",
    "Without those runs, PM - AM mean latency: "
    + "; ".join(
        f"{m} {mean(run_lat[m][r] for r in runs if slot[r] == 'PM' and r not in slow[m]) - mean(run_lat[m][r] for r in runs if slot[r] == 'AM' and r not in slow[m]):+.2f} s"
        for m in models
        if slow[m]
    )
    + ".\n",
    "![latency by time of day](noise-report-latency.svg)\n",
]
bars_svg(
    OUT / "noise-report-latency.svg",
    "Mean latency (s) by time of day",
    lat_items,
    legend=[("AM runs", "#2a78d6"), ("PM runs (15:00 UTC)", "#eb6834")],
)

# Notes
fails = (
    connect(BURST_DB)
    .execute("""SELECT c.request_string, SUM(s.details LIKE '%not a valid numeric%'), COUNT(*)
    FROM calls c JOIN scores s ON s.call_id = c.call_id JOIN probes p ON p.probe_id = c.probe_id
    WHERE p.category = 'math' AND s.score = 0 AND c.request_string NOT LIKE '%opus%' GROUP BY 1 ORDER BY 1""")
    .fetchall()
)
math_pass = {
    m: 100
    * mean(t[0] for p in probes_of["math"] for t in burst[(m, p)] if t[0] is not None)
    for m in models
}
md += [
    "## Notes (read before using the numbers)\n",
    "1. **Latency is not thresholded from the burst** (section 4). It needs its own scheduled-data baseline.",
    (
        f"2. **Math pass rates mostly measure answer format, not maths.** The prompts say to reply with only the number. The scorer "
        "(`numeric_match`) requires the whole reply to parse as one number, so a reply that shows working and ends with the answer "
        f"scores 0 even if the final value is correct. Sonnet's math pass rate is {math_pass['claude-sonnet-5-5']:.1f}%; in "
        "`docs/noise_findings.md` all 5 samples checked had a correct final value. Share of each model's failed math calls whose reply was "
        "not a valid number: "
        + "; ".join(f"{m} {100 * n / t:.1f}% of {t}" for m, n, t in fails)
        + ". Sonnet's overall pass rate, "
        "its extra run-to-run variation and its possible trend should be read as format-compliance behaviour."
    ),
    (
        "3. **Scoring is strict.** The scorers are not loosened to forgive format, so a model that drifts away from the requested "
        "format shows up as a pass-rate change (which the monitor should see) and format compliance is part of every pass rate."
    ),
    "4. **The burst is one window** (see its span in the header), so its noise is a lower bound on real noise.",
    (
        "5. **Pinning:** Claude model IDs are documented as pinned; OpenAI lists no dated snapshots for these models, so their "
        "pinning is unknown. The burst assumes no model change inside its window."
    ),
    (
        "6. **Categories with no noise** (coding, structured, whenever the table says `no noise`) have no usable threshold, and the "
        "permutation method gives none. They need a different alarm rule (for example alarm on any new failure). No such rule exists yet."
    ),
    (
        "7. **Monte Carlo error:** an empirical 99.8% threshold rests on about 40 of the 20,000 splits lying beyond it, so the "
        "10-probe categories carry an error of a couple of points (haiku instructions came out at 13.9 in an earlier run and "
        f"{100 * floor['claude-haiku-5-5']['instructions']['thr']:.1f} here). Report them as approximate."
    ),
    (
        f"8. **Many tests:** {n_tests} time-of-day tests, 5 models x 5 categories of floors and several ratios. A 1-in-500 threshold "
        "applies per check, not overall. The many-tests correction for the detectors is still open in the plan."
    ),
    "",
]

(OUT / "noise-report.md").write_text("\n".join(md), encoding="utf-8")
print(f"wrote {OUT / 'noise-report.md'}", file=sys.stderr)
