# Decisions log

Format: **Decision** (date, status). Why. What would change it.
Newest at the bottom. Never edit an old entry silently; add a new entry that supersedes it.

---

### D1. Project and scope (2026-10-28, active)
Build a developer-facing tool and public dashboard that detects silent LLM API changes, validated against changes I cause myself, plus a small PyTorch model that identifies which LLM wrote an output. Free and public only; no payments, paid features or business use.
**Would change it:** a hard blocker in provider terms, or the 10-week timeline slipping by more than 2 days at the W4 go/no-go.

### D2. Repo name and license (2026-10-28, active)
Repo `model-performance-tracker`, MIT license. Repo name avoids provider names.

### D3. Providers (2026-10-28, active)
- **Monitored for real drift:** Claude and GPT through their official APIs.
- **Validation bench:** self-hosted Gemma 4 (Apache 2.0). Fixed weights give a no-change control; known changes (switch between sizes, temperature, system prompt) give ground truth.
- **Excluded:** xAI Grok (enterprise terms bar benchmarking and training on outputs), Meta Muse Spark 1.1 (terms unreadable/unclear, US-only preview), hosted open-weight APIs (host terms add a layer).
**Why:** only these have terms I could read and that allow the project. See `compliance.md`.
**Would change it:** terms change (re-check before launch).

### D4. Claude outputs are not used to train anything (2026-10-28, active, may be revised by Nov 30)
Anthropic's Usage Policy (effective Nov 12, 2026) requires prior authorization to use outputs to train an AI model. Two replies from Anthropic's AI support assistant said a discriminative classifier fits the permitted uses but that it cannot give formal authorization. Claude stays monitoring-only (statistical detectors) unless a human at Anthropic confirms in writing before W5.1 (Nov 30).
**Would change it:** written confirmation from a person at Anthropic (conditions listed in `compliance.md`).

### D5. No public classifier weights (2026-10-28, active)
OpenAI allows classifier training on GPT outputs only if the model is not distributed. Publish code, data card and aggregate results; do not publish weights.
**Would change it:** only if no GPT (or Claude) outputs are in the training data.

### D6. Gemma 4 only, from official sources (2026-10-28, active)
Use Gemma 4 checkpoints only (Apache 2.0, verified at gemma.ai/resources/apache-2/). Do not use Gemma 1-3 or variants (older custom terms). Download from Google's official Hugging Face/Kaggle collections and copy the repo's LICENSE (and NOTICE) into `docs/licenses/`.

### D7. Probes are benign; the swap proxy is private (2026-10-28, active)
Both providers prohibit circumventing safeguards. Probes are math, code, instruction-following and formatting only; no jailbreak or safety-testing prompts. The swap proxy serves only my own runner with my own keys and is never exposed publicly.

### D8. Budget and spending limits (2026-10-28, active)
Draft budget about $1,432 projected against a $2,500 cap (see `MPTbudget.xlsx`). Token counts, GPU price and hosting cost are assumptions to replace after the W2.3 pilot. Scheduler hard stop at 80% of the cap. Provider-side hard limits set: OpenAI $100, Anthropic $20 (raise before the scheduler goes live).
**Would change it:** measured token counts and real GPU prices.

### D9. Evaluation rules (2026-10-28, active)
Write the evaluation plan before collecting data. Tune on dev data only; report once on frozen test sets. Nothing goes in the README, posts or resume unless it comes from a logged run traceable in a claims audit.

### D10. Storage (2026-10-28, active)
SQLite for development and for the scheduler, one database file. Structured columns for analysis (provider, model, probe, score, tokens, cost, latency, timestamps) plus one `raw_response` column holding the full API response as JSON text. Keep the schema Postgres-compatible, and use numbered migration files so schema changes in later weeks don't break old data. The database file is never committed to git.
Why: one file means easy backup and easy reproduction; I already know SQL; the data is row-shaped and analysis is mostly GROUP BY.
Would change it: multiple processes needing to write at the same time, or the file growing beyond what one machine handles comfortably.

### D11. Calls with no reply are not scored (2026-11-04, active)
A call that returns nothing (timeout, API error, no output) gets no `scores` row. The failure goes in `calls.error`. The scorers return `passed = None` with `details = {"reason": "no reply"}`, so it can't be stored as a 0 by accident (`scores.score` is NOT NULL).
- **Pass rate** = passed / calls that returned a reply. An empty string counts as a reply and is scored (a fail).
- **Error rate** = calls with `calls.error` set / all calls. Tracked and reported separately, per provider and category.
Why: an outage or timeout says nothing about answer quality. Counting it as a wrong answer would show a "quality drop" during every provider incident, and it would hide a real quality change inside the outage noise. A rise in error rate is still a signal worth detecting, just a different one.
Cost: if a provider starts failing only on the hard probes, the pass rate looks better than it should. The separate error rate is what exposes that, so it must always be shown next to the pass rate.
Also decided here: `scores.details` is always a JSON object. `json_match` logs `{"reason", "fenced"}`, `numeric_match` logs `{"reason", "value"}`, `rule_check` logs one entry per rule. Fenced JSON replies are accepted. `starts_with` and `ends_with` compare whole words, ignoring case.
**Would change it:** if the error rate turns out to depend on the probe content (for example long outputs timing out), pass rates over returned calls would be biased, and I would need to treat those errors as failures.

### D12 Numeric Scoring: Different Failures

For numeric probes, Cluade fails to give bare numeric answers (although it gives true answers) and luna fails to give correct answers. However for numeric scoring, both giving correct and bare numeric answers are required. Therefore, I won't intervene anything.

### D13 Heldout Probes

I'll create those in W4. Until then, all the written probes' split is "scheduled".

### D14 DB Created

I created the DB and loaded all the probes to it. Therefore PROBES MUST BE THE SAME NOW ON. They shouldn't be changed!!!

### D15 VPC

Using DigitalOcean as the VPC.

### D16 SCHEDULED JOBS

Using systemd for scheduled runs. Both for backup (once a day) and for runs (twice a day). I started collecting data on 13 Nov.

### D17 CLAUDE REASONS, GPT DOESN'T (I DON'T ALLOW)

I turned off the reasoning of openai models because of budget concerns. However, I forgot to do the same with Anthropic models (Opus 5.5 ddoesn't even have that option). So Claude costs ~x10. HUGE difference. So I remove the Opus model.

### D18 SOME PROBES ARE TOO EASY (CODING & STRUCTURED)

All models achieve 100% accuracy. Therefore, I cannot set a threshold for pass rate. I need more complicated and trickier probes.

I won't change v1 probes, I'll add new v2 probes for those.

### D19 LATENCY THRESHOLD

Deciding on latency threshold would be a mistake (burst and real data sd's are almost identical). I need more data for that so I'll wait for real data to accumulate more. There's no rush.

### D20 NAIVE BASELINE: TWO-SIDED, drop_points CHOSEN PER USE (2026-11-22, active)

`src/baselines/naive.py` is the fixed-threshold baseline the real pass-rate detector has to beat. It alarms when the pass rate moves by `drop_points` or more. It does not read `thresholds.json`, and it has no p-value or CI.

- **Same data as the real detector.** It uses `load_window`, the common-probe rule (`restrict_to_common_probes`, now shared with `detect_windows`) and `compare_windows`. A baseline that sees different data is not a fair comparison.
- **Mode: two-sided** (`drops_only=False`, `abs(effect) >= drop_points`). The real detector alarms on moves in both directions, so the baseline does too. Using drops-only would make the two detectors answer different questions. `drops_only=True` stays available (`effect <= -drop_points`) for a side comparison, but it is not the one used in the headline comparison.
- **`drop_points` has no default.** I choose it every time and write it in the table below, with the reason, before looking at the comparison result.
- Boundary: `>=`, so a move of exactly `drop_points` alarms (the real detector uses `>`). A 1e-9 float guard stops a true -5.0 from slipping under as -4.9999999.
- Statuses: `no_data` (empty window or no shared probes), otherwise `ok` or `alarm`. Never `not_enough_power`, because it makes no claim about power.

| Date | drop_points | Mode | Used for | Why this number |
|---|---|---|---|---|
| 2026-11-22 | 5.0 | two-sided | Naive vs pass-rate detector comparison | Round fixed number, chosen before looking at any comparison result. |

### D21 MIN MEAN TOKENS (OUTPUT LENGTH)

Chosen 5 after checking: no probe under 5; 10 would remove all OpenAI math probes

### D22 DON'T ALERT IF THRESHOLD IS UNKNOWN.

Some models' threshold for some categories are unknown as the experiments couldn't catch a meaningful variation. That's alright. It can be caused by limited data, or something else. Doesn't matter. I'll check the thresholds of the detectors constantly and if something changes, extra alerts can be added.

### D23 OUTPUT CONTEXT DETECTION METHOD

Six features per reply, per-probe change in the mean, effect = mean over probes seen in both windows, and a limit per feature of M × sd_f × √(10/12). M is the 99.8th percentile of the largest |effect|/sd across features, so the whole detector runs at 1-in-500 per check. It alarms when any feature is over its limit, and the reason names those features.

The statistic is the largest ratio of effect to limit, and threshold is 1.0. No p-value or CI for this detector (threshold only).

### D24 REASONING COUNTS AS OUTPUT TOKEN

Output-text detector. It catches changes in the shape of replies (format, markdown, structure) that pass rate and length miss, using six cheap reply features, each with a limit set from burst noise. The limits share one multiplier, so the whole detector runs at about 1 false alarm in 500 per check. Features that never vary in the burst are not checked, which is a known blind spot (e.g. markdown on GPT models). Synthetic shifts are caught at 10-25% of replies changed on most keys; on real data it stayed quiet where only reasoning tokens changed (Haiku), as intended.

### D25 CUSUM AND A LONGER WINDOW: NOT BUILT
Persistent shifts smaller than the 4-run detection limit may go unalarmed until they grow. Accepted for now. Revisit if real runs or the W3.3 validation show slow drifts.