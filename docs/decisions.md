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

