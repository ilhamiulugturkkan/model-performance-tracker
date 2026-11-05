# Evaluation plan

**Written:** 2026-10-28, BEFORE any probe data exists. **Status:** draft until the `[FILL LATER]` blanks are completed and committed.
**Rule:** this plan is never edited silently. Changes go in the Addenda section at the bottom, with a date and a reason.

## 1. What this plan is for

Decide, before seeing any data, how success is measured, so I can't pick flattering metrics afterward.

## 2. Claims policy (the one rule)

A number may appear in the README, the dashboard, a post or my resume **only if** it comes from a logged run and appears in `docs/claims-audit.md` with: the claim, the number, the source file or run id, the script that produced it, and the date verified. Anything that can't be traced gets cut or reworded.

Wording rules:
- **Never write "the model did not change" or "unchanged."** The most I can say is "no change detected above X points, with N samples." That X is the detection floor (section 4).
- Say "on my probes, with N providers," never "LLMs" in general.
- Changes I cause myself are labeled "simulated" everywhere.

## 3. Vocabulary

- **Probe:** one fixed test question sent to a model, with an automatic grader (code), for example "17 x 23" graded against 391.
- **Noise:** random wobble between identical runs of the same model. Measured first, in W3.1.
- **Ground truth:** the answer key. For changes I cause myself, the proxy writes the exact time and type of each change to a log before the run. Detectors never see it.
- **Detection floor:** the smallest change my detector can reliably catch, per provider and probe category.

## 4. Part A: drift detection

**What counts as a change:** a shift in any of: pass rate per probe category, output length, latency, or the distribution of output text, between a reference window and a current window.

**Metrics (always all three, never one alone):**
1. **Detection rate:** of the real changes, how many were flagged.
2. **Detection delay:** time (and number of calls) from the change to the alarm.
3. **False-alarm rate:** alarms per N checks when nothing changed. Target: 1 false alarm per 500 checks. A check is one comparison of a reference window against a current window, for one provider, one probe category and one detector.

**What counts as a detection:** an alarm counts as a detection only if it fires within T after a change I caused. T is a number of calls, fixed in the W3.3 addendum after the W3.1 noise report. An alarm that fires before the change, or after T has passed, counts as a false alarm, not a detection. Real, unplanned provider changes are never scored this way, because their true timing is unknown (see "Real-world monitoring").

**Detectors:** pass-rate shift, output-length shift, latency shift, output-text distribution shift, plus a **naive baseline** (alarm when any category's pass rate drops by more than a fixed amount). Statistical test choices and window sizes: `[FILL LATER, after noise report]`. Control for many simultaneous tests: `[FILL LATER]`.

**Baseline to beat:** the naive threshold detector. "Beats" means: `[FILL LATER: higher detection rate at the same false-alarm rate, by at least ___ points]`.

**Ground-truth experiments (pre-registered; run in W4):**

| Where | Change I cause | Why |
|---|---|---|
| Self-hosted Gemma 4 | switch between Gemma 4 sizes; change temperature; edit the system prompt | known changes with exact timing |
| Self-hosted Gemma 4 | **no-change controls** (nothing changes) | any alarm here is a false alarm; gives the false-alarm rate |
| Claude (my own key) | switch between two models of the same API (e.g., Haiku to Sonnet); change temperature; edit the system prompt | known changes on the real provider's infrastructure |
| GPT (my own key) | same three kinds of change | same reason |

- Repeats per case: 20. Change times randomized. Controls interleaved with change runs, not saved for the end.
- **Dev/test split:** tune detector settings on the dev runs only; freeze them (git tag); evaluate once on the test runs.
- Use the held-out probes, never the scheduled ones, for these experiments.
- Report detection rate, delay and false-alarm rate **per case type and per window size**, with confidence intervals. If the naive baseline wins somewhere, say so.

**Detection floor (reported per provider and category):** from the noise measurement and the experiments, the smallest pass-rate drop detectable at the stated false-alarm rate. Different providers will have different floors.

**Real-world monitoring (Claude and GPT):** the scheduler data is monitored with the validated detectors. Real events found (for example the model name returned by the API changing) are reported as **supporting observations only**, because the true timing of a real change is unknown. They don't count as validation.

## 5. Part B: fingerprint classifier (PyTorch)

**Task:** given a piece of text, predict which model wrote it.

**Classes:** Gemma 4 sizes and GPT. **Claude only if a human at Anthropic confirms in writing before W5.1 (Nov 30)** (see `docs/compliance.md`, rule 2). No public weights.

**Splits:** by prompt (all samples of a prompt stay in one split), one Gemma size held out entirely as the "unseen version" test, plus a "later in time" test using scheduler data from weeks after training data was collected. Test sets are frozen (file hashes recorded) and evaluated **once**, after the final model is chosen.

**Baselines:** length-only model, stylometric features (punctuation, formatting, sentence length), TF-IDF plus logistic regression.

**Metrics:** accuracy, macro-F1, per-class recall, confusion matrix, calibration; confidence intervals computed across prompts (not across samples); mean and spread over at least 3 seeds.

**Success threshold (decided now, not after the results):** the fine-tuned encoder counts as better only if it beats the best baseline by at least 3 macro-F1 points on the standard test AND the gap exceeds the seed-to-seed spread. If not, I report that a simple model is enough.

**Leakage checks:** no near-duplicate text across splits; no scheduled-probe prompts in training; self-identification text ("As [model name]...") removed or masked; length-only baseline reported as the shortcut to beat.

## 6. Risks and what I'll do

| Risk | What it means | Response |
|---|---|---|
| Gemma no-change control raises alarms | thresholds are miscalibrated | fix calibration before anything else counts; this is a stop-everything problem |
| Detector misses a subtle change | detector has a floor | report the floor; this is a finding, not a failure |
| Noise swamps signal | normal wobble hides a small real drop | more repeats / longer windows; report the minimum detectable shift from W3.1 |
| Detector works on Gemma but not on Claude/GPT | validation doesn't transfer | the known-change tests on Claude and GPT catch this; report per-provider floors |
| Classifier fails to generalize across prompts | it memorized prompts or style of a few tasks | split by prompt, vary prompts, report which prompt types fail; a bigger model alone won't fix it |
| A provider changes a model during an experiment | contaminates that run | log it (returned model name, dates); mark affected runs; rerun if needed |
| Provider retires a model (OpenAI has announced deprecation dates for older ones) | monitoring series breaks | pin current model IDs; record retirement dates |
| Budget overrun | scheduler hits a hard stop | provider limits plus the scheduler's 80% hard stop; limits raised before W2.4 (Linear ILH-329) |
| Terms change | compliance breaks | re-check all sources in early Jan 2027 |

## 7. Schedule note

Academic load: one final exam, date 12-15-2026. Weeks with exam or deadlines: only the Sunday review and that week's must-do issue are mandatory.

## 8. Blanks I must fill before this is final

These are fixed in a dated addendum after the W3.1 noise report and before any W4 experiment runs.

1. **Statistical tests and window sizes** for each detector, plus how I control for many simultaneous tests.
2. **"Beats the naive baseline" margin:** for example, higher detection rate at the same false-alarm rate, by at least ___ points.
3. **T for the controlled experiments:** how many calls after a change an alarm may fire and still count as a detection (see "What counts as a detection" in section 4).

Already decided: false-alarm target (1 per 500 checks), repeats per case (20), classifier success margin (3 macro-F1 points), production T (4 scheduled runs), final exam date (Dec 15, 2026).

## Addenda (dated changes to this plan)

**2026-11-04: pass-rate denominator and error rate (see D11).** Pass rate is computed over calls that returned a reply. Calls with no reply are not scored and are counted in a separate error rate (`calls.error` set / all calls), reported next to every pass rate. Reason: provider outages should not look like quality drops. This does not change the three detection metrics. Error rate is reported as its own series and is not one of the four detector signals unless a later addendum adds it.