# Provider compliance notes

Read on: **2026-10-27/28**. Terms change; re-check each source before launch (Jan 2027) and update the dates.
Prepared by Claude (an Anthropic model) from the sources listed. The Anthropic rows have a conflict of interest, so the owner should verify them against the linked primary pages. Not legal advice.

**Two policy changes take effect during the project:** Anthropic Usage Policy (effective **Nov 12, 2026**) and OpenAI Usage Policies (effective **Nov 20, 2026**). Everything below assumes the new versions.

## Chosen providers

| Provider | Role in project | Verdict | Primary source read |
|---|---|---|---|
| Anthropic (Claude) via API | Drift monitoring (statistical detectors). Fingerprint class **only after formal confirmation** | Monitoring allowed. Classifier training on outputs: **NOT formally authorized yet** (see rule 2) | Commercial Terms (anthropic.com/legal/commercial-terms); Usage Policy (anthropic.com/aup, effective Nov 12, 2026); help article support.claude.com/en/articles/12326764; two replies from Anthropic's AI support assistant (saved) |
| OpenAI (GPT) via API | Drift monitoring; fingerprint class only if weights are never distributed | Allowed with limits | OpenAI Services Agreement v.010126 (cdn.openai.com/osa/openai-services-agreement.pdf); Sharing & Publication Policy; Usage Policies (effective Nov 20, 2026; read via search excerpts of the official page, full page fetch was blocked) |
| Google Gemma 4 (open weights, **self-hosted**) | Ground-truth experiments, no-change control, fingerprint training | Allowed (Apache 2.0, verified) | gemma.ai/resources/apache-2/ (Google DeepMind; Apache License 2.0 text governing Gemma 4). Same text at ai.google.dev/gemma/apache_2. ai.google.dev/gemma/terms is the *older* custom license for Gemma 1-3 and variants |

## Not used (and why)

| Provider | Reason |
|---|---|
| xAI (Grok) | Enterprise Terms (updated May 12, 2026) bar probing/benchmarking, training on Output, and distilling model behavior. |
| Meta (Muse Spark 1.1) | Meta's own terms could not be read; they changed several times Aug-Sep 2026; US-only public preview. Excluded as unclear. |
| Hosted open-weight APIs (e.g., Together AI) | Only secondary summaries read. Host terms add a layer. Skipped in favor of self-hosting. |

## Rules for this project

1. **Monitoring Claude and GPT through their official APIs is allowed.** No clause banning benchmarking was found in the Anthropic or OpenAI agreements or policies. OpenAI's Usage Policies explicitly allow research, evaluation and benchmarking within their boundaries.
2. **Anthropic, training on outputs: not formally authorized; default is to NOT train on Claude outputs.** The Usage Policy (effective Nov 12, 2026) says using outputs to train an AI model needs prior authorization from Anthropic. Anthropic's published help article (Mar 16, 2026) says outputs may train non-competing models such as classifiers. Anthropic's AI support assistant twice said a discriminative source-identification classifier fits the permitted uses, but in its second reply it stated it **cannot give formal legal authorization**, that weight release is not specifically addressed, and that formal confirmation must come from Anthropic's legal or partnerships team via a human agent (saved in `docs/anthropic-authorization.md`). A human support agent then confirmed (third reply) that Support cannot provide written authorization or legal interpretation, and pointed to a contact route said to be listed in the Usage Policy; **no such route appears on the Usage Policy page read on 2026-10-28** (only usersafety@anthropic.com for reporting harmful outputs, and two unrelated verification programs). So the evidence is: published intent that classifiers are allowed, but no formal authorization. **Rule: do not collect Claude outputs for classifier training unless a human at Anthropic confirms in writing before W5.1 (Nov 30).** If it is not confirmed, Claude stays monitoring-only (statistical detectors). If it is confirmed, the conditions are: discriminative classifier only (no open-ended generation or chatbot), Claude outputs never used as generation targets, no published Claude output dataset, published code must not enable competing models, and no public weights unless Anthropic says so.
3. **OpenAI, training on outputs:** outputs may be used for classifiers only if the resulting model is not distributed or made commercially available to third parties (Services Agreement "Permitted Exception", which the new Usage Policies expressly preserve). **Never publish weights of any model trained on GPT outputs.** The new policy also bars using outputs to build training data or signals for models meant to substitute for OpenAI's, even if you don't train the resulting model.
4. **No red-teaming.** Both providers prohibit circumventing safeguards (OpenAI also lists unsolicited safety testing; Anthropic lists jailbreaking and prompt injection without authorization). Keep every probe benign: math, code, instruction-following, formatting. No jailbreak or refusal-boundary probes.
5. **The swap proxy is private.** OpenAI's new policy requires approval for services that let third parties use its models (routers, gateways), and Anthropic bars reselling or proxying access. The proxy serves only your own runner with your own keys. Never expose it publicly, and the public CLI calls providers directly with the user's own key.
6. **One account per provider; no automation in account creation; no multi-account tricks** to avoid limits. Respect rate limits and supported-regions policies.
7. **Sharing outputs:** OpenAI's Sharing & Publication Policy asks for human review, attribution and AI-generated disclosure when sharing generated content, and welcomes research publications. The dashboard shows aggregates and short excerpts only, not bulk raw outputs.
8. **Gemma 4 only.** Apache 2.0 allows use, modification and redistribution and places no restriction on Outputs. Obligations: give recipients a copy of the license, mark modified files, keep any NOTICE file, no use of Google/Gemma trademarks as a product name. Do not use older Gemma checkpoints (1-3, 3n, variants): they fall under the older custom terms. Download only from Google's official collections linked on gemma.ai (Hugging Face: huggingface.co/collections/google/gemma-4; Kaggle: kaggle.com/models/google/gemma-4).
9. **Public weights:** release classifier weights only if every provider whose outputs were used in training allows redistribution. Default: code + data card + results, no weights.
10. **No payments, no monetization.** Free and public only.

## Design impact

- **Default plan (no Anthropic confirmation needed):** fingerprint classes are **Gemma 4 sizes** (clean license) and **GPT** (private weights only). Claude is monitoring-only with statistical detectors.
- **If a human at Anthropic confirms in writing:** add Claude as a class within the conditions in rule 2.
- Because GPT outputs are in the training data, the default is **no public weights**: publish code, data card and results.
- Validation (known changes, false-alarm rate) is done on self-hosted Gemma 4.

## Open items

- [x] Anthropic help article on training with outputs (read; superseded in effect by the Usage Policy clause, see rule 2).
- [x] Google Gemma 4 license (Apache 2.0) read at gemma.ai and ai.google.dev.
- [x] Anthropic Usage Policy and OpenAI Usage Policies skimmed for probing restrictions (rules 4-6).
- [x] Asked Anthropic support; the AI assistant replied twice and said it cannot give formal authorization (saved).
- [ ] **Optional, by Nov 30:** continue with a human agent and ask for written confirmation from Anthropic's legal or partnerships team (message drafted). Default if no answer: Claude not used for classifier training.
- [ ] At Gemma download: confirm the repo's own LICENSE file says Apache 2.0; copy it (and any NOTICE) into `docs/licenses/`; record repo and commit hash.
- [ ] Re-check all sources before launch (Jan 2027) and update the dates above.