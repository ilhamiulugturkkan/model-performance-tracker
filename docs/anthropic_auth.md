# Anthropic support replies on training a classifier with Claude outputs

**STATUS: NOT authorization. A human support agent ("DV") has now confirmed Support cannot provide written authorization or legal interpretation.** Both replies below came from Anthropic's AI support assistant. The second states it cannot provide formal legal authorization and recommends contacting Anthropic's legal or partnerships team via a human agent. Treat them as evidence of Anthropic's published position, not as permission.

**Received:** [fill in date]  
**Sender address / channel:** [fill in: exact From address, and whether it came from the support team or the AI assistant]  
**My request:** asked whether a discriminative classifier that predicts which model produced a text may be trained on Claude API outputs (under Commercial Terms, own account), given the Usage Policy effective Nov 12, 2026.

## Reply (copied as received)

Hi Ilhami,
Your discriminative classifier for detecting which model produced text falls under the permitted uses in our policy. This type of specialized classifier is explicitly allowed as a non-competing model.
Since you're using the Claude API under Commercial Terms, your outputs belong to you, and you can use them to train models that don't compete with Claude, like sentiment analysis tools, content categorization systems, and classifiers.
Your project fits this category because:

* It's a specialized classifier, not a general-purpose chatbot or text generation model
* You're not using Claude outputs as generation targets
* You're not publishing the output dataset itself

You can proceed with your research project. The key restriction is that the classifier must remain non-competitive with our models, meaning it shouldn't be designed for open-ended text generation or function as a general-purpose chatbot.
If you do publish the classifier code, just ensure it maintains these boundaries and doesn't enable others to build competing models.

## Conditions to follow

1. Classifier only: no open-ended text generation, no general-purpose chatbot.
2. Claude outputs are never used as generation targets.
3. Do not publish the Claude output dataset.
4. If classifier code is published, it must not enable others to build competing models.

## Second reply (copied as received)

Ilhami, I'm an AI support assistant and can't provide formal legal authorization on behalf of Anthropic. What I can confirm is that your discriminative classifier fits within the permitted uses described in our policy: specialized classifiers are explicitly allowed as non-competing models.
Regarding releasing the trained classifier weights: our policy permits using outputs to train non-competing models like classifiers, but doesn't specifically address weight distribution. Since this involves interpretation of our Usage Policy terms, I'd recommend reaching out to our legal or partnerships team directly for written confirmation on weight release.
You can continue this conversation with a human agent who can help connect you with the right team for formal authorization.

## Gaps (resolved: the first reply was from the AI assistant; it is not authorization)

- Unknown whether a person on the support team wrote it or the AI assistant did.
- It does not mention the Usage Policy clause requiring prior authorization.
- It does not address releasing trained classifier weights. Treat weights as NOT authorized.

## Suggested follow-up (one line)

"Thank you. Please confirm that this reply is Anthropic's prior authorization under the Usage Policy for training a discriminative source-identification classifier on Claude outputs, and whether releasing the trained classifier weights is also permitted."


## Third reply: human support agent (signed "DV"), copied in summary

- Support cannot provide written authorization or a legal interpretation of the Usage Policy or Commercial Terms, so it cannot confirm that a specific use, or release of trained weights, is permitted.
- Pointed to the Commercial Terms and Usage Policy, including requirements on using outputs to develop or train models.
- Said that where the Usage Policy requires prior authorization, **the policy itself lists the contact route** for the request; asked me to submit through that channel with the same detail (discriminative classifier only, no generation, no dataset publication, open-source research use).
- Recommended having my institution's legal or research-compliance office review the terms and project plan.

**My check (Usage Policy page read 2026-10-28, effective Nov 12, 2026):** I did not find a contact route for prior-authorization requests on that page. The only contact listed is usersafety@anthropic.com, which is for reporting inaccurate, biased or harmful model outputs. The only application programs listed are the Cyber Verification Program and the Life Sciences Verification Program, which do not apply. So the pointer in the reply could not be matched to anything on the page I read.

**Outcome:** no authorization has been obtained and none is available through Support. Default stands: no training on Claude outputs.