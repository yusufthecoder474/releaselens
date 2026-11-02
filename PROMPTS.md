# PROMPTS.md — prompt versions for the explain node

Prompts live in `/prompts` as versioned files. The explain node loads
one of them (default `v2`, override with env var `PROMPT_VERSION=v1`).
Only the explain node uses an LLM at all: the verdict comes from
`rules.yaml`, never from a prompt.

## v1 — basic (`prompts/v1_explain.md`)

The first version: role ("explain one test failure in plain English"),
the test name, classifier category, failure message and log, and the
required output shape (category, plain-English reason, evidence lines,
next step). No examples, no citation rule. It works, but nothing stops
the model from writing a plausible-sounding log line that is not in the
log — and the verify guard then has to drop it.

## v2 — examples + grounded citations (`prompts/v2_explain.md`)

What changed, and why:

1. **Few-shot examples from retrieved similar failures.** The prompt
   now includes the top similar past failures from ChromaDB as examples
   of how similar failures were described. They are explicitly marked
   as examples only, not facts about this failure. This follows the
   RCACopilot idea (retrieve similar historical incidents, then explain)
   at student scale.
2. **Cite-only-provided-lines instruction.** The prompt now says: cite
   ONLY lines that appear verbatim in the log above, and say so instead
   of guessing when the log does not support a detail. This is the
   prompt-side half of the grounding guarantee; the verify guard is the
   enforcement half (it drops any line that is not verbatim).

## v1 vs v2 comparison

Not evaluated: comparing prompt versions needs an LLM key, and none
was available in this session, so RESULTS.md records "LLM not
evaluated" for the LLM paths, including v1 vs v2. The comparison is
implemented but not run: `scripts/llm_evaluate.py` runs method (C)
once per prompt version on the same GroupKFold folds and the same
fixed failure set, with outputs cached to `eval_cache/llm_cache.json`
so a rerun costs no new calls. When a key is available, run
`python scripts/llm_evaluate.py` and compare, per version: accuracy,
macro-F1, confusion matrix, grounding rate on RAW evidence lines
(before the guard), and invalid-category rate before the guard. Do
not claim v2 is better until those numbers exist: v2's few-shot
examples and cite-only-lines instruction are designed to help
grounding, but on this data that is a hypothesis, not a result.

## No-key mode

With no `LLM_API_KEY`, neither prompt file is used: the explain node
uses a deterministic template built from the classifier result and the
nearest past failure. Its evidence lines are copied from the log, so
grounding is exact by construction (measured 29/29 on the current,
held-out samples).
