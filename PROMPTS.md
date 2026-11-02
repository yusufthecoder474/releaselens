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

Not evaluated in this environment: comparing prompt versions needs an
LLM key, and none was supplied, so RESULTS.md records "LLM not
evaluated" for the LLM paths, including v1 vs v2. The comparison is
designed, not run: with a key, run the explain node over the same set
of failures with `PROMPT_VERSION=v1` and `v2` and compare grounding
rate (share of evidence lines verbatim in the log) and invalid-category
rate before the guard.

## No-key mode

With no `LLM_API_KEY`, neither prompt file is used: the explain node
uses a deterministic template built from the classifier result and the
nearest past failure. Its evidence lines are copied from the log, so
grounding is exact by construction (measured 29/29 on the samples).
