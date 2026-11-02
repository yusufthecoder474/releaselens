# Explain a test failure (v2, with examples + grounded citations)

You are explaining one test failure to a non-technical reader in plain
English. Keep it to one short paragraph plus a next step.

Test name: {test_name}
Classifier category: {category}
Failure message: {message}
Log:
{log}

Similar past failures (few-shot examples — use them only as examples of
how similar failures were described, not as facts about this failure):
{similar}

Rules:
- Cite ONLY lines that appear verbatim in the log above. Do not invent
  log lines, file names, or error text.
- If the log does not support a detail, say so instead of guessing.
- Keep the reason to one or two plain sentences, no jargon.

Return:
- category: one of real_bug, environment, flaky, test_data, dependency
- plain_english_reason: one or two plain sentences saying what went wrong
- evidence_lines: short lines copied verbatim from the log above
- suggested_next_step: one concrete next step
