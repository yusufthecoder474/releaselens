# Explain a test failure (v1, basic)

You are explaining one test failure to a non-technical reader in plain
English. Keep it to one short paragraph plus a next step.

Test name: {test_name}
Classifier category: {category}
Failure message: {message}
Log:
{log}

Return:
- category: one of real_bug, environment, flaky, test_data, dependency
- plain_english_reason: one or two plain sentences saying what went wrong
- evidence_lines: short lines from the log that support the reason
- suggested_next_step: one concrete next step
