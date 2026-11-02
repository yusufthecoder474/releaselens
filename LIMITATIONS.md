# LIMITATIONS.md — ReleaseLens

Honest limits of this student project. Nothing here is a claim that the
system is production-ready.

## Data is synthetic and small

- All 180 labeled failures are synthetic (template families + 30
  hand-written hard cases), and the 20 historical runs are seeded, not
  real CI history. Results describe performance on this synthetic set,
  not on real test suites.
- 180 rows is small. GroupKFold by `template_family` avoids the most
  obvious leakage (near-identical templates in train and test), but
  families still share vocabulary, so accuracy on truly unseen failures
  will be lower than the reported 0.76.
- The hardest category is Dependency (19–20 of 36 correct in the
  confusion matrices): dependency messages overlap with environment and
  test-data wording (install failures, missing packages, version text).

## The LLM paths were not evaluated

- No `LLM_API_KEY` was supplied, so the LLM explain node, the
  LLM-guided investigation loop, the LLM + retrieval method (C), the
  full agent (D), and the v1 vs v2 prompt comparison are all "not
  evaluated". The no-key template path is what was measured.
- In no-key mode, explanations are template sentences from the
  classifier category plus the nearest past failure. They are accurate
  in category and grounded in the log, but generic — they do not reason
  about the specific failure the way an LLM explanation might.

## Embeddings: TF-IDF fallback

- The design prefers sentence-transformers all-MiniLM-L6-v2, but this
  environment could not install it (see RESULTS.md for the exact
  attempts and errors), so ChromaDB uses the documented scikit-learn
  TF-IDF embedding function. Retrieval quality is therefore keyword
  overlap, not semantic similarity. If MiniLM is installed later, the
  collection name changes (`past_failures_minilm`) and the collection
  is re-seeded automatically.

## Rules are simple thresholds

- The verdict rules (critical real bug, pass rate < 90%, only
  flaky/environment, 90–97% band) are deliberately simple so a
  non-technical reader can follow them. Edge case, stated plainly: a
  run with a non-critical real bug at 98% pass rate is GREEN by these
  rules, because neither RED rule fires and neither YELLOW rule fires.
  The report still lists the bug, and a person should confirm.
- "Critical" comes from a JUnit `critical` property or from name
  keywords in `rules.yaml` (login, payment, checkout, auth). A team
  with different critical tests must edit that list.

## Scope limits (by design)

- No chat UI, no login/auth, no multi-user support, no fine-tuning, no
  Kubernetes, no microservices, no dashboard beyond the single report
  page, no cloud deployment (a short guide is not included; run it
  locally or with Docker).
- Sync is synchronous: one upload produces one report in the request.
  Fine for tens of tests; not designed for thousands of tests per run.
- SQLite + a local ChromaDB directory are single-machine stores. They
  are fine for a portfolio demo, not for concurrent team use.

## Environment limits in this VM

- `pytest` passes here (26 tests) via the project venv. GitHub Actions
  CI was not run here (no GitHub connection was available), so CI is
  provided but not claimed as passing.
- Docker was not run here (see RESULTS.md for whether Docker is
  installed in this VM). The Dockerfile and compose file are provided
  but untested in this environment unless RESULTS.md says otherwise.
