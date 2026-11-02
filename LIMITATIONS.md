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
  will be lower than the reported 0.81.
- The hardest category is Dependency (23 of 36 correct with the
  shipped word+char classifier, up from 19 of 36 word-only):
  dependency messages overlap with environment and test-data wording
  (install failures, missing packages, version text). See the error
  analysis in RESULTS.md.

## The LLM paths were not evaluated

- No `LLM_API_KEY` was available in this session (verified repeatedly;
  see RESULTS.md), so the LLM explain node, the LLM-guided
  investigation loop, the LLM + retrieval method (C), the full agent
  (D), and the v1 vs v2 prompt comparison are all "not evaluated".
  The no-key template path is what was measured. The evaluation code
  and a disk cache exist (`scripts/llm_evaluate.py`) and are tested
  with a fake LLM, but no real LLM number is claimed.
- In no-key mode, explanations are template sentences from the
  classifier category plus the nearest past failure. They are accurate
  in category and grounded in the log, but generic — they do not reason
  about the specific failure the way an LLM explanation might.

## Embeddings: MiniLM active, but worse than TF-IDF on this data

- MiniLM (all-MiniLM-L6-v2) is now installed and active; TF-IDF is
  the documented fallback if MiniLM cannot load. The collection name
  includes the embedding name (`past_failures_minilm` vs
  `past_failures_tfidf`), so the two never collide.
- Measured honestly on this synthetic data, MiniLM retrieval is
  WORSE than TF-IDF (kNN accuracy 0.6722 vs 0.7556). The memory
  affects only "seen before" context, never the verdict, but a team
  adopting this should re-measure on their own failures before
  assuming semantic embeddings help.
- The MiniLM model is cached per process and encodes are serialized
  behind a lock (see `app/embeddings.py`). That keeps report time
  reasonable (avg 1.759 s) but means concurrent requests queue on
  embedding work. Fine for a demo; not a production serving design.

## "Tests to run first" is a crude heuristic

- The ranking (`app/selection.py`) uses surface token overlap between
  test names and changed file paths, plus recent failure rate and
  flaky status. It does NOT do dependency analysis, does not read
  code, and cannot know that a change in one file breaks a test with
  an unrelated name. Treat it as a triage hint, not a test-selection
  guarantee.
- Tests with no history in the last 10 runs get failure rate 0.0,
  including brand-new tests — which is exactly when you most want to
  run them. The overlap term is the only signal for new tests.
- The weights (2 / 5 / 1) are fixed and were not tuned on any data;
  they are documented, not learned.

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

- `pytest` passes here (41 tests) via the project venv. GitHub Actions
  CI was not run here (no GitHub connection was available: `gh` not
  logged in, connector `not_connected`), so CI is provided but not
  claimed as passing, and no push or README badge exists.
- Docker was not run here: the daemon runs but cannot pull images
  through this environment's egress proxy (TLS handshake timeout to
  Docker Hub, while shell curl succeeds), and `docker compose` is
  unavailable. See RESULTS.md for the exact attempts. The Dockerfile
  and compose file are provided but untested in this environment, and
  as written they would run the TF-IDF fallback (sentence-transformers
  is not in `requirements.txt`).
