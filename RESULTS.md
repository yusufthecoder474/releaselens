# RESULTS.md — ReleaseLens (honest numbers only)

Every number below was measured in this environment on 2026-11-02.
Nothing was estimated or invented. Anything not run says "not run".

## Environment

- Python 3.12.3, project venv (`~/venvs/releaselens`)
- scikit-learn 1.9.1, ChromaDB 1.5.9, LangGraph 1.2.19, LangChain 1.4.7,
  FastAPI 0.162.0, SQLAlchemy 2.1.6, pytest 9.2.0
- No `LLM_API_KEY` was supplied, so all LLM paths are "not evaluated".

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
python scripts/evaluate.py
python scripts/measure_reports.py
```

## Embedding actually used: TF-IDF (MiniLM fallback documented)

The design prefers sentence-transformers all-MiniLM-L6-v2. It was not
available at evaluation time:

1. First install attempt failed: `ERROR: Could not install packages
   due to an OSError: [Errno 28] No space left on device` — pip used
   the small /tmp filesystem for unpacking the large torch wheels.
2. Second attempt with `TMPDIR` on the home disk was still downloading
   after ~30 minutes when the build finished, so the evaluation was run
   with the documented fallback and the install was stopped.

The app therefore uses the scikit-learn TF-IDF embedding function
(`app/embeddings.py`) for ChromaDB in this environment. The code still
tries MiniLM first on every start, so on a machine where it is
installed, MiniLM is used automatically (collection
`past_failures_minilm` instead of `past_failures_tfidf`).

## (A) scikit-learn baseline — GroupKFold by template_family

Method: TF-IDF (1–2 grams) + LogisticRegression, 5-fold GroupKFold over
180 rows grouped by `template_family` (60 families), so near-identical
templates cannot leak between train and test.

- Accuracy: **0.7611**
- Macro-F1: **0.7559**
- Confusion matrix (rows = true, columns = predicted; order:
  real_bug, environment, flaky, test_data, dependency):

```
[[34,  1,  0,  1,  0],
 [ 5, 27,  3,  0,  1],
 [ 5,  1, 28,  1,  1],
 [ 1,  1,  1, 29,  4],
 [ 3,  7,  0,  7, 19]]
```

Hardest category: Dependency (19/36 correct), most often confused with
Environment (7) and Test data (7).

## (B) Retrieval-only kNN over Chroma — same folds

Method: per fold, index the train fold in an ephemeral Chroma
collection (TF-IDF vectors, cosine), query each test item, majority
vote of top-5 neighbours (ties -> nearest neighbour's category).

- Accuracy: **0.7556**
- Macro-F1: **0.7532**
- Confusion matrix (same label order):

```
[[32,  2,  0,  1,  1],
 [ 2, 28,  2,  1,  3],
 [ 3,  0, 31,  2,  0],
 [ 1,  2,  1, 25,  7],
 [ 2,  7,  0,  7, 20]]
```

## (C) LLM + retrieval, (D) full agent, prompt v1 vs v2

- **LLM not evaluated** (no key supplied). The code paths exist
  (`app/agent/graph.py`: LLM structured-output explain, LLM-guided
  investigation loop, prompt versions in `/prompts`), but no number is
  claimed for them. With a key, run the explain node over a fixed
  failure set with `PROMPT_VERSION=v1` and `v2` and compare grounding
  rate and invalid-category rate before the guard.

## Grounding rate (no-key template path)

Measured with `scripts/measure_reports.py` over the 4 sample reports.
Template evidence lines are copied verbatim from the log, so grounding
is exact by construction — and the measurement confirms it:

- **Grounding rate: 1.0000 (29/29 evidence lines verified)**
- Per sample: healthy 0 lines (no failures), flaky_heavy 6/6,
  server_outage 16/16, real_bug 7/7. Unverified (dropped) lines: 0.

## Invalid-category rate before the guard

- **Not evaluated** (key only: only the LLM can produce an invalid
  category; the template path only emits enum values). The guard itself
  is unit-tested: an invalid category falls back to the classifier's
  category, and a fake evidence line is dropped and counted as
  unverified (`tests/test_verify.py`).

## Release-decision check: 10/10 agreement

10 handcrafted runs with expected verdicts derived from `rules.yaml`
(`scripts/evaluate.py`, function `decision_check`):

| Run | Expected | Got | Rule |
|---|---|---|---|
| all_pass | GREEN | GREEN | all_clear |
| flaky_only_high_pass | YELLOW | YELLOW | only_flaky_or_environment |
| env_only_high_pass | YELLOW | YELLOW | only_flaky_or_environment |
| pass_rate_95_mixed | YELLOW | YELLOW | pass_rate_90_97 |
| pass_rate_90_boundary | YELLOW | YELLOW | pass_rate_90_97 |
| pass_rate_89 | RED | RED | pass_rate_below_90 |
| critical_real_bug | RED | RED | critical_real_bug |
| critical_real_bug_high_pass | RED | RED | critical_real_bug |
| dependency_mid_pass | YELLOW | YELLOW | pass_rate_90_97 |
| pass_rate_97_boundary | GREEN | GREEN | all_clear |

## Report-generation time

Measured with `scripts/measure_reports.py`: one warm-up run (untimed),
then one timed pipeline run per sample (no-key mode, temp DB/Chroma):

| Sample | Verdict | Time |
|---|---|---|
| healthy | GREEN | 0.054 s |
| flaky_heavy | YELLOW | 2.884 s |
| server_outage | RED | 0.140 s |
| real_bug | RED | 0.086 s |
| **Average** | | **0.791 s** |

The flaky_heavy run is slowest because it takes the investigate route
(lowest confidence below 0.75) with 3 tool calls.

## Sample verdicts (live server smoke test)

Running `uvicorn app.main:app` and calling the real endpoints:

- `GET /health` -> `{"status": "ok"}`
- `GET /` -> upload page containing "Try with a sample"
- `POST /api/analyze` with `sample=server_outage` -> RED,
  rule `pass_rate_below_90`, pass rate 40.0%, group
  "Server/environment problem"
- `POST /report` with `sample=real_bug` -> RED report page

## Tests passing

- **26 passed, 0 failed** (`python -m pytest tests/ -q`), covering:
  parser, classifier, verify guard (invalid category, fake evidence),
  release rules, SQL flaky query, tool-error handling (unit + full
  pipeline with a failing tool), API (`/`, `/report`, `/api/analyze`,
  `/health`, `/api/runs`, history), full no-key pipeline (RED real-bug
  sample, GREEN healthy sample).

## Not run

- Docker build/run: **not run** (Docker is not installed in this VM).
  Dockerfile + docker-compose.yml are provided but untested here.
- GitHub Actions CI run: **not run** (no GitHub connection; `gh auth
  status` reports not logged in, so no push to a `releaselens` repo
  was made).
- LLM methods (C), full agent (D), prompt v1 vs v2: **LLM not
  evaluated** (no key).
- Load testing beyond the 4 samples: **not run**.
