# RESULTS.md — ReleaseLens (honest numbers only)

Every number below was measured in this environment on 2026-11-02.
Nothing was estimated or invented. Anything not run says "not run".

## Environment

- Python 3.12.3, project venv (`~/venvs/releaselens`)
- scikit-learn 1.9.1, ChromaDB 1.5.9, LangGraph 1.2.19, LangChain 1.4.7,
  FastAPI 0.162.0, SQLAlchemy 2.1.6, pytest 9.2.0
- sentence-transformers 6.2.0, torch 2.15.0 (used on CPU),
  all-MiniLM-L6-v2 (384-dim) — installed during this improvement run
- No `LLM_API_KEY` was available, so all LLM paths are "not evaluated".

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
python scripts/evaluate.py
python scripts/error_analysis.py
python scripts/measure_reports.py
python scripts/llm_evaluate.py   # prints LLM NOT EVALUATED without a key
```

## Embedding actually used: MiniLM (active), TF-IDF fallback kept

The app tries MiniLM first on every start
(`app/embeddings.py:get_embedding_function`) and falls back to TF-IDF
only if MiniLM cannot load. Verified active: a pipeline run prints
"Loading weights" and uses collection `past_failures_minilm`.

Install history (this run): the first attempt failed with
`OSError: [Errno 28]` (pip unpacking into the small 512M /tmp); the
second attempt with `TMPDIR=$HOME/.tmp-pip` stalled on the large CUDA
torch wheels through the slow proxy; the PyTorch CPU index
(download.pytorch.org) timed out through the proxy, so the standard
PyPI install (CUDA-built torch, used on CPU) was used instead. The
MiniLM model (~90MB) downloads to `~/.cache/huggingface` and is cached
per process (see the timing note below).

## (A) Classifier — GroupKFold by template_family (same folds throughout)

Method: 5-fold GroupKFold over 180 rows grouped by `template_family`
(60 families), so near-identical templates cannot leak between train
and test. All three variants use these same folds.

| Variant | Accuracy | Macro-F1 |
|---|---|---|
| Word-only TF-IDF + LogReg (original baseline) | 0.7611 | 0.7559 |
| Word + char TF-IDF + LogReg (**shipped classifier**) | **0.8111** | **0.8092** |
| MiniLM embeddings + LogReg | 0.7444 | 0.7459 |

Confusion matrices (rows = true, columns = predicted; order:
real_bug, environment, flaky, test_data, dependency):

Word-only (original):

```
[[34,  1,  0,  1,  0],
 [ 5, 27,  3,  0,  1],
 [ 5,  1, 28,  1,  1],
 [ 1,  1,  1, 29,  4],
 [ 3,  7,  0,  7, 19]]
```

Word + char (shipped):

```
[[33,  2,  0,  1,  0],
 [ 5, 29,  1,  0,  1],
 [ 3,  2, 31,  0,  0],
 [ 1,  0,  2, 30,  3],
 [ 2,  4,  0,  7, 23]]
```

MiniLM:

```
[[24,  5,  3,  4,  0],
 [ 4, 28,  2,  2,  0],
 [ 3,  5, 25,  3,  0],
 [ 2,  1,  4, 27,  2],
 [ 1,  1,  0,  4, 30]]
```

Note: MiniLM is the best of the three for Dependency alone (30/36,
vs 19/36 word-only and 23/36 word+char) but the worst overall, because
real_bug falls to 24/36 (vs 34/36 and 33/36).

## (B) Retrieval-only kNN over Chroma — same folds

Method: per fold, index the train fold in an ephemeral Chroma
collection (cosine), query each test item, majority vote of top-5
neighbours (ties -> nearest neighbour's category).

| Embedding | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF vectors | 0.7556 | 0.7532 |
| MiniLM vectors | 0.6722 | 0.6750 |

MiniLM confusion:

```
[[24,  3,  4,  5,  0],
 [ 2, 29,  3,  1,  1],
 [ 2,  5, 22,  7,  0],
 [ 5,  1,  6, 23,  1],
 [ 1,  5,  1,  6, 23]]
```

(TF-IDF confusion is unchanged from the first build:
`[[32,2,0,1,1],[2,28,2,1,3],[3,0,31,2,0],[1,2,1,25,7],[2,7,0,7,20]]`.)

Interpretation (plainly): on this synthetic template data, exact
keyword matching beats semantic vectors. The templates share phrasing
across categories ("upstream API", "not found", "timed out"), and
TF-IDF keys on the distinguishing tokens while MiniLM's semantic
similarity blurs them. MiniLM stays active because it is the required
embedding, and the memory affects only "seen before" context, never
the verdict — but on this dataset TF-IDF would be the better
retrieval choice. Whether MiniLM wins on real, varied logs was not
measured and is not claimed.

## (C) LLM + retrieval, (D) full agent, prompt v1 vs v2

- **LLM not evaluated** (no key). `LLM_API_KEY` was not set in this
  session: an initial presence check appeared to show it set, but
  three repeat checks in fresh shells showed it unset, and no LLM
  call was attempted with a key that could not be verified. No LLM
  numbers are claimed.
- The evaluation code exists and is tested with a fake LLM
  (`scripts/llm_evaluate.py`, `tests/test_llm_eval.py`, 4 passed):
  (C) classifies each fold's test items with the LLM using retrieved
  similar failures from that fold's train items only, prompting with
  the CLASSIFIER's category (never the true label, which would leak
  the answer); (D) runs the full agent on a fixed 20-failure XML
  (4 per category); v1 vs v2 run (C) on the same folds. Outputs are
  cached to `eval_cache/llm_cache.json` (gitignored) keyed by method,
  prompt version, model and item, so reruns make no new calls.
- What the fake-LLM tests prove (mechanics only, NOT LLM quality):
  cache reuse (second run: 0 new calls, identical metrics), invalid
  counting (10/10 bogus categories counted and fallen back), and
  grounding computation (verbatim evidence 1/1, invented evidence
  0/1).
- Metrics the real run would report (all not evaluated): accuracy,
  macro-F1, confusion matrix, grounding rate on RAW evidence lines
  (before the verify guard; post-guard grounding is 1.0 by
  construction), invalid-category rate BEFORE the guard, tool-error
  rate (D only), model name, number of LLM calls, total time.
- If the real results come back worse than baseline (A), that will be
  reported plainly with a hypothesis; no such comparison exists yet
  because no real run happened.

## Error analysis (Dependency, weakest in original baseline A)

Original baseline (A): Dependency 19/36 correct, 17 errors.
By predicted category: environment 7, test_data 7, real_bug 3.

Five most common confusions (family -> predicted, with an actual
failure message):

1. `dep_native_lib` -> test_data (3): "OSError: shared library
   libssl.so.1.1 not found". Hypothesis: "not found" wording overlaps
   test-data "file not found" patterns, so the missing-library signal
   is read as a missing fixture.
2. `dep_api_breaking` -> environment (3): "Upstream API now returns
   410 Gone for deprecated endpoint". Hypothesis: upstream HTTP
   status codes look like server-down codes (503/500/502).
3. `dep_api_breaking` -> test_data (2): "Upstream API field 'price'
   changed from number to string". Hypothesis: field-type changes
   read as test-data format mismatches (wrong format, invalid values).
4. `dep_network_install` -> environment (2): "pip install failed:
   index server unreachable". Hypothesis: install-network wording
   shares timeout/DNS vocabulary with environment failures.
5. Seven further pairs tie at 1 each: `hard_25` -> real_bug
   ("Patch release of requests changed redirect behaviour"),
   `dep_native_lib` -> environment ("OSError: libasound.so.2 not
   found"), `dep_missing_module` -> real_bug ("ModuleNotFoundError:
   No module named 'pandas'"), `hard_28` -> test_data ("Prebuilt
   wheel is for wrong platform"), `dep_version_conflict` ->
   environment ("Dependency resolution failed: conflicting versions
   of requests"), `hard_27` -> test_data ("Tests need Python 3.11
   but runner has 3.9"), `dep_version_conflict` -> real_bug
   ("Version conflict: Django 4.2 required but 5.0 installed").
   Hypothesis: patch releases and version conflicts read as product
   behaviour changes, while platform/python-version issues read as
   configuration problems.

One change tried (and KEPT): add character n-gram features
(`char_wb` 3-5) alongside word n-grams (1-2) via FeatureUnion, same
folds. Before: accuracy 0.7611, macro-F1 0.7559, Dependency 19/36.
After: accuracy **0.8111**, macro-F1 **0.8092**, Dependency **23/36**.
The change was applied to `app/classifier.py` and to the (A) pipeline
in `scripts/evaluate.py`, so the shipped classifier is the improved
one. Rationale: character features catch distinctive substrings
(`lib*.so`, version numbers, error class names) that word features
split or miss.

## Demo leakage fix

Before: all 20 failing sample entries reused training messages
(20/20), and 5 also reused training test names (all 5 real_bug
failures: `test_login_valid_user`, `test_checkout_total`,
`test_discount_applies`, `test_modal_open`, `test_import_users`).
The samples were training rows in disguise, so sample confidences
were inflated.

After: `scripts/make_samples.py` rewrites every failing entry with a
novel test name and paraphrased message/log, and fails generation if
any failing name or message matches the training data
(`Leakage check passed: no failing sample reuses a training name or
message.`). Passing test names are exempt (they are never
classified). The sample verdicts are unchanged.

Sample-run confidences AFTER the fix (shipped word+char classifier;
these are the classifier confidences the router uses):

- flaky_heavy (YELLOW, `only_flaky_or_environment`, 90.0%):
  `test_carousel_slide` flaky 0.695, `test_tooltip_hover` flaky
  0.717, `test_spinner_wait` flaky 0.648. Mean -> Medium, routed to
  investigate.
- server_outage (RED, `pass_rate_below_90`, 40.0%): all 12 novel
  environment failures classified environment: 0.617, 0.571, 0.657,
  0.735, 0.687, 0.590, 0.569, 0.383, 0.531, 0.625, 0.436, 0.629.
  Mean -> Medium, routed to investigate.
- real_bug (RED, `critical_real_bug`, 80.0%):
  `test_signin_valid_account` real_bug 0.472 (critical),
  `test_basket_amount` real_bug 0.472 (critical),
  `test_voucher_applies` real_bug 0.328,
  `test_dialog_open` flaky 0.509,
  `test_seed_customers` test_data 0.614. Mean -> Low, routed to
  investigate.
- healthy: GREEN, `all_clear`, 100.0%.

Confidences are lower than with the copied training rows, which is
expected for genuinely unseen inputs; every sample failure still
classifies into its intended category.

## Tests to run first (deterministic, no LLM)

`app/selection.py` ranks the current run's tests when changed file
paths are submitted with the upload (form field `changed_files`, one
per line or comma-separated). Score, with fixed weights:

```
score = 2 * overlap_tokens + 5 * failure_rate + (1 if flaky else 0)
```

- overlap_tokens: lowercase alphanumeric tokens shared between the
  test name and the changed paths, ignoring filler tokens (test,
  tests, py, js, ts, src, spec).
- failure_rate: the test's failure rate over the last 10 runs from
  SQL (`app/flaky.py:failure_rates`); 0.0 for tests with no history.
- flaky: 1 if the test passed and failed in the last 10 runs.
- Ties break by failure rate desc, then test name asc. Top 10 shown.

Shown on the report page (only when changed files are supplied) and
in `/api/analyze` JSON as `tests_to_run_first` (plus a
`changed_files` echo). Verified by the API test: sample `real_bug`
with `changed_files=src/basket.py` ranks `test_basket_amount` first
(overlap 1); with no changed files the list is empty.

## Grounding rate (no-key template path)

Measured with `scripts/measure_reports.py` over the 4 (new) sample
reports. Template evidence lines are copied verbatim from the log:

- **Grounding rate: 1.0000 (29/29 evidence lines verified)**
- Per sample: healthy 0 lines (no failures), flaky_heavy 6/6,
  server_outage 16/16, real_bug 7/7. Unverified (dropped) lines: 0.

## Invalid-category rate before the guard

- **Not evaluated** (key only: only the LLM can produce an invalid
  category; the template path only emits enum values). The guard
  itself is unit-tested: an invalid category falls back to the
  classifier's category, and a fake evidence line is dropped and
  counted as unverified (`tests/test_verify.py`, plus the fake-LLM
  invalid-count test).

## Release-decision check: 10/10 agreement

10 handcrafted runs with expected verdicts derived from `rules.yaml`
(`scripts/evaluate.py`, function `decision_check`) — unchanged and
still passing:

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

| Sample | Verdict | Time (with model cache) |
|---|---|---|
| healthy | GREEN | 0.069 s |
| flaky_heavy | YELLOW | 3.755 s |
| server_outage | RED | 2.633 s |
| real_bug | RED | 0.580 s |
| **Average** | | **1.759 s** |

History for comparison: original build (TF-IDF embeddings, copied
samples) averaged 0.791 s. After switching to MiniLM without a model
cache the average rose to 2.614 s ([1.351, 5.304, 2.300, 1.503]),
because every pipeline run reloaded the ~90MB model. Caching the
model per process (`app/embeddings.py`) brought it to 1.759 s. The
remaining gap vs 0.791 s is MiniLM encode work plus the investigate
route, which the held-out samples now take more often (their
confidences are lower than the copied rows' were).

## Sample verdicts (live server smoke test, final code)

Running `uvicorn app.main:app` and calling the real endpoints:

- `GET /health` -> `{"status": "ok"}`
- `POST /api/analyze` with `sample=server_outage` -> HTTP 200, RED,
  rule `pass_rate_below_90`, pass rate 40.0%, 8 passed / 12 failed
- `POST /report` with `sample=real_bug` -> HTTP 200, page contains RED

## Tests passing

- **41 passed, 0 failed** (`python -m pytest tests/ -q`, 98.38 s),
  covering: parser, classifier, embeddings (TF-IDF + fallback +
  real MiniLM path), verify guard, release rules, SQL flaky query,
  tool-error handling, LLM-eval cache with a fake LLM, test
  selection (unit + API), API (`/`, `/report`, `/api/analyze`,
  `/health`, `/api/runs`, history), full no-key pipeline (RED
  real-bug sample, GREEN healthy sample).
- Original suite was 26 tests; 15 were added (2 changed-files API,
  3 embeddings, 4 LLM-eval, 6 selection).

## Not run

- Docker build/run: **not run**. What was tried, in order: Docker
  was not installed; `apt-get update` then `apt-get install -y
  docker.io` installed Docker 29.1.3, but the daemon would not start
  under systemd (`policy-rc.d denied execution of start`) and, when
  started manually with `--iptables=false --bridge=none` (this
  environment denies the NET_ADMIN capability the bridge driver
  needs: `iptables failed: Permission denied`), the daemon runs but
  cannot pull any image: every pull fails with `net/http: TLS
  handshake timeout` to `registry-1.docker.io`, reproducible across
  restarts and across proxy hostnames vs direct proxy IPs, while
  `curl` from the shell reaches the registry in 0.6 s (HTTP 401).
  The daemon cannot reach Docker Hub through this environment's
  egress proxy. `docker compose` is additionally unavailable
  (`docker-compose-plugin` is not locatable via apt here). The
  Dockerfile + docker-compose.yml are provided but untested here,
  and as written they install `requirements.txt` only, so a
  container built from them would run the TF-IDF fallback, not
  MiniLM (sentence-transformers is deliberately not in
  requirements.txt for image size).
- GitHub Actions CI run + push + README badge: **not run**.
  `gh auth status` reports not logged in and the GitHub connector
  reports `not_connected`, so no push to a `releaselens` repo was
  made, no CI run exists to confirm, and no badge was added (a badge
  pointing at a nonexistent repo would be dishonest).
- LLM methods (C), full agent (D), prompt v1 vs v2: **LLM not
  evaluated** (no key; see above).
- Load testing beyond the 4 samples: **not run**.


## Additional local Windows validation — 2 November 2026

Evidence: `docs/screenshots/07-windows-41-tests-passed.png` (captured from the user-run Windows 11 PowerShell session).

- Command: `.\.venv\Scripts\python.exe -m pytest -q -rs`
- After installing `sentence-transformers`: **41 passed, 0 failed, 0 skipped, 8 warnings, 53.00 s**.
- Prior to optional model-package installation: **40 passed, 1 skipped** (test embedding dependency unavailable).
- Warnings: ChromaDB `DeprecationWarning` for legacy embedding-function config; tests did not fail.
- Local FastAPI server: `/health` returned `{"status":"ok"}`, Swagger `/docs` displayed six routes, and GREEN / YELLOW / RED sample HTML report pages were visually inspected.
- This additional check does **not** constitute a rerun of GroupKFold model evaluations or live LLM evaluation. All synthetic-data and no-production-validation qualifications remain.
