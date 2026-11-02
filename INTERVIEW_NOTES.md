# INTERVIEW_NOTES.md — ReleaseLens

## 2-minute explanation (for a non-technical listener)

"Imagine your team has just run hundreds of automated tests, and someone
has to decide: can we release this version? The raw test output is
pages of errors nobody outside engineering can read. ReleaseLens takes
that test file and returns one page in plain English. At the top is a
big GREEN, YELLOW or RED verdict. Under it, the failures are grouped by
what they actually mean — a real bug in the product, a server problem,
a flaky test, a test-data mistake, or a dependency problem — each with
the exact log lines that prove it and one next step. It also tells you
which tests to re-check first, based on which ones failed recently —
and if you list the files you changed, it ranks the tests to run first
from those files, with a fixed, explainable rule and no AI involved.
Two things make it trustworthy. First, the verdict does not come from
an AI guessing: it comes from simple written rules, like 'RED if a
critical test has a real bug, or if fewer than 90% of tests pass.' The
AI only writes the explanations. Second, every explanation is checked:
if the AI invents a log line, that line is removed. And a person still
has to confirm before release — the page says so."

## 10 likely interview questions (with short, accurate answers based on THIS code)

### 1. Why doesn't the LLM decide the verdict?

Because a verdict must be explainable and repeatable. `app/rules.py`
implements `rules.yaml`: RED for a critical real bug or pass rate
below 90%, YELLOW for only flaky/environment failures or a 90–97% pass
rate, otherwise GREEN. The report shows which rule fired, so a manager
can follow the decision without trusting a model. The LLM only explains
failures, and its output is verified before it reaches the report.

### 2. How does the LangGraph flow work?

`app/agent/graph.py` builds a `StateGraph` with nodes parse -> classify
-> route -> investigate -> explain -> verify -> decide -> render. The
route is a conditional edge: if the lowest classifier confidence is
0.75 or above, the graph goes straight to explain; otherwise it goes to
investigate first. Investigate runs at most 3 LangChain tool calls
(search similar failures, test history, log excerpt); explain produces
a Pydantic `Explanation`; verify enforces the category enum and drops
non-verbatim evidence; decide applies the rules; render builds the
report and saves the run to SQLite.

### 3. What are the 5 categories, and how are failures classified?

Real product bug; Server/environment problem; Flaky or timing issue;
Test data or configuration problem; Dependency or build problem
(`app/categories.py`). The classifier is scikit-learn TF-IDF (word
n-grams plus character n-grams, added after an error analysis showed
Dependency was the weakest category) + LogisticRegression trained on
`data/failures_labeled.csv` (180 rows), and it returns a category plus
a confidence (the max class probability). Low confidence is what sends
a failure to investigation.

### 4. What is the vector memory for?

ChromaDB stores the 180 labeled past failures. For a new failure, the
explain node retrieves the top-3 similar ones and uses them as few-shot
examples (prompt v2) and as the "seen before" reference in the report.
The active embedding is all-MiniLM-L6-v2 (sentence-transformers,
model cached per process); TF-IDF is the fallback if MiniLM cannot
load. The collection name includes the embedding name, so the two
choices never collide. Honest caveat, measured here: on this synthetic
template data MiniLM retrieval scores lower than TF-IDF (kNN accuracy
0.672 vs 0.756), so the memory is context only — it never decides the
verdict.

### 5. How does the verify guard work?

`verify_explanations` in `app/agent/graph.py` does two things. If an
explanation's category is not one of the 5 enum values, it falls back
to the classifier's category. For each evidence line, it keeps the line
only if it appears verbatim in the original log; otherwise it drops the
line and counts it as unverified. The report shows the unverified
count and the grounding rate.

### 6. How do you find flaky tests and "tests to re-check first"?

With SQL, not Python loops. `app/flaky.py` queries the last 10 runs:
flaky tests are those with both passed and failed results in that
window (GROUP BY + HAVING). Re-check ranking orders tests with failures
by failure rate (failures / appearances), then failure count, then
name. The 20 seeded historical runs exist so these queries have
something to find. Separately, "tests to run first" (`app/selection.py`)
ranks the current run's tests from an optional changed-files list with
a fixed formula — 2 × name overlap + 5 × failure rate + 1 if flaky —
and no LLM is involved; it is a triage hint, not dependency analysis,
and new tests (no history) score on overlap alone.

### 7. How does it work with no LLM key?

The explain node checks for `LLM_API_KEY`. Without one, it uses a
deterministic template: the plain-English category label, the failure
message, the nearest past failure, the per-category next step, and up
to 3 lines copied from the log as evidence. The whole pipeline —
including the 4 sample reports — works with no key, and the tests
prove it (`test_full_pipeline_no_key_*`).

### 8. How did you evaluate it honestly?

`scripts/evaluate.py` uses GroupKFold by `template_family` (5 folds),
so near-identical templates cannot leak between train and test. The
shipped classifier (word + character TF-IDF + LogReg) gets 0.811
accuracy / 0.809 macro-F1, up from 0.761 / 0.756 word-only after an
error analysis on Dependency (the weakest category) motivated the
character features. MiniLM embeddings score lower on this data:
0.744 / 0.746 as a classifier and 0.672 / 0.675 for retrieval kNN
(vs 0.756 / 0.753 TF-IDF retrieval) — reported plainly, not hidden.
The release-decision check (10 handcrafted runs against `rules.yaml`)
agrees 10/10, grounding on the no-key path is 29/29, and 41 tests
pass. LLM methods are recorded as "LLM not evaluated" because no key
was available in this session — no number was invented for them. The
demo samples were also fixed for leakage: they used to copy training
messages, and now they are held-out paraphrases with a generation-time
check, so sample confidences (for example 0.65–0.72 for the flaky
sample) are honest unseen-input numbers.

### 9. What would you change for production?

Three things, in order: (1) replace synthetic data with real labeled
failures from the team's CI and re-measure — the current numbers only
describe the synthetic set, and the MiniLM-vs-TFIDF result in
particular needs re-measuring on real logs; (2) evaluate the LLM paths
with a key, including the v1 vs v2 prompt comparison on grounding rate
(the code and cache for that already exist); (3) move report
generation off the request thread and add auth before any multi-user
use. I would not claim the rules fit every team — the critical-test
list and thresholds in `rules.yaml` are meant to be edited. Docker and
CI are provided but unproven here: the daemon could not reach Docker
Hub through this VM's egress proxy, and GitHub was not connected.

### 10. What was the hardest bug, and what did it teach you?

The ChromaDB embedding interface: this Chroma version calls
`embed_query` for searches, not just `__call__`, so the first version
of the TF-IDF embedding function crashed every query. The fix was to
implement the current interface (`embed_documents` / `embed_query`)
on both embedding classes. The lesson: pin down the library's actual
interface from the traceback, not from memory of an older version —
and the same applies to the Jinja2 `g.items` clash (a dict method
shadowing a template key) fixed by renaming the key to `entries`. A
second one from the improvement run: this Chroma version shares
segment state across `EphemeralClient` instances, so a fixed
collection name collided on the second evaluation run in one process
— fixed with unique names per call. And the MiniLM model reloading on
every pipeline run (average report time 2.614 s) was fixed with a
per-process model cache (1.759 s).
