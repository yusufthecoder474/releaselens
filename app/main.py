"""FastAPI app: one upload page, one report page, plus a JSON API.

This is NOT a chatbot: there is no chat UI anywhere. Endpoints:
  GET  /                        upload page
  POST /report                  HTML report (file upload or sample)
  POST /api/analyze             JSON report (file upload or sample)
  GET  /api/runs                recent runs
  GET  /api/tests/{name}/history  history for one test
  GET  /health                  health check
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .agent.graph import run_pipeline
from .config import samples_dir
from .db import Run, get_session, init_db
from .flaky import failure_rates, flaky_tests, test_history
from .parser import parse_junit_xml
from .seed import seed_historical_runs
from .selection import parse_changed_files, rank_tests

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="ReleaseLens")

SAMPLES = ["healthy", "flaky_heavy", "server_outage", "real_bug"]


def _sample_xml(name: str) -> str:
    path = samples_dir() / f"{name}.xml"
    if not path.exists():
        raise ValueError(f"Unknown sample '{name}'. Choose one of: {', '.join(SAMPLES)}")
    return path.read_text(encoding="utf-8")


def _analyze(file: UploadFile | None, sample: str | None,
             changed_files_raw: str | None = None) -> dict:
    """Shared input handling for /report and /api/analyze.

    When changed file paths are supplied, a deterministic
    "tests to run first" ranking (selection.py, no LLM) is attached as
    report["tests_to_run_first"]; otherwise that list is empty.
    """
    if sample:
        xml_text = _sample_xml(sample)
        source = f"sample:{sample}"
    elif file is not None:
        raw = file.file.read()
        xml_text = raw.decode("utf-8", errors="replace")
        source = f"upload:{file.filename or 'junit.xml'}"
    else:
        raise ValueError("Upload a JUnit XML file or choose a sample.")
    session = get_session()
    try:
        init_db()
        seed_historical_runs(session)
        report = run_pipeline(xml_text, source=source, session=session)
        changed_files = parse_changed_files(changed_files_raw)
        report["changed_files"] = changed_files
        if changed_files:
            test_names = [t.name for t in parse_junit_xml(xml_text).tests]
            rates = failure_rates(session, test_names)
            report["tests_to_run_first"] = rank_tests(
                test_names, changed_files, rates, set(flaky_tests(session)))
        else:
            report["tests_to_run_first"] = []
        return report
    finally:
        session.close()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"samples": SAMPLES})


@app.post("/report", response_class=HTMLResponse)
async def report_page(request: Request, file: UploadFile | None = File(default=None),
                      sample: str | None = Form(default=None),
                      changed_files: str | None = Form(default=None)):
    try:
        report = _analyze(file, sample, changed_files)
    except ValueError as exc:
        return templates.TemplateResponse(request, "index.html",
                                          {"samples": SAMPLES, "error": str(exc)},
                                          status_code=400)
    return templates.TemplateResponse(request, "report.html", {"report": report})


@app.post("/api/analyze")
async def analyze_api(file: UploadFile | None = File(default=None),
                      sample: str | None = Form(default=None),
                      changed_files: str | None = Form(default=None)):
    try:
        report = _analyze(file, sample, changed_files)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return report


@app.get("/api/runs")
def list_runs():
    session = get_session()
    try:
        init_db()
        rows = session.query(Run).order_by(Run.id.desc()).limit(50).all()
        return [{"id": r.id, "created_at": r.created_at.isoformat() if r.created_at else "",
                 "source": r.source, "total": r.total, "passed": r.passed,
                 "failed": r.failed, "verdict": r.verdict, "pass_rate": r.pass_rate}
                for r in rows]
    finally:
        session.close()


@app.get("/api/tests/{name}/history")
def test_history_api(name: str):
    session = get_session()
    try:
        init_db()
        return {"test_name": name, "history": test_history(session, name)}
    finally:
        session.close()


@app.get("/health")
def health():
    return {"status": "ok"}
