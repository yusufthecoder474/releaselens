"""Measure report-generation time and grounding rate (no-key mode).

Uses temp DB/Chroma paths so it never touches the real database.
Warms up once (classifier training + Chroma seeding happen on first
use), then times one pipeline run per sample and aggregates grounding:
verified evidence lines / (verified + unverified) across all reports.
Prints MEASURE_... lines for RESULTS.md.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="releaselens-measure-")
os.environ["RELEASELENS_DB"] = os.path.join(_tmp, "measure.db")
os.environ["RELEASELENS_CHROMA"] = os.path.join(_tmp, "chroma")
os.environ.pop("LLM_API_KEY", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.graph import run_pipeline  # noqa: E402
from app.config import samples_dir  # noqa: E402


def main() -> None:
    # Warm-up (untimed): trains the classifier, seeds Chroma + history.
    warm_xml = (samples_dir() / "healthy.xml").read_text(encoding="utf-8")
    run_pipeline(warm_xml, source="measure-warmup")
    total_verified = 0
    total_lines = 0
    times: list[float] = []
    for name in ("healthy", "flaky_heavy", "server_outage", "real_bug"):
        xml_text = (samples_dir() / f"{name}.xml").read_text(encoding="utf-8")
        start = time.perf_counter()
        report = run_pipeline(xml_text, source=f"measure:{name}")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        verified_lines = sum(len(item["evidence_lines"])
                             for g in report["groups"] for item in g["entries"])
        total_verified += verified_lines
        total_lines += verified_lines + report["unverified_evidence_count"]
        print(f"MEASURE {name} verdict={report['verdict']} "
              f"time_s={elapsed:.3f} grounding_rate={report['grounding_rate']:.4f} "
              f"verified_lines={verified_lines} unverified={report['unverified_evidence_count']}")
    grounding = (total_verified / total_lines) if total_lines else 1.0
    print(f"MEASURE_AVG_TIME_S={sum(times) / len(times):.3f}")
    print(f"MEASURE_GROUNDING_RATE={grounding:.4f} "
          f"({total_verified}/{total_lines} evidence lines verified)")
    print(f"MEASURE_TIMES_S={[round(t, 3) for t in times]}")


if __name__ == "__main__":
    main()
