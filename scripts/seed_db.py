"""Seed 20 historical runs into SQLite (idempotent). Run:
    python scripts/seed_db.py
The app also seeds on first use if the runs table is empty.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import get_session, init_db  # noqa: E402
from app.seed import seed_historical_runs  # noqa: E402


def main() -> None:
    init_db()
    session = get_session()
    try:
        added = seed_historical_runs(session)
        print(f"Seeded {added} historical runs (0 means already seeded).")
    finally:
        session.close()


if __name__ == "__main__":
    main()
