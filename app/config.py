"""Paths and environment settings. Everything is read from the environment
at call time (not import time) so tests can point the app at temp dirs.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def root() -> Path:
    return ROOT


def database_path() -> Path:
    return Path(os.environ.get("RELEASELENS_DB", str(ROOT / "releaselens.db")))


def database_url() -> str:
    return f"sqlite:///{database_path()}"


def chroma_path() -> Path:
    return Path(os.environ.get("RELEASELENS_CHROMA", str(ROOT / "chroma_db")))


def labeled_csv() -> Path:
    return ROOT / "data" / "failures_labeled.csv"


def samples_dir() -> Path:
    return ROOT / "data" / "samples"


def rules_path() -> Path:
    return ROOT / "rules.yaml"


def prompts_dir() -> Path:
    return ROOT / "prompts"


def llm_api_key() -> str | None:
    return os.environ.get("LLM_API_KEY") or None


def llm_base_url() -> str:
    return os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")


def llm_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


def prompt_version() -> str:
    return os.environ.get("PROMPT_VERSION", "v2")
