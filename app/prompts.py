"""Load versioned prompt files from /prompts."""
from __future__ import annotations

from .config import prompts_dir


def load_prompt(version: str) -> str:
    path = prompts_dir() / f"{version}_explain.md"
    if not path.exists():
        path = prompts_dir() / "v2_explain.md"
    return path.read_text(encoding="utf-8")
