"""Pydantic structured output for the explain node (and the JSON API)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .categories import Category


class Explanation(BaseModel):
    """One failure, explained. The verify guard enforces the enum and
    that every evidence line appears verbatim in the original log."""

    category: Category = Field(description="One of the 5 fixed categories")
    plain_english_reason: str = Field(description="One or two plain sentences")
    evidence_lines: list[str] = Field(default_factory=list)
    suggested_next_step: str = Field(default="")
