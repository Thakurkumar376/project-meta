"""Programmatic grader for Task 1 (Email Classification)."""

from __future__ import annotations

from openenv_customer_support.tasks.email_classification import VALID_EMAIL_CATEGORIES


def grade_email_classification(predicted: str, gold: str) -> float:
    """Return 1.0 for exact category match, else 0.0.

    Invalid categories are always graded as 0.0.
    """

    if not predicted:
        return 0.0

    normalized = predicted.strip().lower()
    if normalized not in VALID_EMAIL_CATEGORIES:
        return 0.0

    return 1.0 if normalized == gold.strip().lower() else 0.0
