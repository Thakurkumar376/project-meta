"""Programmatic grader for Task 2 (Response Generation)."""

from __future__ import annotations

from typing import List


def grade_response_quality(response: str, required_keywords: List[str]) -> float:
    """Return a normalized score based on professionalism and relevance.

    Score components:
    - keyword coverage (60%)
    - minimum professional length (20%)
    - empathy marker presence (20%)
    """

    if not response or not response.strip():
        return 0.0

    text = response.lower().strip()

    keyword_hits = sum(1 for kw in required_keywords if kw.lower() in text)
    keyword_score = keyword_hits / max(len(required_keywords), 1)

    length_score = 1.0 if len(text.split()) >= 35 else 0.4

    empathy_markers = ["sorry", "apolog", "understand", "frustrat", "thank you"]
    empathy_score = 1.0 if any(marker in text for marker in empathy_markers) else 0.2

    final_score = (0.6 * keyword_score) + (0.2 * length_score) + (0.2 * empathy_score)
    return max(0.0, min(1.0, round(final_score, 4)))
