"""Programmatic grader for Task 3 (Multi-turn Conversation)."""

from __future__ import annotations

from typing import Iterable, List


def grade_conversation_quality(responses: Iterable[str], expected_elements: List[str]) -> float:
    """Grade a full conversation transcript with a score in [0.0, 1.0].

    Score components:
    - expected element coverage across all turns (70%)
    - non-empty completion across turns (20%)
    - anti-loop penalty / variation bonus (10%)
    """

    response_list = [r.strip() for r in responses]
    if not response_list:
        return 0.0

    joined = " ".join(response_list).lower()

    coverage_hits = sum(1 for el in expected_elements if el.lower() in joined)
    coverage_score = coverage_hits / max(len(expected_elements), 1)

    completion_score = sum(1 for r in response_list if len(r.split()) >= 10) / len(response_list)

    unique_ratio = len(set(response_list)) / len(response_list)
    loop_score = 1.0 if unique_ratio >= 0.8 else 0.2

    final_score = (0.7 * coverage_score) + (0.2 * completion_score) + (0.1 * loop_score)
    return max(0.0, min(1.0, round(final_score, 4)))
