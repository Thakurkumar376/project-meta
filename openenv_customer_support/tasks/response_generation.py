"""Task 2: Customer support response generation prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ResponseExample:
    """Single customer complaint example with target support elements."""

    complaint: str
    required_keywords: List[str]


RESPONSE_GENERATION_DATASET: List[ResponseExample] = [
    ResponseExample(
        complaint=(
            "I ordered a coffee machine two weeks ago and it still has not arrived. "
            "Tracking says delayed with no estimate."
        ),
        required_keywords=["sorry", "tracking", "investigate", "update"],
    ),
    ResponseExample(
        complaint=(
            "The headphones I received are broken out of the box. The left side "
            "doesn't work and I need a replacement fast."
        ),
        required_keywords=["apologize", "replacement", "return", "assist"],
    ),
    ResponseExample(
        complaint=(
            "Your app subscription renewed without warning and I want the charge "
            "reversed right away."
        ),
        required_keywords=["refund", "billing", "confirm", "support"],
    ),
]
