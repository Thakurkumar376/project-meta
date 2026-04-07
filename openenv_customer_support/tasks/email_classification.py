"""Task 1: Email classification samples and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


VALID_EMAIL_CATEGORIES = [
    "support",
    "refund",
    "complaint",
    "order status",
    "spam",
]


@dataclass(frozen=True)
class EmailExample:
    """Single classification sample."""

    prompt: str
    gold_category: str


EMAIL_CLASSIFICATION_DATASET: List[EmailExample] = [
    EmailExample(
        prompt=(
            "Hello team, my internet router has been disconnecting every hour. "
            "Can you help me troubleshoot this issue?"
        ),
        gold_category="support",
    ),
    EmailExample(
        prompt=(
            "I was charged for my order yesterday but I canceled it before shipping. "
            "Please issue a full refund immediately."
        ),
        gold_category="refund",
    ),
    EmailExample(
        prompt=(
            "I have contacted your company 3 times and still no replacement. "
            "This is unacceptable and I am very frustrated."
        ),
        gold_category="complaint",
    ),
    EmailExample(
        prompt=(
            "Can you tell me where package #A12345 is? It was supposed to arrive "
            "on Monday and tracking has not updated."
        ),
        gold_category="order status",
    ),
    EmailExample(
        prompt=(
            "CONGRATULATIONS! You won a free vacation. Click this suspicious link "
            "to claim your prize now."
        ),
        gold_category="spam",
    ),
]
