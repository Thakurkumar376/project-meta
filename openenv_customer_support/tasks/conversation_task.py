"""Task 3: Multi-turn customer support conversation scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ConversationTurn:
    """Represents one user turn in a conversation."""

    customer_message: str


@dataclass(frozen=True)
class ConversationScenario:
    """Conversation definition and grading hints."""

    scenario_id: str
    opening_context: str
    turns: List[ConversationTurn]
    expected_elements: List[str]


CONVERSATION_SCENARIOS: List[ConversationScenario] = [
    ConversationScenario(
        scenario_id="late_refund_followup",
        opening_context=(
            "A customer requested a refund 10 days ago and has not received it. "
            "They are upset and threatening to post negative reviews."
        ),
        turns=[
            ConversationTurn(
                customer_message=(
                    "This is my fourth message. Where is my refund and why is no one helping?"
                )
            ),
            ConversationTurn(
                customer_message=(
                    "I need the exact date and confirmation number, not another generic apology."
                )
            ),
            ConversationTurn(
                customer_message=(
                    "If this isn't resolved today, I'm filing a formal complaint."
                )
            ),
        ],
        expected_elements=[
            "apolog",
            "refund",
            "timeline",
            "confirmation",
            "escalat",
        ],
    ),
]
