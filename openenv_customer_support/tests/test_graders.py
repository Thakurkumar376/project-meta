from openenv_customer_support.graders.conversation_grader import grade_conversation_quality
from openenv_customer_support.graders.email_grader import grade_email_classification
from openenv_customer_support.graders.response_grader import grade_response_quality


def test_email_grader_exact_match():
    assert grade_email_classification("refund", "refund") == 1.0
    assert grade_email_classification("support", "refund") == 0.0
    assert grade_email_classification("unknown", "refund") == 0.0


def test_response_grader_range_and_ordering():
    good = "Sorry for the frustration. I will investigate tracking and provide an update."
    poor = "ok"
    required = ["sorry", "tracking", "investigate", "update"]

    good_score = grade_response_quality(good, required)
    poor_score = grade_response_quality(poor, required)

    assert 0.0 <= good_score <= 1.0
    assert 0.0 <= poor_score <= 1.0
    assert good_score > poor_score


def test_conversation_grader_range():
    responses = [
        "I apologize for the delay. I will escalate this refund now and share a timeline.",
        "Here is your confirmation number and expected processing timeline.",
        "I have escalated the case and will update you again today.",
    ]
    score = grade_conversation_quality(responses, ["apolog", "refund", "timeline", "confirmation", "escalat"])
    assert 0.0 <= score <= 1.0
