from app.agent.graph import verify_explanations

FAILURES = {
    "test_a": {"log": "AssertionError: expected 1 but got 2\nat T.test_a", "classifier_category": "real_bug"},
}


def test_verify_drops_fake_evidence_and_marks_unverified():
    explanations = [{
        "test_name": "test_a", "category": "real_bug",
        "plain_english_reason": "A real bug.",
        "evidence_lines": ["AssertionError: expected 1 but got 2",
                           "Totally invented log line that never happened"],
        "suggested_next_step": "Fix it.",
    }]
    verified, invalid, unverified, grounding = verify_explanations(explanations, FAILURES)
    assert verified[0]["evidence_lines"] == ["AssertionError: expected 1 but got 2"]
    assert unverified == 1
    assert grounding == 0.5
    assert invalid == 0


def test_verify_falls_back_on_invalid_category():
    explanations = [{
        "test_name": "test_a", "category": "not_a_category",
        "plain_english_reason": "x", "evidence_lines": [],
        "suggested_next_step": "y",
    }]
    verified, invalid, unverified, grounding = verify_explanations(explanations, FAILURES)
    assert verified[0]["category"] == "real_bug"
    assert invalid == 1
    assert grounding == 1.0  # no evidence lines at all -> 1.0 by definition
