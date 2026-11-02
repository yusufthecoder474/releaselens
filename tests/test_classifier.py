from app.classifier import load_labeled_rows, predict_category, train_classifier


def test_labeled_data_shape():
    rows = load_labeled_rows()
    assert len(rows) == 180
    assert {r["category"] for r in rows} == {"real_bug", "environment", "flaky", "test_data", "dependency"}
    assert len({r["template_family"] for r in rows}) == 60


def test_classifier_trains_and_predicts_known_kinds():
    train_classifier()  # must not raise
    cat, conf = predict_category("test_db_connect",
                                 "Connection refused: could not connect to database at db.internal:5432",
                                 "psycopg2.OperationalError: Connection refused")
    assert cat == "environment"
    assert 0.0 < conf <= 1.0
    cat2, _ = predict_category("test_user_profile_load",
                               "AttributeError: 'NoneType' object has no attribute 'email'",
                               "Traceback: AttributeError: 'NoneType' object has no attribute 'email'")
    assert cat2 == "real_bug"
