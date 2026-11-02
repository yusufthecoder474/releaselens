"""Tests to run first: deterministic ranking, no LLM."""
from app.selection import (changed_file_tokens, overlap_count,
                           parse_changed_files, rank_tests, tokenize)


def test_tokenize_ignores_fillers():
    assert tokenize("src/test_login.py") == {"login"}
    assert tokenize("test_checkout_total") == {"checkout", "total"}


def test_overlap_count():
    tokens = changed_file_tokens(["src/checkout.py", "src/cart.py"])
    assert overlap_count("test_checkout_total", tokens) == 1
    assert overlap_count("test_login_page", tokens) == 0


def test_parse_changed_files():
    assert parse_changed_files("a.py, b.py\nc.py, ,") == ["a.py", "b.py", "c.py"]
    assert parse_changed_files(None) == []
    assert parse_changed_files("") == []


def test_rank_orders_by_overlap_then_failure_rate_then_name():
    tests = ["test_login_page", "test_checkout_total", "test_cart_add", "test_cart_remove"]
    ranked = rank_tests(tests, ["src/cart.py"], {}, set())
    names = [r["test_name"] for r in ranked]
    # Both cart tests overlap (score 2); name asc breaks the tie.
    assert names[:2] == ["test_cart_add", "test_cart_remove"]
    assert names[2:] == ["test_checkout_total", "test_login_page"]
    assert ranked[0]["overlap"] == 1


def test_rank_failure_rate_and_flaky_bonus():
    tests = ["test_a", "test_b", "test_c"]
    ranked = rank_tests(tests, [], {"test_a": 0.5, "test_b": 0.0, "test_c": 0.0},
                        {"test_c"})
    by_name = {r["test_name"]: r for r in ranked}
    assert by_name["test_a"]["score"] == 2.5
    assert by_name["test_c"]["score"] == 1.0
    assert by_name["test_b"]["score"] == 0.0
    assert [r["test_name"] for r in ranked] == ["test_a", "test_c", "test_b"]


def test_rank_limit():
    tests = [f"test_{i}" for i in range(15)]
    assert len(rank_tests(tests, [], {}, set())) == 10
