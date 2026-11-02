from app.rules import decide, is_critical, load_rules


def test_critical_real_bug_is_red():
    verdict, rule = decide(80.0, [{"category": "real_bug", "critical": True}])
    assert (verdict, rule) == ("RED", "critical_real_bug")


def test_low_pass_rate_is_red():
    verdict, rule = decide(85.0, [{"category": "flaky", "critical": False}])
    assert (verdict, rule) == ("RED", "pass_rate_below_90")


def test_only_flaky_is_yellow():
    verdict, rule = decide(99.0, [{"category": "flaky", "critical": False},
                                 {"category": "environment", "critical": False}])
    assert (verdict, rule) == ("YELLOW", "only_flaky_or_environment")


def test_pass_rate_band_is_yellow():
    verdict, rule = decide(95.0, [{"category": "real_bug", "critical": False}])
    assert (verdict, rule) == ("YELLOW", "pass_rate_90_97")


def test_clean_run_is_green():
    assert decide(100.0, []) == ("GREEN", "all_clear")


def test_critical_keywords():
    rules = load_rules()
    assert is_critical("test_login_valid_user", False, rules) is True
    assert is_critical("test_search_basic", False, rules) is False
    assert is_critical("test_search_basic", True, rules) is True
