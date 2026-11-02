from app.parser import parse_junit_xml

XML = """<testsuites>
  <testsuite name="suite" tests="4" failures="2" errors="0">
    <testcase classname="T" name="test_ok" time="0.1"/>
    <testcase classname="T" name="test_bad" time="0.2">
      <failure message="AssertionError: expected 1 but got 2">AssertionError: expected 1 but got 2
at T.test_bad</failure>
    </testcase>
    <testcase classname="T" name="test_err" time="0.1">
      <error message="NullPointerException">java.lang.NullPointerException</error>
    </testcase>
    <testcase classname="T" name="test_skip" time="0.0">
      <skipped/>
    </testcase>
    <testcase classname="T" name="test_crit" time="0.1">
      <properties><property name="critical" value="true"/></properties>
      <failure message="boom">boom log</failure>
    </testcase>
  </testsuite>
</testsuites>"""


def test_parser_counts_and_fields():
    parsed = parse_junit_xml(XML)
    assert parsed.total == 5
    assert parsed.passed == 1
    assert parsed.failed == 3
    assert parsed.skipped == 1
    by_name = {t.name: t for t in parsed.tests}
    assert by_name["test_bad"].message.startswith("AssertionError")
    assert "NullPointerException" in by_name["test_err"].log
    assert by_name["test_crit"].critical_property is True
    assert by_name["test_ok"].critical_property is False


def test_parser_pass_rate_excludes_skipped():
    parsed = parse_junit_xml(XML)
    # 1 passed / (1 passed + 3 failed) = 25%
    assert parsed.pass_rate == 25.0


def test_parser_rejects_bad_xml():
    import pytest
    with pytest.raises(ValueError):
        parse_junit_xml("<not-xml")
