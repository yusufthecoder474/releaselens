"""Parse JUnit XML into plain test results.

Handles a top-level <testsuites> or a single <testsuite>. A testcase is
failed when it has a <failure> or <error> child, skipped when it has a
<skipped> child, otherwise passed. A testcase is tagged critical when it
carries <property name="critical" value="true"/> (rules.yaml keywords are
applied later, in the decide step).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class TestCase:
    name: str
    classname: str = ""
    status: str = "passed"  # passed | failed | skipped
    message: str = ""
    log: str = ""
    critical_property: bool = False


@dataclass
class ParsedRun:
    tests: list[TestCase] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.tests)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.status == "passed")

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tests if t.status == "failed")

    @property
    def skipped(self) -> int:
        return sum(1 for t in self.tests if t.status == "skipped")

    @property
    def failures(self) -> list[TestCase]:
        return [t for t in self.tests if t.status == "failed"]

    @property
    def pass_rate(self) -> float:
        """Passed / (passed + failed) * 100. Skipped tests are excluded."""
        denom = self.passed + self.failed
        if denom == 0:
            return 100.0
        return 100.0 * self.passed / denom


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in ("true", "1", "yes")


def parse_junit_xml(xml_text: str) -> ParsedRun:
    """Parse JUnit XML text. Raises ValueError on malformed XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Not valid JUnit XML: {exc}") from exc

    if root.tag == "testsuites":
        suites = list(root.iter("testsuite"))
    elif root.tag == "testsuite":
        suites = [root]
    else:
        raise ValueError(f"Unexpected root element <{root.tag}>; expected <testsuites> or <testsuite>")

    tests: list[TestCase] = []
    for suite in suites:
        for tc in suite.findall("testcase"):
            name = tc.get("name", "unnamed_test")
            classname = tc.get("classname", "")
            critical = False
            props = tc.find("properties")
            if props is not None:
                for prop in props.findall("property"):
                    if prop.get("name") == "critical" and _is_true(prop.get("value")):
                        critical = True
            failure_el = tc.find("failure")
            error_el = tc.find("error")
            skipped_el = tc.find("skipped")
            if failure_el is not None or error_el is not None:
                el = failure_el if failure_el is not None else error_el
                assert el is not None
                message = el.get("message") or ""
                log = (el.text or "").strip()
                if not message and log:
                    message = log.splitlines()[0].strip()
                tests.append(TestCase(name=name, classname=classname, status="failed",
                                      message=message, log=log, critical_property=critical))
            elif skipped_el is not None:
                tests.append(TestCase(name=name, classname=classname, status="skipped",
                                      critical_property=critical))
            else:
                tests.append(TestCase(name=name, classname=classname, status="passed",
                                      critical_property=critical))
    return ParsedRun(tests=tests)
