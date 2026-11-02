"""Generate the 4 sample JUnit XML files in data/samples/.
healthy: 20 pass -> GREEN.
flaky_heavy: 30 tests, 27 pass, 3 flaky -> 90% -> YELLOW (only flaky).
server_outage: 20 tests, 8 pass, 12 environment -> 40% -> RED (pass < 90).
real_bug: 25 tests, 20 pass, 5 fail incl. 2 critical real bugs -> RED.
Critical tests are marked with a <property name="critical" value="true"/>.
All data is synthetic.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples"


def testcase(name: str, failure: tuple[str, str] | None = None, critical: bool = False) -> dict:
    return {"name": name, "failure": failure, "critical": critical}


def build(suite_name: str, cases: list[dict]) -> ET.ElementTree:
    n_fail = sum(1 for c in cases if c["failure"] is not None)
    suite = ET.Element("testsuite", {
        "name": suite_name, "tests": str(len(cases)),
        "failures": str(n_fail), "errors": "0", "time": "12.5",
    })
    for c in cases:
        tc = ET.SubElement(suite, "testcase", {
            "classname": "com.example.Tests", "name": c["name"], "time": "0.5",
        })
        if c["critical"]:
            props = ET.SubElement(tc, "properties")
            ET.SubElement(props, "property", {"name": "critical", "value": "true"})
        if c["failure"] is not None:
            msg, log = c["failure"]
            fail = ET.SubElement(tc, "failure", {"message": msg})
            fail.text = log
    tree = ET.ElementTree(ET.Element("testsuites"))
    tree.getroot().append(suite)
    ET.indent(tree, space="  ")
    return tree


PASS_NAMES = [
    "test_login_page_loads", "test_search_basic", "test_cart_add",
    "test_profile_view", "test_homepage_title", "test_logout",
    "test_product_list", "test_order_history", "test_settings_save",
    "test_help_page", "test_footer_links", "test_pagination",
    "test_sort_results", "test_filter_category", "test_wishlist_add",
    "test_newsletter_signup", "test_contact_form", "test_about_page",
    "test_privacy_page", "test_terms_page", "test_sitemap",
    "test_language_switch", "test_currency_switch", "test_avatar_view",
    "test_notification_list", "test_dashboard_widgets", "test_export_csv",
    "test_import_preview", "test_search_filters", "test_cart_remove",
]

FLAKY_FAILS = [
    ("StaleElementReferenceException: element is not attached to the page",
     "selenium.common.exceptions.StaleElementReferenceException: Message: stale element reference: element is not attached to the page document\n  Passed on retry."),
    ("Timeout: element not visible after 10s wait, passed on retry",
     "selenium.common.exceptions.TimeoutException: Message: element not visible after 10 seconds\n  Passed on retry."),
    ("Timing sensitive: assertion ran before animation finished",
     "AssertionError: expected animation complete but was still running\n  Passes when sleep is increased; timing dependent."),
]

ENV_FAILS = [
    ("Connection refused: could not connect to database at db.internal:5432",
     "psycopg2.OperationalError: connection to server at \"db.internal\" (10.0.2.15), port 5432 failed: Connection refused"),
    ("503 Service Unavailable: backend service is down",
     "503 Service Unavailable\nService temporarily unavailable, maintenance window active"),
    ("Gateway Timeout: server did not respond within 30s (504)",
     "504 Gateway Time-out\nnginx: upstream timed out (110: Connection timed out)"),
    ("DNS resolution failed: Name or service not known for api.internal",
     "socket.gaierror: [Errno -2] Name or service not known"),
    ("Connection refused to api.internal:8080",
     "requests.exceptions.ConnectionError: HTTPConnectionPool(host='api.internal', port=8080): Max retries exceeded: Connection refused"),
    ("500 Internal Server Error from payment gateway",
     "500 Internal Server Error\nUpstream payment service returned 500"),
    ("ReadTimeout: HTTPSConnectionPool read timed out after 10s",
     "requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='partner.example.com', port=443): Read timed out. (read timeout=10)"),
    ("502 Bad Gateway: search service returned invalid response",
     "502 Bad Gateway\nnginx: upstream sent invalid header while reading response"),
    ("Connection refused to redis cache at cache.internal:6379",
     "redis.exceptions.ConnectionError: Error connecting to cache.internal:6379. Connection refused."),
    ("503 Service Unavailable from notification service",
     "503 Service Unavailable: notification service is down for maintenance"),
    ("Query timeout: database query exceeded 30s limit",
     "psycopg2.errors.QueryCanceled: canceling statement due to statement timeout"),
    ("Temporary failure in name resolution for db.internal",
     "psycopg2.OperationalError: could not translate host name \"db.internal\" to address: Temporary failure in name resolution"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    healthy = [testcase(n) for n in PASS_NAMES[:20]]
    build("healthy", healthy).write(OUT / "healthy.xml", encoding="utf-8", xml_declaration=True)
    flaky = [testcase(n) for n in PASS_NAMES[:27]]
    for i, f in enumerate(FLAKY_FAILS):
        flaky.append(testcase(f"test_flaky_{i+1}", failure=f))
    build("flaky_heavy", flaky).write(OUT / "flaky_heavy.xml", encoding="utf-8", xml_declaration=True)
    outage = [testcase(n) for n in PASS_NAMES[:8]]
    for i, f in enumerate(ENV_FAILS):
        outage.append(testcase(f"test_env_{i+1}", failure=f))
    build("server_outage", outage).write(OUT / "server_outage.xml", encoding="utf-8", xml_declaration=True)
    bug = [testcase(n) for n in PASS_NAMES[:20]]
    bug.append(testcase("test_login_valid_user",
                        failure=("AssertionError: expected 200 but got 401 for valid credentials",
                                 "AssertionError: expected <200> but was <401>\nLogin failed for valid user alice@example.com"),
                        critical=True))
    bug.append(testcase("test_checkout_total",
                        failure=("AssertionError: expected 59.99 but got 49.99",
                                 "AssertionError: expected <59.99> but was <49.99>"),
                        critical=True))
    bug.append(testcase("test_discount_applies",
                        failure=("AssertionError: expected 10.0 but got 0.0",
                                 "assert cart.discount == 10.0, got 0.0")))
    bug.append(testcase("test_modal_open",
                        failure=("Timeout waiting for modal, succeeded on second attempt",
                                 "TimeoutException: timed out waiting for modal dialog\n  Retry succeeded.")))
    bug.append(testcase("test_import_users",
                        failure=("FileNotFoundError: fixture users.csv not found",
                                 "FileNotFoundError: [Errno 2] No such file or directory: 'fixtures/users.csv'")))
    build("real_bug", bug).write(OUT / "real_bug.xml", encoding="utf-8", xml_declaration=True)
    for p in sorted(OUT.glob("*.xml")):
        print(p.name)


if __name__ == "__main__":
    main()
