"""Generate the 4 sample JUnit XML files in data/samples/.
healthy: 20 pass -> GREEN.
flaky_heavy: 30 tests, 27 pass, 3 flaky -> 90% -> YELLOW (only flaky).
server_outage: 20 tests, 8 pass, 12 environment -> 40% -> RED (pass < 90).
real_bug: 25 tests, 20 pass, 5 fail incl. 2 critical real bugs -> RED.
Critical tests are marked with a <property name="critical" value="true"/>.

LEAKAGE RULE (enforced at generation time): no failing sample may reuse
a test name or failure message from the classifier's training data
(data/failures_labeled.csv). The samples are a held-out demo, not
training rows in disguise. If a collision is found, generation fails
with the colliding entry printed.
All data is synthetic.
"""
from __future__ import annotations

import csv
import sys
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


def check_no_leakage(all_cases: list[dict]) -> None:
    """Fail generation if any failing sample reuses a training name or
    message. Passing tests are not classified, so they are exempt."""
    train_names: set[str] = set()
    train_msgs: set[str] = set()
    with open(ROOT / "data" / "failures_labeled.csv", newline="") as f:
        for row in csv.DictReader(f):
            train_names.add(row["test_name"])
            train_msgs.add(row["message"])
    collisions = []
    for c in all_cases:
        if c["failure"] is None:
            continue
        msg = c["failure"][0]
        if c["name"] in train_names:
            collisions.append(f"name reuse: {c['name']}")
        if msg in train_msgs:
            collisions.append(f"message reuse: {c['name']}: {msg}")
    if collisions:
        print("LEAKAGE CHECK FAILED:", file=sys.stderr)
        for col in collisions:
            print(f"  - {col}", file=sys.stderr)
        raise SystemExit(1)
    print("Leakage check passed: no failing sample reuses a training name or message.")


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
    ("Element detached after carousel refresh, passed on rerun",
     "StaleElementReferenceException: carousel slide element detached after refresh\n  Passed on rerun."),
    ("Tooltip did not appear within 8s wait, passed on rerun",
     "TimeoutException: tooltip not visible after 8 seconds\n  Passed on rerun."),
    ("Assertion ran while spinner still visible, passes with longer wait",
     "AssertionError: expected spinner hidden but still visible\n  Passes when wait is increased; timing dependent."),
]

ENV_FAILS = [
    ("Queue broker unreachable at queue.internal:5672",
     "pika.exceptions.AMQPConnectionError: failed to connect to queue.internal:5672: Connection refused"),
    ("503 from auth service during deployment",
     "503 Service Unavailable\nAuth service is being redeployed"),
    ("Gateway timed out talking to billing service (504)",
     "504 Gateway Time-out\nnginx: upstream billing service timed out"),
    ("DNS lookup failed for cache.internal",
     "socket.gaierror: [Errno -2] Name or service not known for cache.internal"),
    ("Connection refused to search.internal:9200",
     "requests.exceptions.ConnectionError: HTTPConnectionPool(host='search.internal', port=9200): Connection refused"),
    ("500 from email delivery service",
     "500 Internal Server Error\nEmail service returned 500"),
    ("Read timed out calling partner API after 15s",
     "requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='partner.example.com'): Read timed out. (read timeout=15)"),
    ("502 from CDN edge node",
     "502 Bad Gateway\nCDN edge returned an invalid response"),
    ("Session store refused connection at session.internal:11211",
     "pymemcache.exceptions.MemcacheError: Error connecting to session.internal:11211. Connection refused."),
    ("503 from SMS service",
     "503 Service Unavailable: SMS service is down for maintenance"),
    ("Report query canceled after 45s by statement timeout",
     "psycopg2.errors.QueryCanceled: report query canceled after 45 seconds by statement timeout"),
    ("Temporary DNS failure for queue.internal",
     "socket.gaierror: could not resolve host name \"queue.internal\": Temporary failure in name resolution"),
]

REAL_BUG_FAILS = [
    ("Valid account sign-in rejected with 403",
     "AssertionError: expected <200> but was <403>\nSign-in failed for valid account bob@example.com"),
    ("Basket amount 74.50 expected, 64.50 shown",
     "AssertionError: expected <74.50> but was <64.50>"),
    ("Voucher should reduce total by 15.0, reduction was 0.0",
     "assert basket.voucher_reduction == 15.0, got 0.0"),
    ("Dialog appeared only on second try after wait",
     "TimeoutException: timed out waiting for dialog window\n  Retry succeeded."),
    ("Seed file customers.json missing from fixtures",
     "FileNotFoundError: [Errno 2] No such file or directory: 'fixtures/customers.json'"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    healthy = [testcase(n) for n in PASS_NAMES[:20]]
    flaky = [testcase(n) for n in PASS_NAMES[:27]]
    flaky_names = ["test_carousel_slide", "test_tooltip_hover", "test_spinner_wait"]
    for name, f in zip(flaky_names, FLAKY_FAILS):
        flaky.append(testcase(name, failure=f))
    outage = [testcase(n) for n in PASS_NAMES[:8]]
    for i, f in enumerate(ENV_FAILS):
        outage.append(testcase(f"test_outage_{i+1}", failure=f))
    bug = [testcase(n) for n in PASS_NAMES[:20]]
    bug_names = ["test_signin_valid_account", "test_basket_amount",
                 "test_voucher_applies", "test_dialog_open", "test_seed_customers"]
    for i, (name, f) in enumerate(zip(bug_names, REAL_BUG_FAILS)):
        bug.append(testcase(name, failure=f, critical=(i < 2)))
    all_cases = healthy + flaky + outage + bug
    check_no_leakage(all_cases)
    build("healthy", healthy).write(OUT / "healthy.xml", encoding="utf-8", xml_declaration=True)
    build("flaky_heavy", flaky).write(OUT / "flaky_heavy.xml", encoding="utf-8", xml_declaration=True)
    build("server_outage", outage).write(OUT / "server_outage.xml", encoding="utf-8", xml_declaration=True)
    build("real_bug", bug).write(OUT / "real_bug.xml", encoding="utf-8", xml_declaration=True)
    for p in sorted(OUT.glob("*.xml")):
        print(p.name)


if __name__ == "__main__":
    main()
