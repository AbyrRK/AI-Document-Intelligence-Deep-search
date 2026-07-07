"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Pytest shared fixtures, evaluation cases, and OCR metrics reporting.
"""

import pytest
import json
import os

# ═══════════════════════════════════════════════════════════════════════════
# Load evaluation datasets (both optional — won't crash if missing)
# ═══════════════════════════════════════════════════════════════════════════

# ── QA / RAG evaluation cases ─────────────────────────────────────────────
_QA_PATH = os.path.join(os.path.dirname(__file__), "eval_cases.json")
try:
    with open(_QA_PATH, "r", encoding="utf-8") as f:
        EVAL_CASES_QA = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    EVAL_CASES_QA = []

# ── OCR evaluation cases (images + ground truth) ──────────────────────────
_OCR_PATH = os.path.join(os.path.dirname(__file__), "eval_cases_ocr.json")
try:
    with open(_OCR_PATH, "r", encoding="utf-8") as f:
        EVAL_CASES_OCR = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    EVAL_CASES_OCR = []


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def eval_cases():
    """QA/RAG evaluation cases."""
    return EVAL_CASES_QA


@pytest.fixture(scope="session")
def eval_cases_ocr():
    """OCR evaluation cases (images + ground truth)."""
    return EVAL_CASES_OCR


@pytest.fixture(scope="session")
def session_id():
    """Session ID used for retrieval. Must be set via EVAL_SESSION_ID env var."""
    sid = os.environ.get("EVAL_SESSION_ID")
    if not sid:
        pytest.fail(
            "EVAL_SESSION_ID is not set. "
            "Run: $env:EVAL_SESSION_ID = '<your-session-id>'"
        )
    return sid


# ═══════════════════════════════════════════════════════════════════════════
# OCR METRICS REPORTING
# ═══════════════════════════════════════════════════════════════════════════

_metrics_registry = []


def record_metric(name: str, value: float, unit: str = ""):
    """Call this from any test to register a metric for the final report."""
    _metrics_registry.append({"name": name, "value": value, "unit": unit})


@pytest.fixture
def metrics():
    """Fixture for tests to record metric values."""
    return _metrics_registry


def pytest_addoption(parser):
    parser.addoption(
        "--report-metrics",
        action="store_true",
        default=False,
        help="Print OCR metrics summary after test run",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--report-metrics") or not _metrics_registry:
        return

    terminalreporter.section("OCR METRICS SUMMARY", sep="-")
    terminalreporter.write_line(f"{'Metric':<45} {'Value':>12} {'Unit':>8}")
    terminalreporter.write_line("-" * 67)

    for m in _metrics_registry:
        terminalreporter.write_line(
            f"{m['name']:<45} {m['value']:>12.3f} {m['unit']:>8}"
        )