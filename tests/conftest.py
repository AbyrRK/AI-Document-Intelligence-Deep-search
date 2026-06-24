"""
conftest.py
─────────────────────────────────────────────────────────────────────────────
Pytest shared fixtures and configuration.
"""

import pytest
import json
import os

# ── Load test cases once ──────────────────────────────────────────────────

_CASES_PATH = os.path.join(os.path.dirname(__file__), "eval_cases.json")

with open(_CASES_PATH, "r", encoding="utf-8") as f:
    EVAL_CASES = json.load(f)


@pytest.fixture(scope="session")
def eval_cases():
    """Return the full list of evaluation test cases."""
    return EVAL_CASES


@pytest.fixture(scope="session")
def session_id():
    """Session ID used for retrieval. Override via env var if needed."""
    return os.environ.get("EVAL_SESSION_ID", "bf4def75-d317-47f3-bdeb-a4f06f419b6f")