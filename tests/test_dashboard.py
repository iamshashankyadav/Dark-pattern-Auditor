"""Smoke-test every dashboard page with Streamlit's headless AppTest runner."""

import pytest
from streamlit.testing.v1 import AppTest

from auditor.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "dashboard" / "app.py")
PAGES = ["Overview", "Flagged products", "Human review queue"]


@pytest.fixture
def seeded(stub_models, engine, monkeypatch):
    from auditor import db
    from auditor.pipeline import audit

    monkeypatch.setattr(db, "get_engine", lambda url=None: engine)
    texts = {
        "flagged": "Hurry, only 2 left in stock - offer ends tonight!",
        "review": "Our tea is natural.",
        "safe": "Blue cotton socks.",
    }
    with db.session_scope(engine) as s:
        for title, text in texts.items():
            db.apply_audit(s, db.create_product(s, title, "S1", text), audit(text))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(seeded, page):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


def test_review_submission_updates_status(seeded, engine):
    from auditor import db

    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Human review queue").run()
    at.button(key=next(b.key for b in at.button if b.key and b.key.startswith("sub-"))).click().run()
    assert not at.exception
    with db.session_scope(engine) as s:
        assert s.query(db.HumanFeedback).count() == 1
