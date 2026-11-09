from auditor import db
from auditor.pipeline import FLAGGED, NEEDS_REVIEW, PASSED, audit, audit_many


def test_safe_text_passes_in_phase_one(stub_models):
    r = audit("Ceramic mug, 350 ml, dishwasher safe.")
    assert r.status == PASSED and r.phase_reached == 1


def test_strong_violation_is_flagged_with_span(stub_models):
    text = "Great blender. Hurry, only 2 left in stock - offer ends tonight!"
    r = audit(text)
    assert r.status == FLAGGED
    assert r.flags and r.flags[0].category == "deceptive_urgency"
    for f in r.flags:
        assert text[f.start:f.end] == f.snippet


def test_weak_signal_goes_to_review(stub_models):
    r = audit("Our tea is natural.")
    assert r.status == NEEDS_REVIEW
    assert r.review_category == "unsubstantiated_claims"


def test_batch_preserves_order(stub_models):
    texts = ["Plain socks.", "Clinically proven to cure diabetes and melt fat.", "Blue pen."]
    statuses = [r.status for r in audit_many(texts)]
    assert statuses == [PASSED, FLAGGED, PASSED]


def test_persist_audit_and_feedback(stub_models, engine):
    text = "Free trial, then billed monthly until you cancel. Auto-renews yearly."
    with db.session_scope(engine) as s:
        p = db.create_product(s, "Vitamins", "seller-1", text)
        db.apply_audit(s, p, audit(text))
        pid = p.product_id
        assert p.status == FLAGGED and p.flags

    with db.session_scope(engine) as s:
        db.record_feedback(s, pid, "safe", reviewer="qa")
    with db.session_scope(engine) as s:
        p = s.get(db.AuditedProduct, pid)
        assert p.status == PASSED
        assert all(f.resolved for f in p.flags)

    assert db.fetch_feedback_examples(engine) == [(text, "safe")]
