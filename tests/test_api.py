import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(stub_models, engine, monkeypatch):
    from auditor import api, db

    monkeypatch.setattr(db, "get_engine", lambda url=None: engine)
    with TestClient(api.app) as c:
        yield c


def _wait_for(client, pid, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/products/{pid}").json()
        if body["status"] != "Pending":
            return body
        time.sleep(0.05)
    raise AssertionError("product was never audited")


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_audit_endpoint(client):
    body = client.post("/audit", json={"text": "Hurry, only 1 left - deal ends tonight!"}).json()
    assert body["status"] == "Flagged"


def test_ingest_review_and_stats(client):
    r = client.post("/products", json={"title": "Tea", "seller_id": "s1", "text": "Our tea is natural."})
    assert r.status_code == 202
    pid = r.json()["product_id"]
    assert _wait_for(client, pid)["status"] == "Needs Review"
    assert any(p["product_id"] == pid for p in client.get("/review-queue").json())

    reviewed = client.post(f"/products/{pid}/review",
                           json={"corrected_category": "unsubstantiated_claims", "reviewer": "qa"}).json()
    assert reviewed["status"] == "Flagged"

    stats = client.get("/stats").json()
    assert stats["top_violating_sellers"][0]["seller_id"] == "s1"


def test_review_rejects_unknown_category(client):
    pid = client.post("/products", json={"title": "x", "seller_id": "s", "text": "Blue pen."}).json()["product_id"]
    assert client.post(f"/products/{pid}/review", json={"corrected_category": "nope"}).status_code == 422
