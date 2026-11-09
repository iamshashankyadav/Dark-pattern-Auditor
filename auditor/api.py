"""FastAPI ingestion service.

Products submitted to ``POST /products`` are stored as *Pending* and pushed onto an
asyncio queue. A background worker drains the queue in micro-batches and runs the
(blocking) model pipeline in a thread so the event loop keeps accepting traffic.

    uvicorn auditor.api:app --reload
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from auditor import db
from auditor.pipeline import NEEDS_REVIEW, audit_many
from auditor.taxonomy import BY_KEY, LABEL_KEYS

log = logging.getLogger("auditor.api")
BATCH_SIZE = 16


# --------------------------------------------------------------------------- schemas
class AuditRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class ProductIn(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    seller_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=20_000)


class ReviewIn(BaseModel):
    corrected_category: str = Field(description=f"One of {LABEL_KEYS}")
    reviewer: str = "anonymous"
    notes: str | None = None


def _product_out(p: db.AuditedProduct) -> dict:
    return {
        "product_id": p.product_id,
        "title": p.title,
        "seller_id": p.seller_id,
        "text": p.raw_text,
        "status": p.status,
        "phase_reached": p.phase_reached,
        "review_category": p.review_category,
        "scores": p.scores,
        "created_at": p.created_at,
        "audited_at": p.audited_at,
        "flags": [
            {
                "flag_id": f.flag_id,
                "category": f.violation_category,
                "category_name": BY_KEY[f.violation_category].name,
                "confidence": float(f.confidence_score),
                "snippet": f.flagged_text_snippet,
                "start": f.start_char,
                "end": f.end_char,
                "source": f.source,
                "resolved": f.resolved,
            }
            for f in p.flags
        ],
    }


# --------------------------------------------------------------------------- worker
def _process_batch(product_ids: list[int]) -> None:
    with db.session_scope() as s:
        products = [s.get(db.AuditedProduct, pid) for pid in product_ids]
        products = [p for p in products if p is not None]
        results = audit_many([p.raw_text for p in products])
        for product, result in zip(products, results):
            db.apply_audit(s, product, result)


async def _worker(queue: asyncio.Queue) -> None:
    while True:
        batch = [await queue.get()]
        while len(batch) < BATCH_SIZE and not queue.empty():
            batch.append(queue.get_nowait())
        try:
            await asyncio.to_thread(_process_batch, batch)
        except Exception:  # keep the worker alive; items stay Pending for a retry
            log.exception("Audit batch failed for products %s", batch)
        finally:
            for _ in batch:
                queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    app.state.queue = asyncio.Queue()
    # Re-enqueue anything left Pending from a previous run.
    with db.session_scope() as s:
        pending = s.scalars(
            select(db.AuditedProduct.product_id).where(db.AuditedProduct.status == "Pending")
        ).all()
    for pid in pending:
        app.state.queue.put_nowait(pid)
    worker = asyncio.create_task(_worker(app.state.queue))
    yield
    worker.cancel()


app = FastAPI(
    title="E-Commerce Compliance Auditor",
    description="Detects dark patterns in marketplace product copy.",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- routes
@app.get("/health")
async def health():
    return {"status": "ok", "queue_size": app.state.queue.qsize()}


@app.post("/audit")
async def audit_text(req: AuditRequest):
    """Synchronously audit a piece of text without storing it."""
    [result] = await asyncio.to_thread(audit_many, [req.text])
    return result.to_dict()


@app.post("/products", status_code=202)
async def submit_product(item: ProductIn):
    with db.session_scope() as s:
        product = db.create_product(s, item.title, item.seller_id, item.text)
        pid = product.product_id
    await app.state.queue.put(pid)
    return {"product_id": pid, "status": "Pending"}


@app.post("/products/batch", status_code=202)
async def submit_batch(items: list[ProductIn]):
    with db.session_scope() as s:
        ids = [db.create_product(s, i.title, i.seller_id, i.text).product_id for i in items]
    for pid in ids:
        await app.state.queue.put(pid)
    return {"product_ids": ids, "status": "Pending"}


@app.get("/products/{product_id}")
async def get_product(product_id: int):
    with db.session_scope() as s:
        product = s.get(db.AuditedProduct, product_id)
        if product is None:
            raise HTTPException(404, "Product not found")
        return _product_out(product)


@app.get("/products")
async def list_products(
    status: str | None = None,
    seller_id: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    with db.session_scope() as s:
        q = select(db.AuditedProduct).order_by(db.AuditedProduct.product_id.desc())
        if status:
            q = q.where(db.AuditedProduct.status == status)
        if seller_id:
            q = q.where(db.AuditedProduct.seller_id == seller_id)
        return [_product_out(p) for p in s.scalars(q.limit(limit).offset(offset))]


@app.get("/review-queue")
async def review_queue(limit: int = Query(50, ge=1, le=500)):
    return await list_products(status=NEEDS_REVIEW, seller_id=None, limit=limit, offset=0)


@app.post("/products/{product_id}/review")
async def review_product(product_id: int, review: ReviewIn):
    if review.corrected_category not in LABEL_KEYS:
        raise HTTPException(422, f"corrected_category must be one of {LABEL_KEYS}")
    with db.session_scope() as s:
        try:
            db.record_feedback(s, product_id, review.corrected_category, review.reviewer, review.notes)
        except KeyError:
            raise HTTPException(404, "Product not found")
        return _product_out(s.get(db.AuditedProduct, product_id))


@app.get("/stats")
async def stats():
    with db.session_scope() as s:
        by_status = dict(
            s.execute(
                select(db.AuditedProduct.status, func.count()).group_by(db.AuditedProduct.status)
            ).all()
        )
        by_category = dict(
            s.execute(
                select(db.ComplianceFlag.violation_category, func.count())
                .where(db.ComplianceFlag.resolved.is_(False))
                .group_by(db.ComplianceFlag.violation_category)
            ).all()
        )
        sellers = s.execute(
            select(db.AuditedProduct.seller_id, func.count(db.ComplianceFlag.flag_id).label("n"))
            .join(db.ComplianceFlag)
            .where(db.ComplianceFlag.resolved.is_(False))
            .group_by(db.AuditedProduct.seller_id)
            .order_by(func.count(db.ComplianceFlag.flag_id).desc())
            .limit(10)
        ).all()
    return {
        "products_by_status": by_status,
        "open_flags_by_category": by_category,
        "top_violating_sellers": [{"seller_id": sid, "open_flags": n} for sid, n in sellers],
    }
