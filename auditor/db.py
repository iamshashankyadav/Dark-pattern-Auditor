"""Persistence layer (SQLAlchemy). Works with PostgreSQL in production and SQLite locally."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from auditor.config import get_settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AuditedProduct(Base):
    __tablename__ = "audited_products"

    product_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255))
    seller_id: Mapped[str] = mapped_column(String(100), index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="Pending", index=True)
    phase_reached: Mapped[int | None] = mapped_column(Integer)
    review_category: Mapped[str | None] = mapped_column(String(150))
    scores_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    audited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    flags: Mapped[list[ComplianceFlag]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    feedback: Mapped[list[HumanFeedback]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )

    @property
    def scores(self) -> dict:
        return json.loads(self.scores_json) if self.scores_json else {}


class ComplianceFlag(Base):
    __tablename__ = "compliance_flags"

    flag_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("audited_products.product_id"), index=True)
    violation_category: Mapped[str] = mapped_column(String(150))
    confidence_score: Mapped[float] = mapped_column(Numeric(5, 4))
    flagged_text_snippet: Mapped[str] = mapped_column(Text)
    start_char: Mapped[int | None] = mapped_column(Integer)
    end_char: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str | None] = mapped_column(String(50))
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    flagged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    product: Mapped[AuditedProduct] = relationship(back_populates="flags")


class HumanFeedback(Base):
    __tablename__ = "human_feedback"

    feedback_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("audited_products.product_id"), index=True)
    corrected_category: Mapped[str] = mapped_column(String(150))
    reviewer: Mapped[str] = mapped_column(String(100), default="anonymous")
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    product: Mapped[AuditedProduct] = relationship(back_populates="feedback")


@lru_cache
def get_engine(url: str | None = None):
    url = url or get_settings().database_url
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, **kwargs)


def init_db(engine=None):
    Base.metadata.create_all(engine or get_engine())


@contextmanager
def session_scope(engine=None):
    session = sessionmaker(bind=engine or get_engine(), expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_product(session, title: str, seller_id: str, raw_text: str) -> AuditedProduct:
    product = AuditedProduct(title=title, seller_id=seller_id, raw_text=raw_text, status="Pending")
    session.add(product)
    session.flush()
    return product


def apply_audit(session, product: AuditedProduct, result) -> AuditedProduct:
    """Store an ``AuditResult`` on a product, replacing any previous flags."""
    product.status = result.status
    product.phase_reached = result.phase_reached
    product.review_category = result.review_category
    product.scores_json = json.dumps(result.scores)
    product.audited_at = _now()
    product.flags.clear()
    for f in result.flags:
        product.flags.append(
            ComplianceFlag(
                violation_category=f.category,
                confidence_score=f.confidence,
                flagged_text_snippet=f.snippet,
                start_char=f.start,
                end_char=f.end,
                source=f.source,
            )
        )
    return product


def record_feedback(session, product_id: int, corrected_category: str, reviewer: str,
                    notes: str | None = None) -> HumanFeedback:
    """Store a reviewer decision and update the product's status/flags accordingly."""
    product = session.get(AuditedProduct, product_id)
    if product is None:
        raise KeyError(product_id)
    fb = HumanFeedback(product_id=product_id, corrected_category=corrected_category,
                       reviewer=reviewer, notes=notes)
    session.add(fb)
    if corrected_category == "safe":
        product.status = "Passed"
        for flag in product.flags:
            flag.resolved = True
    else:
        product.status = "Flagged"
        if not any(f.violation_category == corrected_category for f in product.flags):
            product.flags.append(ComplianceFlag(
                violation_category=corrected_category,
                confidence_score=1.0,
                flagged_text_snippet=product.raw_text[:500],
                source="human",
            ))
    return fb


def fetch_feedback_examples(engine=None) -> list[tuple[str, str]]:
    """Latest human label per product, as ``(text, label)`` training pairs."""
    with session_scope(engine) as s:
        latest = (
            select(HumanFeedback.product_id, func.max(HumanFeedback.feedback_id).label("fid"))
            .group_by(HumanFeedback.product_id)
            .subquery()
        )
        rows = s.execute(
            select(AuditedProduct.raw_text, HumanFeedback.corrected_category)
            .join(HumanFeedback, HumanFeedback.product_id == AuditedProduct.product_id)
            .join(latest, latest.c.fid == HumanFeedback.feedback_id)
        ).all()
    return [(text, label) for text, label in rows]
