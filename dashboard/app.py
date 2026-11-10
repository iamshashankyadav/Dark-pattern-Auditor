"""Streamlit dashboard for compliance reviewers.

    streamlit run dashboard/app.py
"""

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor import db  # noqa: E402
from auditor.pipeline import FLAGGED, NEEDS_REVIEW, PASSED  # noqa: E402
from auditor.taxonomy import BY_KEY, CATEGORIES, LABEL_KEYS, VIOLATIONS  # noqa: E402

st.set_page_config(page_title="Compliance Auditor", page_icon="🛡️", layout="wide")

COLORS = {
    "deceptive_urgency": "rgba(234, 88, 12, 0.30)",
    "unsubstantiated_claims": "rgba(22, 163, 74, 0.30)",
    "subscription_trap": "rgba(37, 99, 235, 0.30)",
    "confirm_shaming": "rgba(220, 38, 38, 0.30)",
    "drip_pricing": "rgba(147, 51, 234, 0.30)",
    "bait_and_switch": "rgba(217, 119, 6, 0.30)",
}
SOLID = {
    "deceptive_urgency": "#ea580c",
    "unsubstantiated_claims": "#16a34a",
    "subscription_trap": "#2563eb",
    "confirm_shaming": "#dc2626",
    "drip_pricing": "#9333ea",
    "bait_and_switch": "#d97706",
}
SHORT = {
    "deceptive_urgency": "Urgency",
    "unsubstantiated_claims": "Claims",
    "subscription_trap": "Pricing",
    "confirm_shaming": "Shaming",
    "drip_pricing": "Drip",
    "bait_and_switch": "Bait",
    "safe": "Safe",
}


def highlight(text: str, spans: list[dict]) -> str:
    """Return HTML with each span wrapped in a colored <mark>."""
    spans = sorted((s for s in spans if s.get("start") is not None), key=lambda s: s["start"])
    out, cursor = [], 0
    for s in spans:
        if s["start"] < cursor:
            continue
        out.append(html.escape(text[cursor : s["start"]]))
        out.append(
            f'<mark title="{html.escape(BY_KEY[s["category"]].name)} ({s["confidence"]:.0%})" '
            f'style="background:{COLORS[s["category"]]};padding:1px 3px;border-radius:4px;'
            f'border-bottom:2px solid {SOLID[s["category"]]};color:inherit">'
            f"{html.escape(text[s['start']:s['end']])}</mark>"
        )
        cursor = s["end"]
    out.append(html.escape(text[cursor:]))
    return f'<div style="line-height:1.8;font-size:1.02rem">{"".join(out)}</div>'


def legend():
    chips = " ".join(
        f'<span style="background:{COLORS[c.key]};border-bottom:2px solid {SOLID[c.key]};'
        f'padding:2px 8px;border-radius:4px;margin-right:6px">{c.name}</span>'
        for c in VIOLATIONS
    )
    st.markdown(chips, unsafe_allow_html=True)


def score_chart(scores: dict):
    df = pd.DataFrame(
        {"category": [SHORT[k] for k in LABEL_KEYS], "score": [scores.get(k, 0) for k in LABEL_KEYS]}
    )
    fig = px.bar(df, x="score", y="category", orientation="h", range_x=[0, 1], height=180,
                 color="category",
                 color_discrete_map={SHORT[k]: SOLID.get(k, "#64748b") for k in LABEL_KEYS})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def _init():
    db.init_db()
    return True


def load_products(status: str | None = None) -> list[db.AuditedProduct]:
    with db.session_scope() as s:
        q = select(db.AuditedProduct).options(selectinload(db.AuditedProduct.flags))
        if status:
            q = q.where(db.AuditedProduct.status == status)
        return list(s.scalars(q.order_by(db.AuditedProduct.product_id.desc())))


def flag_dicts(product) -> list[dict]:
    return [
        {"category": f.violation_category, "confidence": float(f.confidence_score),
         "start": f.start_char, "end": f.end_char, "snippet": f.flagged_text_snippet,
         "source": f.source, "resolved": f.resolved}
        for f in product.flags
    ]


# --------------------------------------------------------------------------- pages
def page_overview():
    products = load_products()
    if not products:
        st.info("No products yet. Run `python scripts/seed_db.py` or submit items through the API.")
        return
    counts = pd.Series([p.status for p in products]).value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products audited", len(products))
    c2.metric("Flagged", int(counts.get(FLAGGED, 0)))
    c3.metric("Awaiting review", int(counts.get(NEEDS_REVIEW, 0)))
    c4.metric("Passed", int(counts.get(PASSED, 0)))

    flags = [(p.seller_id, f.violation_category) for p in products for f in p.flags if not f.resolved]
    left, right = st.columns(2)
    with left:
        st.subheader("Open flags by category")
        if flags:
            df = pd.DataFrame(flags, columns=["seller", "category"])
            df["category"] = df.category.map(SHORT)
            fig = px.histogram(df, y="category", color="category",
                               color_discrete_map={SHORT[k]: v for k, v in SOLID.items()})
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No open flags.")
    with right:
        st.subheader("Repeat offenders")
        if flags:
            df = pd.DataFrame(flags, columns=["seller", "category"])
            pivot = df.pivot_table(index="seller", columns="category", aggfunc="size", fill_value=0)
            pivot = pivot.rename(columns=SHORT)
            pivot["total"] = pivot.sum(axis=1)
            st.dataframe(pivot.sort_values("total", ascending=False), use_container_width=True)


def page_flagged():
    legend()
    products = load_products(FLAGGED)
    if not products:
        st.success("Nothing flagged.")
        return
    sellers = sorted({p.seller_id for p in products})
    chosen = st.multiselect("Seller", sellers)
    cats = st.multiselect("Category", [c.key for c in VIOLATIONS], format_func=lambda k: BY_KEY[k].name)
    for p in products:
        flags = flag_dicts(p)
        if chosen and p.seller_id not in chosen:
            continue
        if cats and not any(f["category"] in cats for f in flags):
            continue
        open_flags = [f for f in flags if not f["resolved"]]
        with st.expander(f"#{p.product_id} · {p.title} · {p.seller_id} · {len(open_flags)} open flag(s)"):
            st.markdown(highlight(p.raw_text, open_flags), unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(flags)[["category", "confidence", "snippet", "source", "resolved"]],
                use_container_width=True, hide_index=True,
            )
            if open_flags and st.button("Mark resolved (seller fixed the copy)", key=f"res-{p.product_id}"):
                with db.session_scope() as s:
                    for f in s.get(db.AuditedProduct, p.product_id).flags:
                        f.resolved = True
                st.rerun()


def page_review():
    st.caption(
        "Items whose violation confidence fell in the uncertain band. Your decisions are stored "
        "in `human_feedback` and used by `scripts/train_setfit.py --with-feedback` to retrain."
    )
    products = load_products(NEEDS_REVIEW)
    if not products:
        st.success("Review queue is empty.")
        return
    reviewer = st.text_input("Reviewer name", value="reviewer")
    for p in products:
        with st.container(border=True):
            st.markdown(f"**#{p.product_id} · {p.title}** · seller `{p.seller_id}`")
            left, right = st.columns([3, 2])
            with left:
                st.write(p.raw_text)
                if p.review_category:
                    st.caption(f"Model's best guess: {BY_KEY[p.review_category].name}")
            with right:
                score_chart(p.scores)
            default = LABEL_KEYS.index(p.review_category) if p.review_category in LABEL_KEYS else 0
            label = st.radio("Correct label", LABEL_KEYS, index=default, horizontal=True,
                             format_func=lambda k: SHORT[k], key=f"lab-{p.product_id}")
            if st.button("Submit decision", key=f"sub-{p.product_id}", type="primary"):
                with db.session_scope() as s:
                    db.record_feedback(s, p.product_id, label, reviewer)
                st.toast(f"Saved #{p.product_id} as {BY_KEY[label].name}")
                st.rerun()

    with db.session_scope() as s:
        n = s.scalar(select(func.count()).select_from(db.HumanFeedback))
    st.caption(f"{n} human decisions collected so far.")


def page_playground():
    from auditor.pipeline import audit

    st.caption("Audit any piece of copy through the full three-phase pipeline (nothing is stored).")
    text = st.text_area(
        "Product copy",
        height=140,
        value="Premium ceramic mug, 350 ml. Hurry, only 2 left at this price! "
              "Clinically proven to boost your metabolism. Free trial, then $9.99 billed monthly.",
    )
    if st.button("Audit", type="primary") and text.strip():
        with st.spinner("Running pipeline..."):
            result = audit(text)
        badge = {FLAGGED: "🔴", NEEDS_REVIEW: "🟡", PASSED: "🟢"}[result.status]
        st.subheader(f"{badge} {result.status}")
        st.caption(f"Reached phase {result.phase_reached} · Phase 1 reason: {result.heuristic_reason} "
                   f"· {result.latency_ms:.0f} ms")
        legend()
        st.markdown(highlight(text, [vars(f) for f in result.flags]), unsafe_allow_html=True)
        if result.scores:
            score_chart(result.scores)
        if result.flags:
            st.dataframe(pd.DataFrame([vars(f) for f in result.flags]), use_container_width=True,
                         hide_index=True)


_init()
st.title("🛡️ E-Commerce Compliance Auditor")
pages = {
    "Overview": page_overview,
    "Flagged products": page_flagged,
    "Human review queue": page_review,
    "Live audit": page_playground,
}
choice = st.sidebar.radio("View", list(pages))
st.sidebar.markdown("---")
st.sidebar.caption("Categories")
for c in CATEGORIES:
    st.sidebar.caption(f"• {c.name}")
pages[choice]()
