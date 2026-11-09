"""Phase 3: isolate the exact manipulative phrases inside flagged copy.

Primary: a fine-tuned transformer token classifier (BIO tags, see
``scripts/train_span_extractor.py``). Fallback when no model is trained: regex lexicon
matches. Every span is returned with character offsets so the dashboard can highlight it.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from auditor.config import get_settings
from auditor.lexicon import regex_hits
from auditor.taxonomy import BY_TAG
from auditor.text_utils import merge_spans

log = logging.getLogger(__name__)


_CONNECTOR = re.compile(r"^[\s,;:\-–]*(?:(?:and|or|but|with|while|to|&)[\s,;:\-–]*)?$", re.IGNORECASE)


class TransformerSpanExtractor:
    name = "token_classifier"

    def __init__(self, model_dir, min_score: float = 0.6):
        from transformers import pipeline

        from auditor.classifier import _device

        self.min_score = min_score
        # "first": each word takes the label of its first sub-token, which avoids
        # spans that start or stop in the middle of a word.
        self._pipe = pipeline(
            "token-classification",
            model=str(model_dir),
            aggregation_strategy="first",
            device=_device(),
        )

    def extract(self, text: str) -> list[dict]:
        spans = []
        for ent in self._pipe(text):
            category = BY_TAG.get(ent["entity_group"])
            if category is None:
                continue
            spans.append({
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "category": category.key,
                "score": float(ent["score"]),
                "source": self.name,
            })
        spans = _join_fragments(text, _expand_to_words(text, merge_spans(spans)))
        return [s for s in spans if s["score"] >= self.min_score and _has_word(text, s)]


class RegexSpanExtractor:
    name = "regex"

    def extract(self, text: str) -> list[dict]:
        spans = [
            {"start": s, "end": e, "category": cat, "score": 1.0, "source": self.name}
            for cat, hits in regex_hits(text).items()
            for s, e, _ in hits
        ]
        return merge_spans(spans)


def _join_fragments(text: str, spans: list[dict]) -> list[dict]:
    """Join same-category spans separated only by punctuation or a connective
    ("balances hormones and" + "prevents illness" -> one span)."""
    joined: list[dict] = []
    for span in sorted(spans, key=lambda s: s["start"]):
        prev = joined[-1] if joined else None
        if (prev and prev["category"] == span["category"]
                and _CONNECTOR.match(text[prev["end"]:span["start"]])):
            # length-weighted score so a short noisy fragment doesn't dominate
            a, b = prev["end"] - prev["start"], span["end"] - span["start"]
            prev["score"] = (prev["score"] * a + span["score"] * b) / (a + b)
            prev["end"] = span["end"]
        else:
            joined.append(dict(span))
    return joined


def _has_word(text: str, span: dict) -> bool:
    """Drop degenerate spans: a lone number, a single short word, or punctuation."""
    piece = text[span["start"]:span["end"]]
    words = re.findall(r"[A-Za-z]{3,}", piece)
    return len(words) >= 2 or (len(words) == 1 and len(piece) >= 6)


def _expand_to_words(text: str, spans: list[dict]) -> list[dict]:
    """Snap span edges outward to word boundaries (subword pieces can cut words)."""
    for span in spans:
        while span["start"] > 0 and text[span["start"] - 1].isalnum():
            span["start"] -= 1
        while span["end"] < len(text) and text[span["end"]].isalnum():
            span["end"] += 1
    return spans


@lru_cache
def get_extractor():
    model_dir = get_settings().span_model_dir
    if (model_dir / "config.json").exists():
        return TransformerSpanExtractor(model_dir)
    log.warning("Span model not found at %s; using regex span extractor.", model_dir)
    return RegexSpanExtractor()


def extract_spans(text: str, categories: set[str] | None = None, extractor=None) -> list[dict]:
    extractor = extractor or get_extractor()
    spans = extractor.extract(text)
    if categories is not None:
        spans = [s for s in spans if s["category"] in categories]
    for span in spans:
        span["text"] = text[span["start"] : span["end"]]
    return spans
