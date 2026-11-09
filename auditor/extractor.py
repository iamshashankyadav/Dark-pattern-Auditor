"""Phase 3: isolate the exact manipulative phrases inside flagged copy.

Primary: a fine-tuned transformer token classifier (BIO tags, see
``scripts/train_span_extractor.py``). Fallback when no model is trained: regex lexicon
matches. Every span is returned with character offsets so the dashboard can highlight it.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from auditor.config import get_settings
from auditor.lexicon import regex_hits
from auditor.taxonomy import BY_TAG
from auditor.text_utils import merge_spans

log = logging.getLogger(__name__)


class TransformerSpanExtractor:
    name = "token_classifier"

    def __init__(self, model_dir):
        from transformers import pipeline

        from auditor.classifier import _device

        self._pipe = pipeline(
            "token-classification",
            model=str(model_dir),
            aggregation_strategy="simple",
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
        return _expand_to_words(text, merge_spans(spans))


class RegexSpanExtractor:
    name = "regex"

    def extract(self, text: str) -> list[dict]:
        spans = [
            {"start": s, "end": e, "category": cat, "score": 1.0, "source": self.name}
            for cat, hits in regex_hits(text).items()
            for s, e, _ in hits
        ]
        return merge_spans(spans)


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
