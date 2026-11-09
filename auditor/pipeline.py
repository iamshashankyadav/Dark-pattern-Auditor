"""Tiered audit pipeline: Phase 1 heuristics -> Phase 2 classifier -> Phase 3 span extraction."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from auditor.classifier import score_documents
from auditor.config import get_settings
from auditor.extractor import extract_spans
from auditor.heuristics import screen
from auditor.taxonomy import BY_KEY, VIOLATIONS

PENDING = "Pending"
PASSED = "Passed"
FLAGGED = "Flagged"
NEEDS_REVIEW = "Needs Review"


@dataclass
class Flag:
    category: str
    category_name: str
    confidence: float
    snippet: str
    start: int
    end: int
    source: str


@dataclass
class AuditResult:
    text: str
    status: str
    phase_reached: int
    heuristic_reason: str
    scores: dict[str, float] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    review_category: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _build_flags(text: str, category_keys: list[str], doc_scores) -> list[Flag]:
    spans = extract_spans(text, set(category_keys))
    flags = []
    for key in category_keys:
        name = BY_KEY[key].name
        confidence = round(doc_scores.scores[key], 4)
        cat_spans = [s for s in spans if s["category"] == key]
        if cat_spans:
            flags += [
                Flag(key, name, confidence, s["text"], s["start"], s["end"], s["source"])
                for s in cat_spans
            ]
        else:
            # No phrase-level span found: fall back to the sentence the classifier blamed.
            start, end, sentence, _ = doc_scores.best_sentence(key)
            flags.append(Flag(key, name, confidence, sentence, start, end, "sentence"))
    return flags


def audit_many(texts: list[str], use_semantic: bool = True) -> list[AuditResult]:
    settings = get_settings()
    t0 = time.perf_counter()
    screens = [screen(t, use_semantic=use_semantic) for t in texts]
    suspicious_idx = [i for i, s in enumerate(screens) if s.suspicious]
    doc_scores = dict(zip(suspicious_idx, score_documents([texts[i] for i in suspicious_idx])))

    results = []
    for i, (text, h) in enumerate(zip(texts, screens)):
        if not h.suspicious:
            results.append(AuditResult(text, PASSED, 1, h.reason))
            continue

        ds = doc_scores[i]
        scores = {k: round(v, 4) for k, v in ds.scores.items()}
        flagged = [c.key for c in VIOLATIONS if ds.scores[c.key] >= settings.flag_threshold]
        top_key, top_score = ds.top_violation

        if flagged:
            flagged.sort(key=lambda k: -ds.scores[k])
            result = AuditResult(text, FLAGGED, 3, h.reason, scores, _build_flags(text, flagged, ds))
        elif top_score >= settings.review_threshold or ds.scores["safe"] < settings.review_threshold:
            result = AuditResult(text, NEEDS_REVIEW, 2, h.reason, scores, review_category=top_key)
        else:
            result = AuditResult(text, PASSED, 2, h.reason, scores)
        results.append(result)

    per_item = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
    for r in results:
        r.latency_ms = round(per_item, 2)
    return results


def audit(text: str, use_semantic: bool = True) -> AuditResult:
    return audit_many([text], use_semantic=use_semantic)[0]
