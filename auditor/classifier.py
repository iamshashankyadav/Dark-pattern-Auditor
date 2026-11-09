"""Phase 2: broad risk classification.

Two interchangeable backends share one interface - ``predict_proba(sentences)`` returning
one ``{category_key: probability}`` dict per sentence:

* ``ZeroShotClassifier`` - ``facebook/bart-large-mnli`` NLI model; needs no training data.
* ``SetFitClassifier``   - a sentence-transformer fine-tuned with contrastive loss on the
  seed corpus (see ``scripts/train_setfit.py``); much faster and runs fine on CPU.

Documents are scored sentence by sentence so a single manipulative line inside a long,
otherwise honest description is not diluted. The document score of a violation is the
maximum over its sentences; the document "safe" score is the minimum over its sentences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from auditor.config import get_settings
from auditor.taxonomy import CATEGORIES, LABEL_KEYS, SAFE, VIOLATIONS
from auditor.text_utils import split_sentences

log = logging.getLogger(__name__)


def _device() -> int:
    configured = get_settings().device
    if configured != -1:
        return configured
    import torch

    return 0 if torch.cuda.is_available() else -1


class ZeroShotClassifier:
    name = "zero_shot"

    def __init__(self, model_name: str | None = None):
        from transformers import pipeline

        self.model_name = model_name or get_settings().zero_shot_model
        self._pipe = pipeline("zero-shot-classification", model=self.model_name, device=_device())
        self._hyp_to_key = {c.hypothesis: c.key for c in CATEGORIES}

    def predict_proba(self, sentences: list[str]) -> list[dict[str, float]]:
        if not sentences:
            return []
        outputs = self._pipe(
            sentences,
            candidate_labels=list(self._hyp_to_key),
            hypothesis_template="This product text {}.",
            multi_label=False,
            batch_size=16,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [
            {self._hyp_to_key[label]: float(score) for label, score in zip(o["labels"], o["scores"])}
            for o in outputs
        ]


class SetFitClassifier:
    name = "setfit"

    def __init__(self, model_dir=None):
        from setfit import SetFitModel

        self.model_dir = model_dir or get_settings().setfit_model_dir
        self._model = SetFitModel.from_pretrained(str(self.model_dir))
        if _device() == 0:
            self._model.to("cuda")

    def predict_proba(self, sentences: list[str]) -> list[dict[str, float]]:
        if not sentences:
            return []
        probs = self._model.predict_proba(sentences, as_numpy=True)
        return [{key: float(p) for key, p in zip(LABEL_KEYS, row)} for row in probs]


@lru_cache
def get_classifier(backend: str | None = None):
    backend = backend or get_settings().classifier_backend
    if backend == "setfit":
        model_dir = get_settings().setfit_model_dir
        if (model_dir / "config.json").exists() or (model_dir / "model_head.pkl").exists():
            return SetFitClassifier(model_dir)
        log.warning("SetFit model not found at %s; falling back to zero-shot.", model_dir)
    return ZeroShotClassifier()


@dataclass
class DocumentScores:
    scores: dict[str, float]                       # document-level score per category
    sentences: list[tuple[int, int, str]] = field(default_factory=list)
    sentence_probs: list[dict[str, float]] = field(default_factory=list)

    @property
    def top_violation(self) -> tuple[str, float]:
        key = max((c.key for c in VIOLATIONS), key=lambda k: self.scores[k])
        return key, self.scores[key]

    def best_sentence(self, category: str) -> tuple[int, int, str, float]:
        idx = max(range(len(self.sentences)), key=lambda i: self.sentence_probs[i][category])
        start, end, text = self.sentences[idx]
        return start, end, text, self.sentence_probs[idx][category]


def score_documents(texts: list[str], classifier=None) -> list[DocumentScores]:
    """Classify each document by batching all of its sentences through the classifier."""
    classifier = classifier or get_classifier()
    per_doc = [split_sentences(t) or [(0, len(t), t)] for t in texts]
    flat = [s[2] for sents in per_doc for s in sents]
    probs = classifier.predict_proba(flat)

    results, i = [], 0
    for sents in per_doc:
        doc_probs = probs[i : i + len(sents)]
        i += len(sents)
        scores = {c.key: max(p[c.key] for p in doc_probs) for c in VIOLATIONS}
        scores[SAFE.key] = min(p[SAFE.key] for p in doc_probs)
        results.append(DocumentScores(scores, sents, doc_probs))
    return results
