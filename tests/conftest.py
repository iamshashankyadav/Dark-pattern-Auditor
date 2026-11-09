import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor.taxonomy import LABEL_KEYS, SAFE  # noqa: E402
from auditor.lexicon import PATTERNS  # noqa: E402


class KeywordClassifier:
    """Deterministic stand-in for the transformer classifier, driven by the regex lexicon."""

    name = "stub"

    def predict_proba(self, sentences):
        out = []
        for s in sentences:
            hits = {k: sum(bool(p.search(s)) for p in pats) for k, pats in PATTERNS.items()}
            best = max(hits, key=hits.get)
            if hits[best] == 0:
                probs = {k: 0.02 for k in LABEL_KEYS}
                probs[SAFE.key] = 0.94
            elif hits[best] == 1:  # single weak cue -> uncertain
                probs = {k: 0.1 for k in LABEL_KEYS}
                probs[best] = 0.6
                probs[SAFE.key] = 0.2
            else:
                probs = {k: 0.03 for k in LABEL_KEYS}
                probs[best] = 0.91
            out.append(probs)
        return out


@pytest.fixture
def stub_models(monkeypatch):
    """Route the pipeline through cheap stubs: no model downloads, no GPU."""
    from auditor import classifier, extractor, heuristics, pipeline

    stub = KeywordClassifier()
    regex_extractor = extractor.RegexSpanExtractor()

    def score_documents(texts):
        return classifier.score_documents(texts, classifier=stub)

    def extract_spans(text, categories):
        return extractor.extract_spans(text, categories, extractor=regex_extractor)

    def screen(text, use_semantic=True):
        return heuristics.screen(text, use_semantic=False)  # skip the embedding model

    monkeypatch.setattr(pipeline, "score_documents", score_documents)
    monkeypatch.setattr(pipeline, "extract_spans", extract_spans)
    monkeypatch.setattr(pipeline, "screen", screen)
    return stub


@pytest.fixture
def engine(tmp_path):
    from auditor import db

    eng = db.get_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    db.init_db(eng)
    return eng
