"""Sentence-embedding semantic similarity against the risk phrase bank."""

from functools import lru_cache

from auditor.config import get_settings
from auditor.lexicon import PHRASE_BANK
from auditor.text_utils import split_sentences


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


@lru_cache
def _bank_embeddings():
    model = _model()
    return {
        category: model.encode(phrases, convert_to_tensor=True, normalize_embeddings=True)
        for category, phrases in PHRASE_BANK.items()
    }


def semantic_scores(text: str) -> dict[str, dict]:
    """Max cosine similarity of any sentence in ``text`` to each category's phrase bank.

    Returns ``{category: {"score": float, "sentence": str, "start": int, "end": int}}``.
    """
    from sentence_transformers import util

    sentences = split_sentences(text) or [(0, len(text), text)]
    sent_emb = _model().encode(
        [s[2] for s in sentences], convert_to_tensor=True, normalize_embeddings=True
    )
    results = {}
    for category, bank in _bank_embeddings().items():
        sims = util.cos_sim(sent_emb, bank)  # [n_sentences, n_phrases]
        best_per_sentence = sims.max(dim=1).values
        idx = int(best_per_sentence.argmax())
        start, end, sentence = sentences[idx]
        results[category] = {
            "score": float(best_per_sentence[idx]),
            "sentence": sentence,
            "start": start,
            "end": end,
        }
    return results
