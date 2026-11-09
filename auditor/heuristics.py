"""Phase 1: cheap, high-recall filter that decides whether text needs the transformer tier."""

from dataclasses import dataclass, field

from auditor.config import get_settings
from auditor.lexicon import regex_hits


@dataclass
class HeuristicResult:
    suspicious: bool
    regex_hits: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)
    semantic: dict[str, dict] = field(default_factory=dict)
    reason: str = ""


def screen(text: str, use_semantic: bool = True) -> HeuristicResult:
    """Return whether ``text`` is suspicious.

    Regex runs first (microseconds). Only text that the regex layer considers clean is
    embedded, so paraphrased dark patterns ("supplies are vanishing quickly") are still
    caught without paying the embedding cost on every obviously risky listing.
    """
    hits = regex_hits(text)
    if hits:
        return HeuristicResult(True, hits, reason="regex:" + ",".join(sorted(hits)))

    if use_semantic:
        from auditor.embeddings import semantic_scores

        threshold = get_settings().semantic_threshold
        sem = semantic_scores(text)
        triggered = sorted(c for c, r in sem.items() if r["score"] >= threshold)
        if triggered:
            return HeuristicResult(True, hits, sem, reason="semantic:" + ",".join(triggered))
        return HeuristicResult(False, hits, sem, reason="clean")

    return HeuristicResult(False, hits, reason="clean")
