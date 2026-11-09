"""Span-level evaluation.

Two matching modes, as is standard for span-detection tasks:

* ``strict``  - a prediction counts only if type and both boundaries match exactly (CoNLL).
* ``lenient`` - a prediction counts if it overlaps a gold span of the same type. This is the
  more relevant number for a highlighting UI, where slightly shorter/longer spans still
  point the reviewer at the right phrase.
"""

from collections import defaultdict

Span = tuple[str, int, int]  # (type, start, end_exclusive) - token or char offsets


def bio_to_entities(tags: list[str]) -> set[Span]:
    """Convert a BIO tag sequence into a set of ``(type, start_idx, end_idx_exclusive)``."""
    entities, current = set(), None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, etype = tag.partition("-")
        if current and (prefix in ("O", "B") or etype != current[0]):
            entities.add((current[0], current[1], i))
            current = None
        if prefix == "B" or (prefix == "I" and current is None):
            current = (etype, i)
    return entities


def _overlaps(a: Span, b: Span) -> bool:
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def span_prf(gold_docs: list[set[Span]], pred_docs: list[set[Span]], mode: str = "strict") -> dict:
    """Micro and per-type precision/recall/F1 over per-document span sets."""
    if mode not in ("strict", "lenient"):
        raise ValueError(mode)
    tp_p = defaultdict(int)   # predictions that are correct (for precision)
    tp_g = defaultdict(int)   # gold spans that are found   (for recall)
    n_pred, n_gold = defaultdict(int), defaultdict(int)
    for gold, pred in zip(gold_docs, pred_docs):
        for p in pred:
            n_pred[p[0]] += 1
            tp_p[p[0]] += (p in gold) if mode == "strict" else any(_overlaps(p, g) for g in gold)
        for g in gold:
            n_gold[g[0]] += 1
            tp_g[g[0]] += (g in pred) if mode == "strict" else any(_overlaps(g, p) for p in pred)

    def prf(tpp, tpg, npred, ngold):
        prec = tpp / npred if npred else 0.0
        rec = tpg / ngold if ngold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "support": ngold}

    types = sorted(set(n_pred) | set(n_gold))
    report = {"micro": prf(sum(tp_p.values()), sum(tp_g.values()),
                           sum(n_pred.values()), sum(n_gold.values()))}
    for t in types:
        report[t] = prf(tp_p[t], tp_g[t], n_pred[t], n_gold[t])
    return report


def entity_prf(true_seqs: list[list[str]], pred_seqs: list[list[str]], mode: str = "strict") -> dict:
    """Span P/R/F1 for BIO tag sequences."""
    return span_prf([bio_to_entities(t) for t in true_seqs],
                    [bio_to_entities(p) for p in pred_seqs], mode)


def format_report(report: dict) -> str:
    lines = [f"{'type':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}"]
    for name, m in report.items():
        lines.append(
            f"{name:<10}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>9}"
        )
    return "\n".join(lines)
