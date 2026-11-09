"""Entity-level evaluation for BIO sequences (strict span match, CoNLL style)."""

from collections import defaultdict


def bio_to_entities(tags: list[str]) -> set[tuple[str, int, int]]:
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


def entity_prf(true_seqs: list[list[str]], pred_seqs: list[list[str]]) -> dict:
    """Micro and per-type precision/recall/F1 over entity spans."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for seq_id, (t, p) in enumerate(zip(true_seqs, pred_seqs)):
        gold, pred = bio_to_entities(t), bio_to_entities(p)
        for etype, *_ in gold & pred:
            counts[etype]["tp"] += 1
        for etype, *_ in pred - gold:
            counts[etype]["fp"] += 1
        for etype, *_ in gold - pred:
            counts[etype]["fn"] += 1

    def prf(c):
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "support": c["tp"] + c["fn"]}

    total = {k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn")}
    return {"micro": prf(total), **{etype: prf(c) for etype, c in sorted(counts.items())}}


def format_report(report: dict) -> str:
    lines = [f"{'type':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}"]
    for name, m in report.items():
        lines.append(
            f"{name:<10}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>9}"
        )
    return "\n".join(lines)
