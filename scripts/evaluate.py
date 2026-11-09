"""Benchmark the Phase 2 classifiers and the full tiered pipeline on the held-out test set.

    python scripts/evaluate.py                     # all systems
    python scripts/evaluate.py --systems regex setfit

Writes a Markdown report to ``reports/classification_report.md``.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor.config import PROJECT_ROOT  # noqa: E402
from auditor.lexicon import regex_hits  # noqa: E402
from auditor.taxonomy import LABEL_KEYS, SAFE  # noqa: E402


def predict_regex(texts):
    preds = []
    for t in texts:
        hits = {k: len(v) for k, v in regex_hits(t).items()}
        preds.append(max(hits, key=hits.get) if hits else SAFE.key)
    return preds


def predict_with(classifier, texts):
    from auditor.classifier import score_documents

    preds = []
    for ds in score_documents(texts, classifier=classifier):
        key, score = ds.top_violation
        preds.append(key if score > ds.scores[SAFE.key] else SAFE.key)
    return preds


def evaluate_pipeline(texts, labels):
    from auditor.pipeline import FLAGGED, NEEDS_REVIEW, audit_many

    results = audit_many(texts)
    auto = [(r, y) for r, y in zip(results, labels) if r.status != NEEDS_REVIEW]
    correct = 0
    for r, y in auto:
        if r.status == FLAGGED:
            correct += y in {f.category for f in r.flags}
        else:
            correct += y == SAFE.key
    phase1_pass = sum(r.phase_reached == 1 for r in results)
    return {
        "auto-decided": len(auto),
        "sent to human review": len(results) - len(auto),
        "accuracy on auto-decided": correct / len(auto) if auto else float("nan"),
        "passed at phase 1 (no transformer cost)": phase1_pass,
    }


def evaluate_spans() -> str:
    """End-to-end Phase 3 span extraction on the hand-annotated test set (char offsets)."""
    from auditor.config import get_settings
    from auditor.extractor import RegexSpanExtractor, TransformerSpanExtractor
    from auditor.metrics import format_report, span_prf
    from auditor.spans import parse_markup
    from auditor.taxonomy import BY_KEY

    lines = (PROJECT_ROOT / "data" / "span_test.txt").read_text(encoding="utf-8").splitlines()
    docs = [parse_markup(l) for l in lines if l.strip()]
    gold = [{(s["label"], s["start"], s["end"]) for s in d["spans"]} for d in docs]

    extractors = [RegexSpanExtractor()]
    if (get_settings().span_model_dir / "config.json").exists():
        extractors.append(TransformerSpanExtractor(get_settings().span_model_dir))

    out = [f"## Phase 3 span extraction\n\nTest set: `span_test.txt` ({len(docs)} documents, "
           f"{sum(map(len, gold))} gold spans)\n"]
    for ex in extractors:
        pred = [{(BY_KEY[s["category"]].span_tag, s["start"], s["end"]) for s in ex.extract(d["text"])}
                for d in docs]
        for mode in ("strict", "lenient"):
            report = format_report(span_prf(gold, pred, mode))
            out.append(f"### {ex.name} ({mode})\n\n```\n{report}\n```\n")
            print(f"\n== spans: {ex.name} ({mode}) ==\n{report}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+",
                        default=["regex", "zero_shot", "setfit", "pipeline", "spans"])
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "test_set.csv")
    args = parser.parse_args()

    from sklearn.metrics import accuracy_score, classification_report, f1_score

    df = pd.read_csv(args.data)
    texts, labels = df.text.tolist(), df.label.tolist()
    sections = [f"# Classification report\n\nTest set: `{args.data.name}` ({len(df)} examples)\n"]
    summary = []

    for system in args.systems:
        if system in ("pipeline", "spans"):
            continue
        t0 = time.perf_counter()
        if system == "regex":
            preds = predict_regex(texts)
        elif system == "zero_shot":
            from auditor.classifier import ZeroShotClassifier

            preds = predict_with(ZeroShotClassifier(), texts)
        elif system == "setfit":
            from auditor.classifier import SetFitClassifier

            preds = predict_with(SetFitClassifier(), texts)
        else:
            raise SystemExit(f"Unknown system {system}")
        ms = (time.perf_counter() - t0) * 1000 / len(texts)
        acc, f1 = accuracy_score(labels, preds), f1_score(labels, preds, average="macro")
        summary.append((system, acc, f1, ms))
        report = classification_report(labels, preds, labels=LABEL_KEYS, digits=3, zero_division=0)
        sections.append(f"## {system}\n\n```\n{report}```\n")
        print(f"\n== {system} ==  acc={acc:.3f}  macro-F1={f1:.3f}  {ms:.1f} ms/item\n{report}")

    table = ["| System | Accuracy | Macro-F1 | ms / item (incl. model load) |", "|---|---|---|---|"]
    table += [f"| {s} | {a:.3f} | {f:.3f} | {m:.1f} |" for s, a, f, m in summary]
    sections.insert(1, "\n".join(table) + "\n")

    if "pipeline" in args.systems:
        stats = evaluate_pipeline(texts, labels)
        lines = "\n".join(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |"
                          for k, v in stats.items())
        sections.append(f"## Full tiered pipeline\n\n| Metric | Value |\n|---|---|\n{lines}\n")
        print("\n== pipeline ==", stats)

    if "spans" in args.systems:
        sections.append(evaluate_spans())

    out = PROJECT_ROOT / "reports" / "classification_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
