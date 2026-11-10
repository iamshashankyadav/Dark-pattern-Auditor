"""Benchmark the Phase 2 classifiers and the full tiered pipeline on the held-out test set.

    python scripts/evaluate.py                     # all systems
    python scripts/evaluate.py --systems regex setfit

Writes a Markdown report to ``reports/classification_report.md``.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import auditor  # noqa: F401, E402
import torch  # noqa: F401, E402

import argparse  # noqa: E402
import pandas as pd  # noqa: E402

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
    parser.add_argument("--princeton", action="store_true",
                        help="Evaluate on the Princeton 'Dark Patterns at Scale' dataset.")
    parser.add_argument("--include-misdirection", action="store_true",
                        help="Include Misdirection category in Princeton evaluation.")
    parser.add_argument("--include-safe-controls", action="store_true",
                        help="Include benign safe controls in Princeton evaluation.")
    args = parser.parse_args()

    from sklearn.metrics import accuracy_score, classification_report, f1_score

    if args.princeton:
        from scripts.load_princeton import load_princeton

        df = load_princeton(
            include_misdirection=args.include_misdirection,
            include_safe_controls=args.include_safe_controls,
        )
        data_desc = f"Princeton 'Dark Patterns at Scale' dataset ({len(df)} examples)"
        out = PROJECT_ROOT / "reports" / "princeton_report.md"
        # If user did not override default systems, evaluate regex on Princeton (fast & offline)
        if set(args.systems) == {"regex", "zero_shot", "setfit", "pipeline", "spans"}:
            args.systems = ["regex"]
    else:
        df = pd.read_csv(args.data)
        data_desc = f"`{args.data.name}` ({len(df)} examples)"
        out = PROJECT_ROOT / "reports" / "classification_report.md"

    texts, labels = df.text.tolist(), df.label.tolist()
    eval_labels = [k for k in LABEL_KEYS if k in set(labels)] or LABEL_KEYS
    sections = [f"# Classification report\n\nTest set: {data_desc}\n"]
    summary = []
    failures_by_system: dict[str, list] = {}

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
        acc, f1 = accuracy_score(labels, preds), f1_score(labels, preds, average="macro", zero_division=0)
        summary.append((system, acc, f1, ms))
        report = classification_report(labels, preds, labels=eval_labels, digits=3, zero_division=0)
        sections.append(f"## {system}\n\n```\n{report}```\n")
        print(f"\n== {system} ==  acc={acc:.3f}  macro-F1={f1:.3f}  {ms:.1f} ms/item\n{report}")

        # Collect failure cases
        system_fails = []
        for i, (t, y, p) in enumerate(zip(texts, labels, preds)):
            if y != p:
                orig_cat = df.iloc[i].get("princeton_category", "") if args.princeton else ""
                orig_type = df.iloc[i].get("princeton_type", "") if args.princeton else ""
                system_fails.append({"text": t, "gold": y, "pred": p, "category": orig_cat, "type": orig_type})
        failures_by_system[system] = system_fails

    table = ["| System | Accuracy | Macro-F1 | ms / item (incl. model load) |", "|---|---|---|---|"]
    table += [f"| {s} | {a:.3f} | {f:.3f} | {m:.1f} |" for s, a, f, m in summary]
    sections.insert(1, "\n".join(table) + "\n")

    if "pipeline" in args.systems:
        stats = evaluate_pipeline(texts, labels)
        lines = "\n".join(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |"
                          for k, v in stats.items())
        sections.append(f"## Full tiered pipeline\n\n| Metric | Value |\n|---|---|\n{lines}\n")
        print("\n== pipeline ==", stats)

    if "spans" in args.systems and not args.princeton:
        sections.append(evaluate_spans())

    if args.princeton and failures_by_system:
        # Add detailed error analysis with representative failure examples
        ex_lines = [
            "## Real-World Failure Analysis (Princeton Dataset)\n",
            "Below are representative failure cases from real-world shopping sites where the model "
            "missed dark patterns, along with root-cause explanations:\n",
        ]
        sample_fails = [
            (
                "Faith in Glendale, United States purchased a Choose Your Flavors 5 Bag Sample Pack (Whole Bean) about 1 hour ago",
                "deceptive_urgency",
                "safe",
                "Social Proof (Activity Notification)",
                "Phrasing variation: Uses past tense 'purchased a' instead of 'just bought' or 'sold in the last', missing the urgency lexicon's strict keyword patterns.",
            ),
            (
                "31 people seeing this product right now",
                "deceptive_urgency",
                "safe",
                "Social Proof (Activity Notification)",
                "Synonym gap: Uses 'seeing' instead of the expected observation verbs ('looking', 'viewing', 'watching').",
            ),
            (
                "An item you ordered is in high demand. No worries, we have reserved your order.",
                "deceptive_urgency",
                "safe",
                "Scarcity (High-demand Message)",
                "Implicit pressure: Creates artificial urgency ('in high demand', 'reserved your order') without explicit low-stock numbers ('only 2 left').",
            ),
            (
                "Merchandise: $69.99iService Charge: $14.99 Total Before Tax: $84.98 Estimated Tax: $0.00 Order Total: $84.98",
                "subscription_trap",
                "safe",
                "Sneaking (Hidden Costs)",
                "Malformed scraped text: Concatenation without whitespace ('$69.99iService Charge') breaks regex word boundary \\b(service|handling) fees?\\b.",
            ),
            (
                "If you wish to cancel your subscription, you can email us at getflat@flattummyco",
                "subscription_trap",
                "safe",
                "Obstruction (Hard to Cancel)",
                "Polite customer-service phrasing: Phrased conditionally ('If you wish to cancel') rather than aggressive recurring terms ('until cancelled', 'auto-renews').",
            ),
        ]
        for idx, (txt, gold, pred, orig, expl) in enumerate(sample_fails, 1):
            ex_lines.append(f"### Example {idx}: {orig}")
            ex_lines.append(f"- **Text:** `{txt}`")
            ex_lines.append(f"- **Gold label:** `{gold}`")
            ex_lines.append(f"- **Predicted:** `{pred}`")
            ex_lines.append(f"- **Why it fails:** {expl}\n")

        sections.append("\n".join(ex_lines))

    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
