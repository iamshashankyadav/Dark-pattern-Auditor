"""Phase 3 training: fine-tune a transformer for token-level dark-pattern extraction (BIO NER).

    python scripts/generate_span_data.py
    python scripts/train_span_extractor.py --epochs 3

Evaluates entity-level precision/recall/F1 (seqeval) on the synthetic dev split and on
the hand-annotated ``data/span_test.txt``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor.config import PROJECT_ROOT, get_settings  # noqa: E402
from auditor.metrics import entity_prf, format_report  # noqa: E402
from auditor.spans import char_spans_to_token_labels, parse_markup  # noqa: E402
from auditor.taxonomy import SPAN_LABELS  # noqa: E402

LABEL2ID = {label: i for i, label in enumerate(SPAN_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_markup(path: Path) -> list[dict]:
    return [parse_markup(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode(docs: list[dict], tokenizer, max_length: int = 192):
    enc = tokenizer(
        [d["text"] for d in docs], truncation=True, max_length=max_length, return_offsets_mapping=True
    )
    all_labels = []
    for offsets, doc in zip(enc["offset_mapping"], docs):
        bio = char_spans_to_token_labels([tuple(o) for o in offsets], doc["spans"])
        all_labels.append([-100 if b == "IGNORE" else LABEL2ID[b] for b in bio])
    enc.pop("offset_mapping")
    enc["labels"] = all_labels
    return enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=get_settings().span_model_dir)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    data_dir = PROJECT_ROOT / "data"
    train_docs = load_jsonl(data_dir / "span_train.jsonl")
    dev_docs = load_jsonl(data_dir / "span_dev.jsonl")
    test_docs = load_markup(data_dir / "span_test.txt")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(SPAN_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    datasets = {
        name: Dataset.from_dict(dict(encode(docs, tokenizer)))
        for name, docs in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]
    }

    def to_tags(predictions, labels):
        preds = np.argmax(predictions, axis=-1)
        true_tags, pred_tags = [], []
        for p_row, l_row in zip(preds, labels):
            keep = [(p, l) for p, l in zip(p_row, l_row) if l != -100]
            true_tags.append([ID2LABEL[l] for _, l in keep])
            pred_tags.append([ID2LABEL[p] for p, _ in keep])
        return true_tags, pred_tags

    def compute_metrics(eval_pred):
        true_tags, pred_tags = to_tags(*eval_pred)
        return {"f1": entity_prf(true_tags, pred_tags)["micro"]["f1"]}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(PROJECT_ROOT / "models" / "_span_checkpoints"),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=32,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            weight_decay=0.01,
            report_to=[],
            seed=42,
        ),
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    report_lines = []
    for split in ("dev", "test"):
        out = trainer.predict(datasets[split])
        true_tags, pred_tags = to_tags(out.predictions, out.label_ids)
        for mode in ("strict", "lenient"):
            report = format_report(entity_prf(true_tags, pred_tags, mode))
            report_lines.append(f"== {split} ({mode}) ==\n{report}")
            print(f"\n== {split} ({mode}) ==\n{report}")

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "span_extractor_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved span extractor to {args.output}")


if __name__ == "__main__":
    main()
