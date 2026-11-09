"""Phase 2 fine-tuning: train a SetFit classifier on the seed corpus.

SetFit fine-tunes a sentence transformer with a contrastive objective on pairs of
same-label / different-label sentences, then fits a logistic-regression head on the
resulting embeddings. 20 examples per class are enough to beat zero-shot NLI.

Human reviewer corrections stored in the database (``human_feedback`` table) are merged
into the training data with ``--with-feedback``; this is the weekly retraining loop that
counters concept drift.

    python scripts/train_setfit.py [--with-feedback] [--epochs 1]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor.config import PROJECT_ROOT, get_settings  # noqa: E402
from auditor.taxonomy import LABEL_KEYS  # noqa: E402


def load_training_frame(with_feedback: bool) -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "data" / "seed_corpus.csv")
    if with_feedback:
        from auditor.db import fetch_feedback_examples

        fb = pd.DataFrame(fetch_feedback_examples(), columns=["text", "label"])
        print(f"Adding {len(fb)} human-reviewed examples from the database")
        df = pd.concat([df, fb], ignore_index=True)
    unknown = set(df.label) - set(LABEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown labels in training data: {unknown}")
    return df.drop_duplicates("text")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="sentence-transformers/paraphrase-mpnet-base-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20, help="contrastive pairs per example")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--with-feedback", action="store_true")
    parser.add_argument("--output", type=Path, default=get_settings().setfit_model_dir)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    df = load_training_frame(args.with_feedback)
    df["label"] = df.label.map(LABEL_KEYS.index)
    print(df.label.map(lambda i: LABEL_KEYS[i]).value_counts().to_string())

    train_ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    model = SetFitModel.from_pretrained(args.base_model, labels=LABEL_KEYS)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            num_iterations=args.iterations,
            seed=args.seed,
            report_to="none",  # don't auto-enable wandb/tensorboard if installed
            output_dir=str(PROJECT_ROOT / "models" / "_setfit_checkpoints"),
            save_strategy="no",
        ),
        train_dataset=train_ds,
    )
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    print(f"Saved SetFit model to {args.output}")


if __name__ == "__main__":
    main()
