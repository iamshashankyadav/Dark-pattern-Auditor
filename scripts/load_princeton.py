"""Loader and preprocessor for the Princeton 'Dark Patterns at Scale' dataset (Mathur et al., 2019).

The public dataset consists of dark-pattern strings crawled from ~11,000 shopping websites:
https://github.com/aruneshmathur/dark-patterns-at-scale

Usage:
    python scripts/load_princeton.py
    python scripts/load_princeton.py --output data/princeton_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_URL = (
    "https://raw.githubusercontent.com/aruneshmathur/dark-patterns-at-scale/master/dark-patterns.csv"
)

# High-level mapping from Princeton Pattern Categories to Auditor Risk Categories
CATEGORY_MAPPING: dict[str, str] = {
    "Urgency": "deceptive_urgency",
    "Scarcity": "deceptive_urgency",
    "Social Proof": "deceptive_urgency",
    "Obstruction": "subscription_trap",
    "Sneaking": "subscription_trap",
    "Forced Action": "subscription_trap",
}

# Granular mapping from Princeton Pattern Types
PATTERN_TYPE_MAPPING: dict[str, str] = {
    # Urgency / Scarcity / Social Proof -> deceptive_urgency
    "Countdown Timer": "deceptive_urgency",
    "Limited-time Message": "deceptive_urgency",
    "Low-stock Message": "deceptive_urgency",
    "High-demand Message": "deceptive_urgency",
    "Activity Notification": "deceptive_urgency",
    "Testimonials of Uncertain Origin": "deceptive_urgency",
    # Sneaking / Obstruction / Forced Action -> subscription_trap
    "Hard to Cancel": "subscription_trap",
    "Hidden Subscription": "subscription_trap",
    "Hidden Costs": "subscription_trap",
    "Sneak into Basket": "subscription_trap",
    "Forced Enrollment": "subscription_trap",
    # Misdirection types: Not in baseline 4-class taxonomy (addressed in CCPA B1)
    "Confirmshaming": "misdirection",
    "Pressured Selling": "misdirection",
    "Visual Interference": "misdirection",
    "Trick Questions": "misdirection",
}


def find_or_download_dataset(path: str | Path | None = None) -> Path:
    if path:
        p = Path(path)
        if p.exists():
            return p

    candidates = [
        PROJECT_ROOT / "dark-patterns.csv",
        PROJECT_ROOT / "data" / "dark-patterns.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Download if not present
    dest = PROJECT_ROOT / "data" / "dark-patterns.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Princeton dataset to {dest}...")
    urllib.request.urlretrieve(DATASET_URL, dest)
    return dest


def load_princeton(
    csv_path: str | Path | None = None,
    include_misdirection: bool = False,
    include_safe_controls: bool = False,
) -> pd.DataFrame:
    """Load and map Princeton dataset into Auditor standard format (text, label)."""
    src_path = find_or_download_dataset(csv_path)
    df = pd.read_csv(src_path)

    # Drop missing or empty pattern strings
    df = df.dropna(subset=["Pattern String"]).copy()
    df["Pattern String"] = df["Pattern String"].astype(str).str.strip()
    df = df[df["Pattern String"].str.len() > 0]

    # Map categories
    records = []
    for _, row in df.iterrows():
        cat = str(row.get("Pattern Category", "")).strip()
        ptype = str(row.get("Pattern Type", "")).strip()
        text = row["Pattern String"]

        if cat in CATEGORY_MAPPING:
            label = CATEGORY_MAPPING[cat]
        elif cat == "Misdirection":
            if not include_misdirection:
                continue
            # When included in 4-class mapping, misdirection is categorized as out-of-scope / misdirection
            label = "misdirection"
        else:
            continue

        records.append({
            "text": text,
            "label": label,
            "princeton_category": cat,
            "princeton_type": ptype,
            "deceptive_verified": row.get("Deceptive?", "No"),
            "website_page": row.get("Website Page", ""),
        })

    out_df = pd.DataFrame(records)

    if include_safe_controls:
        # Load safe examples from test_set.csv and sample_products.csv for balanced evaluation
        safe_samples = []
        test_csv = PROJECT_ROOT / "data" / "test_set.csv"
        if test_csv.exists():
            tdf = pd.read_csv(test_csv)
            safe_rows = tdf[tdf["label"] == "safe"]
            for _, r in safe_rows.iterrows():
                safe_samples.append({
                    "text": r["text"],
                    "label": "safe",
                    "princeton_category": "Safe Control",
                    "princeton_type": "Benign Copy",
                    "deceptive_verified": "No",
                    "website_page": "Control",
                })
        if safe_samples:
            out_df = pd.concat([out_df, pd.DataFrame(safe_samples)], ignore_index=True)

    return out_df


def main():
    parser = argparse.ArgumentParser(description="Load and map Princeton dark-patterns dataset.")
    parser.add_argument("--input", default=None, help="Path to dark-patterns.csv")
    parser.add_argument("--output", default=PROJECT_ROOT / "data" / "princeton_mapped.csv",
                        type=Path, help="Output mapped CSV path")
    parser.add_argument("--include-misdirection", action="store_true",
                        help="Include Misdirection patterns (confirmshaming, trick questions)")
    parser.add_argument("--include-safe-controls", action="store_true",
                        help="Append benign product descriptions for negative control evaluation")
    args = parser.parse_args()

    df = load_princeton(
        args.input,
        include_misdirection=args.include_misdirection,
        include_safe_controls=args.include_safe_controls,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Processed {len(df)} rows. Saved to {args.output}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nPrinceton Category breakdown:")
    print(df["princeton_category"].value_counts().to_string())


if __name__ == "__main__":
    main()

