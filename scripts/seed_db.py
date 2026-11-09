"""Load ``data/sample_products.csv`` into the database and audit each product.

    python scripts/seed_db.py [--reset]

Useful for demoing the dashboard without running the API.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor import db  # noqa: E402
from auditor.config import PROJECT_ROOT  # noqa: E402
from auditor.pipeline import audit_many  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "data" / "sample_products.csv")
    parser.add_argument("--reset", action="store_true", help="drop all tables first")
    args = parser.parse_args()

    if args.reset:
        db.Base.metadata.drop_all(db.get_engine())
    db.init_db()

    df = pd.read_csv(args.csv)
    results = audit_many(df.text.tolist())
    with db.session_scope() as s:
        for row, result in zip(df.itertuples(), results):
            product = db.create_product(s, row.title, row.seller_id, row.text)
            db.apply_audit(s, product, result)
            print(f"[{result.status:<12}] {row.title}")
    print(f"Seeded {len(df)} products into {db.get_engine().url}")


if __name__ == "__main__":
    main()
