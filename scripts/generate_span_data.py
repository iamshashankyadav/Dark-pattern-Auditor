"""Generate weakly-supervised span-annotation data for the Phase 3 token classifier.

Real annotated dark-pattern corpora are scarce, so training documents are composed from
templates: honest product sentences are mixed with manipulative sentences whose trigger
phrase is annotated. Slots ({n}, {price}, ...) are filled randomly to add lexical variety.
The hand-annotated ``data/span_test.txt`` uses different wording and is the real benchmark.

    python scripts/generate_span_data.py --train 2000 --dev 300
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor.spans import parse_markup  # noqa: E402

PRODUCTS = [
    "wireless earbuds", "air fryer", "yoga mat", "face serum", "protein powder", "phone case",
    "running shoes", "coffee maker", "desk lamp", "water bottle", "vitamin gummies", "backpack",
    "smart watch", "herbal tea", "mattress", "bamboo toothbrush", "hair oil", "laptop stand",
    "streaming service", "meal kit", "skincare set", "gaming mouse", "travel mug", "blender",
]

SAFE = [
    "This {p} is available in three colours.",
    "The {p} ships within {n} business days.",
    "Free returns within 30 days of delivery.",
    "Each {p} comes with a {n}-month warranty.",
    "Dimensions: {n} x {n} x {n} cm.",
    "Rated {r} out of 5 by verified buyers.",
    "The {p} is made from durable materials.",
    "Package includes the {p} and a user manual.",
    "Customer support is available seven days a week.",
    "Our {p} is designed for everyday use.",
    "Price includes all taxes and delivery.",
    "Compatible with most standard accessories.",
    "Weighs only {n}00 grams.",
    "Hand wash recommended to keep it like new.",
    "Batteries are included in the box.",
    "Assembly takes about {n} minutes with the included tools.",
    # Hard negatives: honest copy that mentions prices, fees, time or health-adjacent words.
    "Pay once and keep it, there are no recurring fees.",
    "One-time purchase, no subscription required.",
    "Battery lasts up to {n} hours per charge.",
    "Delivery charges are shown before you pay.",
    "Order before {n} pm for same-day dispatch.",
    "Cancel anytime online from your account page in two clicks.",
    "The monthly plan costs ${price} and you can cancel at any time.",
    "Made with natural cotton fibres.",
    "Contains {n}0 mg of vitamin C per serving.",
    "Packaging is made from {n}0% recycled cardboard.",
    "Charges in {n} hours with the included adapter.",
    "Sale prices are valid until stock of the seasonal range is cleared.",
    "Tested by an independent lab for lead and BPA.",
    "Seats from ${small}0, taxes included in the displayed price.",
    "Free shipping on orders over ${price}.",
    "Lasts about {n} months with daily use.",
]

URGENCY = [
    "Hurry, [[URGENCY|only {n} left in stock]]!",
    "[[URGENCY|Offer ends in {n} minutes]], so grab your {p} today.",
    "[[URGENCY|{nn} people are viewing this right now]].",
    "[[URGENCY|Last chance]] to get the {p} at this price.",
    "[[URGENCY|This deal expires at midnight]].",
    "Get yours before [[URGENCY|the timer runs out]].",
    "[[URGENCY|Selling out fast]] - order now.",
    "[[URGENCY|Price goes up in {n} hours]].",
    "[[URGENCY|Only today]]: save {pct}% on the {p}.",
    "[[URGENCY|Don't miss out]], stock is limited.",
    "[[URGENCY|Just {n} units remaining]] at this price.",
    "[[URGENCY|Act now]] before the discount disappears.",
    "The {p} is [[URGENCY|almost sold out]].",
    "[[URGENCY|{nn} sold in the last hour]].",
    "[[URGENCY|Your cart is reserved for {n} minutes]].",
    "[[URGENCY|We will never restock this item]].",
    "Flash sale [[URGENCY|ending tonight]].",
    "[[URGENCY|Countdown: 0{n}:{nn}:{nn} left]] on this offer.",
    "[[URGENCY|Final hours]] to claim the {p}.",
    "[[URGENCY|Someone just bought this]], only a few remain.",
]

CLAIMS = [
    "This {p} [[CLAIM|cures {disease}]].",
    "[[CLAIM|Clinically proven to {effect}]].",
    "The {p} is [[CLAIM|100% eco-friendly with zero environmental impact]].",
    "[[CLAIM|Detoxifies your body]] in just {n} days.",
    "[[CLAIM|Boosts your immune system]] naturally.",
    "Our {p} [[CLAIM|prevents all infections]].",
    "[[CLAIM|Lose {n} kg in a week]] with the {p}.",
    "[[CLAIM|Doctors hate this miracle formula]].",
    "[[CLAIM|Completely carbon neutral]] from start to finish.",
    "The {p} [[CLAIM|reverses aging overnight]].",
    "Every {p} [[CLAIM|saves the planet]].",
    "[[CLAIM|Scientifically proven to {effect}]].",
    "[[CLAIM|Kills 99.9% of viruses]] on contact.",
    "[[CLAIM|Heals {disease} without medication]].",
    "Made with [[CLAIM|chemical-free, non-toxic ingredients that protect your health]].",
    "[[CLAIM|Guaranteed to {effect}]].",
    "[[CLAIM|Replaces your prescription medicine]].",
    "The {p} is [[CLAIM|totally harmless to nature]].",
]

PRICING = [
    "[[PRICING|Free trial, then ${price} per month until cancelled]].",
    "Only ${small} today, [[PRICING|then it automatically renews at full price]].",
    "[[PRICING|Price excludes mandatory service fees added at checkout]].",
    "[[PRICING|Your card will be charged every 30 days]] unless you opt out.",
    "[[PRICING|Membership fee applies after {n} days]].",
    "[[PRICING|Cancellation requires a phone call during business hours]].",
    "[[PRICING|A protection plan is added to your order by default]].",
    "[[PRICING|Auto-shipments continue indefinitely]].",
    "Total [[PRICING|does not include a non-refundable convenience fee]].",
    "[[PRICING|Trial converts to an annual plan billed upfront]].",
    "[[PRICING|Requires a {n}-month commitment]].",
    "[[PRICING|Recurring charges of ${price} apply thereafter]].",
    "Ordering [[PRICING|enrolls you in our VIP club]].",
    "[[PRICING|Booking and handling surcharges are charged at pickup]].",
    "Starting at ${small}* [[PRICING|(*per month with a 24-month contract)]].",
    "[[PRICING|Monthly refills are pre-selected at checkout]].",
    "[[PRICING|After the promo period your rate increases]].",
]

DISEASES = ["diabetes", "arthritis", "anxiety", "insomnia", "acne", "hair loss", "back pain", "migraines"]
EFFECTS = ["burn belly fat", "regrow hair", "double your energy", "remove wrinkles", "cure fatigue",
           "boost memory", "fix joint pain"]


def fill(template: str, rng: random.Random) -> str:
    out = template
    for slot, fn in {
        "{p}": lambda: rng.choice(PRODUCTS),
        "{nn}": lambda: str(rng.randint(10, 59)),
        "{n}": lambda: str(rng.randint(2, 9)),
        "{r}": lambda: f"{rng.uniform(3.5, 4.9):.1f}",
        "{pct}": lambda: str(rng.choice([20, 30, 40, 50, 60, 70])),
        "{price}": lambda: f"{rng.randint(9, 79)}.99",
        "{small}": lambda: str(rng.choice([1, 2, 5])),
        "{disease}": lambda: rng.choice(DISEASES),
        "{effect}": lambda: rng.choice(EFFECTS),
    }.items():
        while slot in out:
            out = out.replace(slot, fn(), 1)
    return out


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_seed_fragments() -> tuple[list[str], list[str]]:
    """Hand-annotated seed-corpus violations and its safe sentences (real, varied wording)."""
    import csv

    bad = [l.strip() for l in (DATA_DIR / "span_seed.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    with (DATA_DIR / "seed_corpus.csv").open(encoding="utf-8") as f:
        safe = [row["text"] for row in csv.DictReader(f) if row["label"] == "safe"]
    return bad, safe


SEED_BAD, SEED_SAFE = load_seed_fragments()


def make_document(rng: random.Random) -> dict:
    safe_pool = SAFE + SEED_SAFE
    sentences = [fill(rng.choice(safe_pool), rng) for _ in range(rng.randint(1, 3))]
    n_bad = rng.choices([0, 1, 2, 3], weights=[0.2, 0.45, 0.25, 0.1])[0]
    for _ in range(n_bad):
        pool = rng.choice([URGENCY, CLAIMS, PRICING, SEED_BAD])
        sentences.append(fill(rng.choice(pool), rng))
    rng.shuffle(sentences)
    return parse_markup(" ".join(sentences))


def write_jsonl(path: Path, docs: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--dev", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    docs = [make_document(rng) for _ in range(args.train + args.dev)]
    write_jsonl(args.out_dir / "span_train.jsonl", docs[: args.train])
    write_jsonl(args.out_dir / "span_dev.jsonl", docs[args.train :])
    print(f"Wrote {args.train} train / {args.dev} dev documents to {args.out_dir}")


if __name__ == "__main__":
    main()
