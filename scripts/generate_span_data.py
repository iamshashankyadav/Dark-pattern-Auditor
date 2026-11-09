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


def make_document(rng: random.Random) -> dict:
    sentences = [fill(rng.choice(SAFE), rng) for _ in range(rng.randint(1, 3))]
    n_bad = rng.choices([0, 1, 2, 3], weights=[0.2, 0.45, 0.25, 0.1])[0]
    for _ in range(n_bad):
        pool = rng.choice([URGENCY, CLAIMS, PRICING])
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
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    docs = [make_document(rng) for _ in range(args.train + args.dev)]
    write_jsonl(args.out_dir / "span_train.jsonl", docs[: args.train])
    write_jsonl(args.out_dir / "span_dev.jsonl", docs[args.train :])
    print(f"Wrote {args.train} train / {args.dev} dev documents to {args.out_dir}")


if __name__ == "__main__":
    main()
