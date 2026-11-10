import pandas as pd
import pytest

from auditor.config import PROJECT_ROOT
from auditor.heuristics import screen
from auditor.lexicon import regex_hits


@pytest.mark.parametrize("text,category", [
    ("Hurry! Only 2 left in stock.", "deceptive_urgency"),
    ("Offer ends in 10 minutes", "deceptive_urgency"),
    ("Clinically proven to melt belly fat", "unsubstantiated_claims"),
    ("100% eco-friendly packaging", "unsubstantiated_claims"),
    ("Free trial, then billed monthly until you cancel", "subscription_trap"),
    ("Price excludes mandatory service fees", "subscription_trap"),
    ("No thanks, I hate saving money and prefer full price", "confirm_shaming"),
    ("Mandatory internet handling fee and booking surcharge added at checkout", "drip_pricing"),
    ("Advertised at 70% off but discount applies only if upgraded to platinum plan", "bait_and_switch"),
])
def test_regex_detects_category(text, category):
    assert category in regex_hits(text)


def test_plain_copy_is_clean():
    assert not screen("Cotton t-shirt in sizes S to XL.", use_semantic=False).suspicious


def test_regex_recall_on_seed_violations():
    """Phase 1 must let (almost) every known violation through to the classifier."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "seed_corpus.csv")
    bad = df[df.label != "safe"]
    caught = sum(screen(t, use_semantic=False).suspicious for t in bad.text)
    assert caught / len(bad) >= 0.9
