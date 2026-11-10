"""Compliance risk taxonomy shared by every phase of the pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskCategory:
    key: str            # stable identifier stored in the database
    name: str           # human readable name from the blueprint
    span_tag: str       # BIO tag used by the token classifier ("" for safe copy)
    hypothesis: str     # natural-language description fed to the NLI zero-shot model


URGENCY = RiskCategory(
    key="deceptive_urgency",
    name="Deceptive Urgency (Fake Countdown Timers)",
    span_tag="URGENCY",
    hypothesis="pressures the buyer with fake scarcity, countdown timers or a deadline to buy now",
)
CLAIMS = RiskCategory(
    key="unsubstantiated_claims",
    name="Unsubstantiated Health or Environmental Claims",
    span_tag="CLAIM",
    hypothesis="makes an unproven medical, health, or environmental claim",
)
PRICING = RiskCategory(
    key="subscription_trap",
    name="Hidden Subscription Traps or Obfuscated Pricing",
    span_tag="PRICING",
    hypothesis="hides a recurring subscription, automatic renewal or extra fees in the price",
)
SHAMING = RiskCategory(
    key="confirm_shaming",
    name="Confirm-shaming (Guilt-inducing Opt-outs)",
    span_tag="SHAMING",
    hypothesis="uses guilt, shame, or emotional manipulation to discourage opting out",
)
DRIP = RiskCategory(
    key="drip_pricing",
    name="Drip Pricing (Hidden Incremental Charges)",
    span_tag="DRIP",
    hypothesis="conceals mandatory fees, handling surcharges, or taxes revealed incrementally at checkout",
)
BAIT = RiskCategory(
    key="bait_and_switch",
    name="Bait and Switch (Misleading Offer or Terms)",
    span_tag="BAIT",
    hypothesis="advertises an attractive offer that is unavailable, restricted, or replaced with higher charges",
)
SAFE = RiskCategory(
    key="safe",
    name="Safe E-Commerce Copy",
    span_tag="",
    hypothesis="is an honest, factual description of the product",
)

CATEGORIES: list[RiskCategory] = [URGENCY, CLAIMS, PRICING, SHAMING, DRIP, BAIT, SAFE]
VIOLATIONS: list[RiskCategory] = [URGENCY, CLAIMS, PRICING, SHAMING, DRIP, BAIT]
BY_KEY: dict[str, RiskCategory] = {c.key: c for c in CATEGORIES}
BY_TAG: dict[str, RiskCategory] = {c.span_tag: c for c in VIOLATIONS}
LABEL_KEYS: list[str] = [c.key for c in CATEGORIES]

# BIO label set for token classification (Phase 3).
SPAN_LABELS: list[str] = ["O"] + [f"{p}-{c.span_tag}" for c in VIOLATIONS for p in ("B", "I")]
