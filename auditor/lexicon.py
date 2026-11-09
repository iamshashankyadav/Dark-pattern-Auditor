"""Hand-curated risk lexicon.

* ``PATTERNS`` - regular expressions used by the Phase 1 heuristic filter and by the
  rule-based fallback span extractor. They are tuned for *recall*: false positives are
  cheap because the transformer layer re-checks them, false negatives are not.
* ``PHRASE_BANK`` - prototypical manipulative phrases per category. Their sentence
  embeddings let Phase 1 catch paraphrases that contain none of the banned keywords.
"""

import re

_URGENCY = [
    r"\bhurry\b",
    r"\bact (?:now|fast|quickly)\b",
    r"\b(?:only|just)\s+(?:\d+|a few|a couple|one|two|three|a handful)\s+(?:\w+\s+)?(?:left|remaining|remain)\b",
    r"\b\d+\s+(?:people|shoppers|others|customers)\s+(?:are\s+)?(?:looking|viewing|watching|have this)",
    r"\b(?:ends?|ending|expires?|closing)\s+(?:in|soon|tonight|today|at midnight)\b",
    r"\b(?:last|final)\s+(?:chance|call|hours?)\b",
    r"\blimited[- ]time\b",
    r"\b(?:selling|sold)\s+out\b",
    r"\balmost\s+(?:gone|sold out)\b",
    r"\bcount\s?down\b",
    r"\btimer\b",
    r"\bclock\s+is\s+ticking\b|\bticking\b",
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:minutes|mins|left)?\b",
    r"\b(?:don'?t|do not)\s+(?:miss|wait|lose)\b",
    r"\bbefore\s+(?:it'?s|its)\s+(?:gone|too late)\b",
    r"\b(?:never|won'?t)\s+(?:be\s+)?(?:restock|this low|again)\w*\b",
    r"\b(?:next|within the next)\s+\d+\s+(?:minutes|hours|mins)\b",
    r"\blow stock\b|\bstock\s+is\s+(?:almost|running)\b",
    r"\bonly today\b|\btoday only\b",
    r"\bjust bought\b|\bsold in the last\b",
]

_CLAIMS = [
    r"\bcures?\b|\bcured\b|\bheals?\b",
    r"\bclinically\s+(?:proven|tested)\b|\bscientifically\s+(?:proven|formulated)\b",
    r"\bproven\s+to\b",
    r"\bdetox\w*\b|\bflush(?:es)?\s+out\b|\btoxins?\b",
    r"\b(?:prevents?|reverses?|eliminates?)\s+(?:all\s+)?(?:disease|illness|aging|infections?|wrinkles|cancer)",
    r"\bboosts?\s+(?:your\s+)?(?:immun\w*|iq|memory|metabolism)",
    r"\b(?:lose|melts?|burns?)\s+(?:\d+\s*(?:kg|lbs?|pounds)|belly fat|fat)\b|\bfat[- ]burning\b|\bweight loss\b",
    r"\bmiracle\b|\bdoctors hate\b",
    r"\beco[- ]?friendly\b|\bcarbon[- ]neutral\b|\bzero\s+(?:environmental\s+impact|emissions|waste|footprint)\b",
    r"\bsaves?\s+the\s+(?:planet|oceans?|earth)\b|\bplanet[- ]safe\b|\bharmless to nature\b",
    r"\b(?:100%\s+)?(?:natural|chemical[- ]free|non[- ]toxic)\b",
    r"\bgreen(?:est)?\b|\bsustainable\b|\bbiodegradable\b",
    r"\bkills?\s+\d+(?:\.\d+)?%|\bkills?\s+(?:germs|viruses|coronavirus|bacteria)",
    r"\bguarantee[sd]?\s+(?:to|permanent)\b",
    r"\b(?:anxiety|depression|diabetes|arthritis|cancer|migraines?|insomnia|ulcers?)\b",
    r"\breplace\s+your\s+(?:prescription|medication|medicine)\b",
]

_PRICING = [
    r"\bfree\s+trial\b|\btry\s+it\s+free\b|\bfree\s+(?:sample|today)\b",
    r"\bauto(?:matically)?[- ]?(?:renews?|renewal|charged?|shipments?|upgrades?)\b",
    r"\brecurring\b|\bbilled\b|\bmonthly\b|\bper month\b|\beach month\b|\bquarterly\b",
    r"\bsubscri\w+\b|\bmembership\b|\benrol+\w*\b|\bclub\b",
    r"\b(?:service|handling|convenience|booking|platform|resort|activation|membership|cancellation)\s+fees?\b",
    r"\bsurcharges?\b|\bhidden fees?\b|\bfees? (?:apply|are added)\b",
    r"\bexcludes?\b|\bdoes not include\b|\bnot included\b|\bsold separately\b",
    r"\bpre[- ]?(?:selected|added|checked)\b|\badded to your (?:order|cart) by default\b",
    r"\b(?:until|unless)\s+(?:you\s+)?(?:cancel\w*|opt out|call)\b",
    r"\bto cancel\b|\bcancellation requires\b",
    r"\bcommitment\b|\bcontract\b|\btermination fees?\b",
    r"\bthen\b.{0,40}\b(?:full price|renews?|charged)\b|\bafterwards\b|\bthereafter\b",
    r"\$\s?\d+(?:\.\d{2})?\s*\*",
]

PATTERNS: dict[str, list[re.Pattern]] = {
    "deceptive_urgency": [re.compile(p, re.IGNORECASE) for p in _URGENCY],
    "unsubstantiated_claims": [re.compile(p, re.IGNORECASE) for p in _CLAIMS],
    "subscription_trap": [re.compile(p, re.IGNORECASE) for p in _PRICING],
}

PHRASE_BANK: dict[str, list[str]] = {
    "deceptive_urgency": [
        "only a few items left in stock, buy now",
        "this offer expires in minutes",
        "hurry before the deal is gone",
        "many people are viewing this product right now",
        "last chance to get this price",
        "the countdown timer is running out",
        "will never be available at this price again",
    ],
    "unsubstantiated_claims": [
        "cures disease naturally",
        "clinically proven miracle results",
        "lose weight fast without effort",
        "completely eco-friendly with zero impact on the planet",
        "detoxifies your body and removes toxins",
        "prevents illness and boosts immunity",
        "reverses aging and heals the body",
    ],
    "subscription_trap": [
        "free trial that turns into a paid monthly subscription",
        "automatically renews and charges your card",
        "extra fees are added at checkout",
        "price does not include mandatory charges",
        "difficult cancellation process",
        "recurring billing until you cancel",
        "add-on included in your order by default",
    ],
}


def regex_hits(text: str) -> dict[str, list[tuple[int, int, str]]]:
    """Return ``{category: [(start, end, matched_text), ...]}`` for every pattern hit."""
    hits: dict[str, list[tuple[int, int, str]]] = {}
    for category, patterns in PATTERNS.items():
        found = []
        for pattern in patterns:
            for m in pattern.finditer(text):
                found.append((m.start(), m.end(), m.group(0)))
        if found:
            hits[category] = sorted(set(found))
    return hits
