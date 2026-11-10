# E-Commerce Compliance & Dark Pattern Auditor

[![Tests](https://github.com/iamshashankyadav/Dark-pattern-Auditor/actions/workflows/tests.yml/badge.svg)](https://github.com/iamshashankyadav/Dark-pattern-Auditor/actions/workflows/tests.yml)

An NLP pipeline that audits marketplace product copy for manipulative **dark patterns**
before they turn into regulatory problems (FTC, EU consumer-protection rules). Instead of
brittle keyword blocklists, it combines sentence embeddings, zero-shot NLI, few-shot
contrastive fine-tuning (SetFit) and token-level sequence labelling to find *what* is wrong
and *exactly where* in the text.

| Risk category | Example |
|---|---|
| Deceptive Urgency (fake countdown timers, artificial scarcity) | "Hurry, only 2 left! Offer ends in 09:59" |
| Unsubstantiated Health / Environmental Claims | "Clinically proven to melt belly fat", "zero environmental impact" |
| Hidden Subscription Traps / Obfuscated Pricing | "Free trial, then $49.99 billed monthly until cancelled" |
| Confirm-Shaming (CCPA 2023) | "No thanks, I hate saving money and prefer paying full price" |
| Drip Pricing (CCPA 2023) | "Only $19 today! Mandatory $7 handling fee and $5 service fee added at checkout" |
| Bait and Switch (CCPA 2023) | "Special promo price $9.99! *Standard $49.99 applied at payment step" |
| Safe E-Commerce Copy | "Stainless steel bottle, 750 ml, free returns within 30 days" |

Full design: [docs/compliance_auditor_blueprint.md](docs/compliance_auditor_blueprint.md)

## Architecture

```
 FastAPI  POST /products ──► asyncio queue ──► batching worker
                                                    │
        ┌───────────────────────────────────────────┘
        ▼
 Phase 1  Heuristic filter          regex lexicon  +  MiniLM sentence-embedding similarity
          (auditor/heuristics.py)   to a risk-phrase bank (catches synonym substitution)
        │ clean ───────────────────────────────────────────────► Passed
        ▼ suspicious
 Phase 2  Risk classifier           SetFit (contrastive fine-tuned mpnet) or zero-shot
          (auditor/classifier.py)   bart-large-mnli, scored per sentence then aggregated
        │ top violation < 0.50 ────────────────────────────────► Passed
        │ 0.50 ≤ score < 0.70  ────────────────────────────────► Needs Review (human queue)
        ▼ score ≥ 0.70
 Phase 3  Span extractor            DistilBERT token classifier (BIO tags) isolates the
          (auditor/extractor.py)    exact manipulative phrases, with char offsets
        ▼
 Database (PostgreSQL / SQLite) ──► Streamlit dashboard (highlighting, review, retraining data)
```

Reviewer decisions are stored in `human_feedback` and merged into SetFit training with
`--with-feedback`. That is the feedback loop for concept drift.

## Project layout

```
auditor/            core library
  taxonomy.py       risk categories, NLI hypotheses, BIO label set
  lexicon.py        regex patterns + semantic phrase bank
  heuristics.py     Phase 1 filter
  embeddings.py     sentence-transformer similarity
  classifier.py     Phase 2 (SetFit / zero-shot) + document aggregation
  extractor.py      Phase 3 (token classifier, regex fallback)
  pipeline.py       tiered orchestration and routing
  db.py             SQLAlchemy models (products, flags, human feedback)
  api.py            FastAPI service
  spans.py, metrics.py, text_utils.py
dashboard/app.py    Streamlit UI
scripts/            training, data generation, evaluation, DB seeding
data/               seed corpus, test sets, span data, demo products
sql/schema.sql      PostgreSQL DDL
tests/              pytest suite (uses stub models, runs offline)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
cp .env.example .env            # optional; defaults use SQLite
```

PostgreSQL (optional): `docker compose up -d`, then set
`DATABASE_URL=postgresql+psycopg2://auditor:auditor@localhost:5432/compliance` in `.env`.

## Train the models

```bash
python scripts/train_setfit.py                  # Phase 2 classifier  -> models/setfit
python scripts/generate_span_data.py            # synthetic BIO data  -> data/span_*.jsonl
python scripts/train_span_extractor.py          # Phase 3 extractor   -> models/span_extractor
python scripts/evaluate.py                      # benchmark           -> reports/
```

Without trained models the pipeline still works. Phase 2 falls back to zero-shot
`bart-large-mnli`, and Phase 3 falls back to regex spans.

Retrain with reviewer feedback (weekly job):

```bash
python scripts/train_setfit.py --with-feedback
```

## Run

```bash
python scripts/seed_db.py --reset               # load demo products
uvicorn auditor.api:app --reload                # API on http://localhost:8000/docs
streamlit run dashboard/app.py                  # dashboard on http://localhost:8501
```

API endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/audit` | Audit text synchronously (not stored) |
| `POST` | `/products`, `/products/batch` | Queue products for asynchronous auditing |
| `GET` | `/products/{id}`, `/products?status=&seller_id=` | Results with flags and spans |
| `GET` | `/review-queue` | Items awaiting human judgement |
| `POST` | `/products/{id}/review` | Record a reviewer's corrected label |
| `GET` | `/stats` | Status counts, open flags, repeat-offender sellers |

## Tests

```bash
pytest -q
```

## Results

Hardware: single consumer GPU (CUDA). Full reports are in [reports/](reports/).

**Phase 2: document classification**, on the held-out `data/test_set.csv` (40 hand-written
examples, 10 per class, with no overlap with the 80-example training corpus):

| System | Accuracy | Macro-F1 |
|---|---|---|
| Regex lexicon baseline | 0.900 | 0.902 |
| Zero-shot `bart-large-mnli` (no training) | 0.700 | 0.664 |
| **SetFit** (`paraphrase-mpnet-base-v2`, 20 examples/class) | **0.950** | **0.950** |

#### CCPA 2023 taxonomy extension: 4-Class vs 7-Class (Before & After)

Extending the taxonomy to India's CCPA guidelines (+3 categories: Confirm-Shaming, Drip Pricing, Bait and Switch; evaluated on 70 examples in `data/test_set.csv`, 10 per class):

| Taxonomy | Classes | Test Examples | Regex Acc / F1 | SetFit Acc / F1 | Key Additions |
|---|---|---|---|---|---|
| **Original Baseline** | 4 (Urgency, Claims, Pricing, Safe) | 40 | 0.900 / 0.902 | **0.950 / 0.950** | FTC core categories |
| **CCPA Extended** | 7 (+ Confirm-Shaming, Drip Pricing, Bait & Switch) | 70 | 0.829 / 0.840 | **0.871 / 0.868** | India CCPA 2023 compliance |

Even with almost double the label space (7 classes vs 4), few-shot SetFit achieves **87.1% accuracy** and **0.868 macro-F1** with only 20 seed examples per class. Full per-class precision and recall metrics are in [reports/classification_report.md](reports/classification_report.md).

**Full tiered pipeline** (Phase 1 → 2 → 3 with routing): 39 of 40 items are decided
automatically at **97.4 % accuracy** and 1 ambiguous item goes to human review. 7 of the 10
safe items exit at Phase 1, so they never reach a transformer.

**Phase 3: span extraction**, on the hand-annotated `data/span_test.txt` (30 documents,
27 gold spans):

| Extractor | Strict F1 | Lenient (overlap) F1 |
|---|---|---|
| Regex lexicon | 0.091 | 0.906 |
| **DistilBERT token classifier** | **0.414** | **0.915** |

Strict F1 requires exact phrase boundaries. Lenient F1 counts any same-type overlap, which is
what matters for highlighting. The token classifier finds every claim and urgency phrase in the
test set (recall 1.0). Most remaining strict errors are boundary disagreements, such as
"timer says 4 minutes left" vs the gold "the timer says 4 minutes left".

### Real-world results (Princeton "Dark Patterns at Scale" dataset)

Evaluated on 1,275 real-world dark-pattern texts crawled from ~11,000 shopping sites ([Mathur et al., 2019](https://github.com/aruneshmathur/dark-patterns-at-scale)) mapped to our taxonomy (see [docs/label_mapping.md](docs/label_mapping.md) and [reports/princeton_report.md](reports/princeton_report.md)):

| System | Accuracy | Deceptive Urgency F1 | Subscription Trap F1 | Speed |
|---|---|---|---|---|
| Regex lexicon baseline | 0.636 | 0.777 | 0.779 | 0.1 ms / item |

Key takeaways from real-world shopping copy:
- **High precision (0.97 - 1.00):** When the regex fires, it is almost always correct.
- **Recall drop on open-ended phrasing (~64%):** Real sites use long-tail synonyms (e.g. "purchased a" vs "bought", "seeing" vs "viewing", implicit scarcity like "in high demand" without numbers), highlighting why Phase 2 transformer embeddings and few-shot fine-tuning are vital.
- Detailed failure analysis and 5 representative cases are documented in [reports/princeton_report.md](reports/princeton_report.md).

### Observations and limitations

* **Zero-shot NLI struggles with health/eco claims** (recall 0.10). The NLI model reads
  "cures migraines" as a plausible product description rather than an *unproven* claim. A
  few labelled examples (SetFit) fix this, which is the motivation for Phase 2 fine-tuning.
* **The regex baseline is optimistic.** The lexicon and the test set were written by the
  same team, so the lexicon partly anticipates test phrasing. It also cannot localise spans
  (strict F1 0.09).
* **Negation is a known failure mode.** "Pay once, there are no recurring fees" is
  classified as a subscription trap with high confidence. Reviewer feedback on such cases is
  exactly what the retraining loop is for.
* **The data is small and partly synthetic.** Span training data is template-generated plus
  60 hand-annotated sentences. Dev F1 is 1.0 on synthetic data, so only the hand-written test
  sets are meaningful.
* **Thresholds are calibrated on training data.** The Phase 1 semantic threshold (0.30) was
  chosen on the seed corpus (97 % violation recall), not on the test set.
