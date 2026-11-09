# E-Commerce Compliance & Dark Pattern Auditor

An NLP pipeline that audits marketplace product copy for manipulative **dark patterns**
before they turn into regulatory problems (FTC, EU consumer-protection rules). Instead of
brittle keyword blocklists, it combines sentence embeddings, zero-shot NLI, few-shot
contrastive fine-tuning (SetFit) and token-level sequence labelling to find *what* is wrong
and *exactly where* in the text.

| Risk category | Example |
|---|---|
| Deceptive Urgency (fake countdown timers) | "Hurry, only 2 left! Offer ends in 09:59" |
| Unsubstantiated Health / Environmental Claims | "Clinically proven to melt belly fat", "zero environmental impact" |
| Hidden Subscription Traps / Obfuscated Pricing | "Free trial, then $49.99 billed monthly until cancelled" |
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

<!-- results -->
