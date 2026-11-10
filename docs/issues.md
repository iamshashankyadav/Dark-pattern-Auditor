# Project issues

Three workstreams, so each teammate can work in parallel without merge conflicts.
Setup for everyone: clone the repo, follow the README "Setup" section, run `pytest -q`
(should be green) before starting. Work on a branch (`feature/<issue-number>-short-name`),
open a pull request, and commit from your own GitHub account.

| Workstream | Focus | Main files |
|---|---|---|
| **A: Data and evaluation** | Real-world data, benchmarks, experiments | `scripts/evaluate.py`, `data/` |
| **B: Taxonomy and models** | CCPA categories, Hinglish, negation | `auditor/taxonomy.py`, `auditor/lexicon.py`, `data/` |
| **C: Engineering and delivery** | CI, Docker, demo, report | `.github/`, `Dockerfile`, `docs/` |

---

## Workstream A: Data and evaluation

### A1. Evaluate on the Princeton "Dark Patterns at Scale" dataset
**Labels:** `data`, `evaluation`, `priority: high`

Our test sets are small and written by us. Mathur et al. (2019) published a public dataset of
dark-pattern texts crawled from about 11,000 shopping sites.

**Tasks**
- [ ] Download the dataset (search "dark-patterns-at-scale github Princeton") and add a loader script `scripts/load_princeton.py`
- [ ] Map their labels (Urgency, Scarcity, Social Proof, Misdirection, etc.) onto our four categories; document the mapping in `docs/label_mapping.md`
- [ ] Run our current models on it **without retraining** and report accuracy and macro-F1 per category
- [ ] Add a `princeton` option to `scripts/evaluate.py` and save results to `reports/`

**Acceptance criteria**
- Results table in `reports/` and a short "real-world results" section in the README
- Includes at least 5 examples where we fail, with an explanation each

---

### A2. Learning curve: how many examples does SetFit need?
**Labels:** `experiment`, `report`

**Tasks**
- [ ] Train SetFit with 2, 5, 10, 15 and 20 examples per class (use a fixed seed; repeat with 3 seeds)
- [ ] Evaluate each on `data/test_set.csv`
- [ ] Plot accuracy vs examples per class (mean and spread), including the zero-shot and regex lines as horizontal references
- [ ] Save script as `scripts/learning_curve.py`, figure as `reports/learning_curve.png`

**Acceptance criteria:** reproducible with one command; figure ready to paste into the report.

---

### A3. Ablation study of the three-phase design
**Labels:** `experiment`, `report`

Does each phase earn its place?

**Tasks**
- [ ] Compare: (a) classifier only, (b) regex + classifier, (c) regex + semantic + classifier (current)
- [ ] For each, measure accuracy, number of items reaching the transformer, and time per item
- [ ] Sweep `FLAG_THRESHOLD` and `REVIEW_THRESHOLD` and plot the share of items sent to human review against accuracy on the rest
- [ ] Write findings (what helped, what did not) in `reports/ablation.md`

**Acceptance criteria:** a table and at least one chart; conclusions stated honestly even if a phase turns out not to help.

---

### A4. Grow the test set and measure annotation agreement
**Labels:** `data`

40 test examples is too few for reliable numbers.

**Tasks**
- [ ] Two people independently label 150 new product descriptions (mix of safe and violating, include multi-violation and tricky cases)
- [ ] Compute Cohen's kappa between the two annotators
- [ ] Resolve disagreements and add the result as `data/test_set_v2.csv`
- [ ] Re-run `scripts/evaluate.py` on the new set and update the README table

**Acceptance criteria:** at least 150 items, 30+ per class, kappa reported.

---

## Workstream B: Taxonomy and models

### B1. Extend categories to India's CCPA dark-pattern guidelines
**Labels:** `nlp`, `taxonomy`, `priority: high`

India's CCPA (2023) lists 13 dark patterns (false urgency, basket sneaking, confirm-shaming,
forced action, subscription trap, interface interference, bait and switch, drip pricing,
disguised advertisement, nagging, trick wording, SaaS billing, rogue malware).

**Tasks**
- [ ] Pick the patterns that are detectable from text (at least 3 new ones, e.g. confirm-shaming, drip pricing, bait and switch)
- [ ] Add them in `auditor/taxonomy.py` (key, name, BIO tag, NLI hypothesis)
- [ ] Add regex patterns and phrase-bank entries in `auditor/lexicon.py`
- [ ] Add 20 labelled examples per new category to `data/seed_corpus.csv`, plus 10 per category to `data/test_set.csv`
- [ ] Add annotated sentences to `data/span_seed.txt`
- [ ] Retrain (`scripts/train_setfit.py`, `scripts/generate_span_data.py`, `scripts/train_span_extractor.py`) and update `tests/`
- [ ] Update dashboard colours in `dashboard/app.py`

**Acceptance criteria:** all tests pass; new categories appear in the dashboard; before/after results table.

---

### B2. Fix negation false positives
**Labels:** `nlp`, `bug`

Known failure: "Pay once, there are no recurring fees" is flagged as a subscription trap with high confidence.

**Tasks**
- [ ] Collect 30+ negated and reassuring examples ("no hidden fees", "cancel anytime", "not clinically tested", "does not auto-renew")
- [ ] Add them as `safe` examples to `data/seed_corpus.csv` and as hard negatives in `scripts/generate_span_data.py`
- [ ] Optionally add a negation check in `auditor/heuristics.py` or `auditor/extractor.py`
- [ ] Retrain and compare before and after on a dedicated negation test set, `data/negation_test.csv`

**Acceptance criteria:** the example above is classified as safe, and overall test accuracy does not drop.

---

### B3. Hinglish and Hindi support
**Labels:** `nlp`, `multilingual`, `stretch`

Indian marketplaces mix Hindi and English ("Jaldi karo, sirf 2 bache hain!").

**Tasks**
- [ ] Write 15+ Hinglish examples per category (Roman script) and 40 for a test set
- [ ] Measure how the current model performs on them (baseline)
- [ ] Try a multilingual base model (e.g. `paraphrase-multilingual-mpnet-base-v2`) in `scripts/train_setfit.py` via `--base-model`
- [ ] Report accuracy for the English-only vs multilingual model on both English and Hinglish test sets

**Acceptance criteria:** results table and a recommendation on which model to ship.

---

### B4. Improve span boundaries
**Labels:** `nlp`, `evaluation`

Exact-boundary F1 is 0.41 against 0.92 overlap F1: the model finds the right phrase but the edges differ.

**Tasks**
- [ ] Review the failures in `reports/classification_report.md` and write down the annotation rules (e.g. include leading "the"? include the trailing number?) in `docs/annotation_guide.md`
- [ ] Re-annotate `data/span_test.txt` and `data/span_seed.txt` consistently with the guide
- [ ] Try a larger base model (e.g. `roberta-base`) with `--base-model`, and compare
- [ ] Report strict and lenient F1 before and after

**Acceptance criteria:** annotation guide committed; strict F1 improves or a clear explanation of why not.

---

## Workstream C: Engineering and delivery

### C1. GitHub Actions CI
**Labels:** `devops`, `priority: high`

**Tasks**
- [ ] Add `.github/workflows/tests.yml` that installs dependencies (CPU torch) and runs `pytest -q` on every push and pull request
- [ ] Cache pip downloads
- [ ] Add a status badge to the top of the README

**Acceptance criteria:** green check on the main branch. Tests use stub models, so no model downloads are required.

---

### C2. Dockerfile and one-command run
**Labels:** `devops`

**Tasks**
- [ ] Add a `Dockerfile` (Python 3.11 slim, CPU) for the API and dashboard
- [ ] Extend `docker-compose.yml` with `api`, `dashboard` and `postgres` services using `DATABASE_URL`
- [ ] Verify the Postgres schema (`sql/schema.sql`) works with the app end to end
- [ ] Document `docker compose up` in the README

**Acceptance criteria:** a fresh clone runs with one command; dashboard shows data.

---

### C3. Public demo on Hugging Face Spaces
**Labels:** `deployment`, `stretch`

**Tasks**
- [ ] Push trained models to a Hugging Face model repo (they are not stored in git because of size)
- [ ] Make `auditor/config.py` load models from the Hub when local folders are missing
- [ ] Create a Space (Streamlit) showing the **Live audit** page only
- [ ] Add the link to the README

**Acceptance criteria:** anyone can open the link and audit text without installing anything.

---

### C4. Dashboard improvements
**Labels:** `frontend`

**Tasks**
- [ ] Add CSV upload on the dashboard: upload a file of products and audit them in bulk
- [ ] Add "Download flagged items as CSV"
- [ ] Add a per-seller page with violation history over time
- [ ] Extend `tests/test_dashboard.py` for each new feature

**Acceptance criteria:** tests pass; features visible in the dashboard.

---

### C5. Project report and presentation
**Labels:** `documentation`, `report`

Starts once A1 to A3 and B1 have results.

**Tasks**
- [ ] Report: abstract, introduction and motivation (CCPA/FTC), related work, method (three phases), experiments, error analysis, limitations, conclusion
- [ ] 10 to 12 slide deck with the architecture diagram, a live demo section and results charts
- [ ] Contributions page stating who did which workstream

**Acceptance criteria:** report and slides committed under `docs/`; every number traceable to a file in `reports/`.

---

## Suggested order

1. **Week 1:** A1, B1, C1 (each person's highest-priority item; independent of each other)
2. **Week 2:** A2, A3, B2, C2
3. **Week 3:** A4, B3, B4, C4
4. **Final week:** C3 (demo), C5 (report and slides), and fill the results into the README
