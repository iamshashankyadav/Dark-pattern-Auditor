# Project Blueprint: Autonomous E-Commerce Compliance & "Dark Pattern" Auditor

## 1. Executive Summary & Core NLP Alignment
This project is an automated content auditing engine designed to protect e-commerce platforms from regulatory fines (e.g., FTC, EU Consumer Protection). Instead of relying on rigid, easily bypassed keyword matching, this system leverages advanced **Natural Language Processing (NLP)** to identify complex, nuanced, and structurally manipulative linguistic tactics used by third-party sellers. 

### Core NLP Capabilities Utilized:
*   **Zero-Shot Sequence Classification:** Dynamically categorizes unseen product descriptions against shifting legal risk definitions.
*   **Sentence Embeddings & Semantic Similarity:** Detects deceptive copywriting strategies when sellers substitute banned keywords with semantic synonyms.
*   **Token-Level Classification (NER Architecture):** Fine-grained sequence labeling to isolate precise manipulative phrases within large blocks of marketing copy.

---

## 2. System Architecture Blueprint
The application is structured into a modular, decoupled processing pipeline to ensure high throughput and low execution costs.

```
[ Data Ingestion API ] 
         │
         ▼
┌─────────────────────────────────┐
│     Phase 1: Regex Filter       │ ──(Safe Copy)──> [ Database: Passed ]
└─────────────────────────────────┘
         │ (Suspicious Copy)
         ▼
┌─────────────────────────────────┐
│  Phase 2: Zero-Shot Classifier   │ ──(Confidence < 0.70)──> [ Human Evaluation Queue ]
└─────────────────────────────────┘
         │ (High Confidence Violations)
         ▼
┌─────────────────────────────────┐
│ Phase 3: Token Classification   │ ──(Highlight Triggers)
└─────────────────────────────────┘
         │
         ▼
[ Database: Flagged Table ] ──> [ Streamlit Web Dashboard UI ]
```

### Components Matrix
| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Ingestion Pipeline** | FastAPI / Python | High performance, async capacity to ingest marketplace updates. |
| **Heuristics Layer** | Python `re` module | Drops computational load by filtering out instantly obvious text. |
| **Broad Classifier** | `facebook/bart-large-mnli` | Industry-standard NLI engine for out-of-the-box accuracy without training data. |
| **Extraction Engine** | `SetFit` / Custom SpaCy NER | Extracts exact spans of text containing compliance traps. |
| **Data Warehouse** | PostgreSQL | Relational structure to map items, specific flag logs, and severity thresholds. |
| **Frontend UI** | Streamlit | Rapid prototyping for compliance operations teams to review flags. |

---

## 3. Step-by-Step Implementation Strategy

### Phase 1: Ingestion & Broad Classification Pipeline
*   **Objective:** Ingest raw text and categorize it under distinct compliance risk definitions.
*   **Execution:** Deploy an asynchronous worker queue that parses incoming text feeds through the Zero-Shot Classification model.
*   **Taxonomy Classes Managed:**
    1.  `Deceptive Urgency (Fake Countdown Timers)`
    2.  `Unsubstantiated Health or Environmental Claims`
    3.  `Hidden Subscription Traps or Obfuscated Pricing`
    4.  `Safe E-Commerce Copy`

### Phase 2: Building the SetFit Fine-Tuning Module
*   **Objective:** Optimize categorization accuracy while slashing GPU runtime costs by switching to Sentence Transformers.
*   **Execution:** Collect a baseline corpus of 20 sample sentences per category. Fine-tune a lightweight model using the Contrastive Loss function. This allows the system to reach deep-learning scale precision on an ordinary CPU runtime.

### Phase 3: Database Design (PostgreSQL Schema)
The storage layer isolates items and tracks recurring historical violations from specific vendor accounts.

```sql
CREATE TABLE audited_products (
    product_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    seller_id VARCHAR(100) NOT NULL,
    raw_text TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'Pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE compliance_flags (
    flag_id SERIAL PRIMARY KEY,
    product_id INT REFERENCES audited_products(product_id),
    violation_category VARCHAR(150) NOT NULL,
    confidence_score NUMERIC(5,2) NOT NULL,
    flagged_text_snippet TEXT NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. Production Engineering & Scalability Challenges
1.  **Latent Inference Bottlenecks:** Transformer models are heavy. Running millions of reviews directly through a large model creates massive queues.
    *   *Solution:* Implement a tiered workflow. Run a high-velocity, low-cost heuristic checker first. Only pass items that trigger semantic thresholds to the heavy transformer pool.
2.  **Concept Drift:** Creative copywriters constantly invent new ways to manipulate text to bypass current regulatory flags.
    *   *Solution:* Setup a continuous feedback loop. Flag items with intermediate confidence values (e.g., 55% - 70%) to a human compliance team. Store their manual corrections to retrain the SetFit architecture weekly.
