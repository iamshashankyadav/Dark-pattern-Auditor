-- PostgreSQL schema (mirrors auditor/db.py). Loaded automatically by docker-compose.

CREATE TABLE IF NOT EXISTS audited_products (
    product_id      SERIAL PRIMARY KEY,
    title           VARCHAR(255) NOT NULL,
    seller_id       VARCHAR(100) NOT NULL,
    raw_text        TEXT NOT NULL,
    status          VARCHAR(50) NOT NULL DEFAULT 'Pending',
    phase_reached   INT,
    review_category VARCHAR(150),
    scores_json     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    audited_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_audited_products_seller_id ON audited_products (seller_id);
CREATE INDEX IF NOT EXISTS ix_audited_products_status ON audited_products (status);

CREATE TABLE IF NOT EXISTS compliance_flags (
    flag_id              SERIAL PRIMARY KEY,
    product_id           INT NOT NULL REFERENCES audited_products(product_id) ON DELETE CASCADE,
    violation_category   VARCHAR(150) NOT NULL,
    confidence_score     NUMERIC(5,4) NOT NULL,
    flagged_text_snippet TEXT NOT NULL,
    start_char           INT,
    end_char             INT,
    source               VARCHAR(50),
    resolved             BOOLEAN NOT NULL DEFAULT FALSE,
    flagged_at           TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_compliance_flags_product_id ON compliance_flags (product_id);

CREATE TABLE IF NOT EXISTS human_feedback (
    feedback_id        SERIAL PRIMARY KEY,
    product_id         INT NOT NULL REFERENCES audited_products(product_id) ON DELETE CASCADE,
    corrected_category VARCHAR(150) NOT NULL,
    reviewer           VARCHAR(100) NOT NULL DEFAULT 'anonymous',
    notes              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_human_feedback_product_id ON human_feedback (product_id);
