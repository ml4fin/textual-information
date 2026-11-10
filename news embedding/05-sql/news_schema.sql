-- ============================================================================
-- news_schema.sql
-- PostgreSQL schema for the "Emergence and Prediction of New Financial Risk
-- Categories" thesis project.
--
-- Reconstructed from metadata exported from the live PostgreSQL database
-- (database: research; schema: risk_emergence) on 2026-10-02.
-- This file creates STRUCTURE ONLY. It does not contain the news corpus or
-- research outputs.
--
-- IMPORTANT: the metadata export used to reconstruct this file did not include
-- the SQL expressions of four custom CHECK constraints (see README.md).
-- They are therefore documented but intentionally not guessed here.
-- ============================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS risk_emergence;

-- Sequences ------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS risk_emergence.candidate_themes_theme_id_seq AS integer START WITH 1 INCREMENT BY 1;
CREATE SEQUENCE IF NOT EXISTS risk_emergence.firm_universe_firm_id_seq AS integer START WITH 1 INCREMENT BY 1;
CREATE SEQUENCE IF NOT EXISTS risk_emergence.news_articles_article_id_seq AS bigint START WITH 1 INCREMENT BY 1;
CREATE SEQUENCE IF NOT EXISTS risk_emergence.news_firm_mentions_mention_id_seq AS bigint START WITH 1 INCREMENT BY 1;
CREATE SEQUENCE IF NOT EXISTS risk_emergence.news_indicators_indicator_id_seq AS bigint START WITH 1 INCREMENT BY 1;
CREATE SEQUENCE IF NOT EXISTS risk_emergence.news_sources_source_id_seq AS smallint START WITH 1 INCREMENT BY 1;
CREATE SEQUENCE IF NOT EXISTS risk_emergence.news_theme_annotations_annotation_id_seq AS bigint START WITH 1 INCREMENT BY 1;
CREATE SEQUENCE IF NOT EXISTS risk_emergence.news_theme_scores_score_id_seq AS bigint START WITH 1 INCREMENT BY 1;

-- Core tables ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_emergence.news_sources (
    source_id smallint NOT NULL DEFAULT nextval('risk_emergence.news_sources_source_id_seq'::regclass),
    source_code varchar(30) NOT NULL,
    source_name varchar(100) NOT NULL,
    source_domain varchar(100),
    notes text,
    CONSTRAINT news_sources_pkey PRIMARY KEY (source_id),
    CONSTRAINT news_sources_source_code_key UNIQUE (source_code)
);

CREATE TABLE IF NOT EXISTS risk_emergence.candidate_themes (
    theme_id integer NOT NULL DEFAULT nextval('risk_emergence.candidate_themes_theme_id_seq'::regclass),
    theme_name varchar(100) NOT NULL,
    theme_label varchar(200),
    description text,
    seed_words text[],
    is_active boolean DEFAULT true,
    created_at timestamptz DEFAULT now(),
    notes text,
    CONSTRAINT candidate_themes_pkey PRIMARY KEY (theme_id),
    CONSTRAINT candidate_themes_theme_name_key UNIQUE (theme_name)
);

CREATE TABLE IF NOT EXISTS risk_emergence.firm_universe (
    firm_id integer NOT NULL DEFAULT nextval('risk_emergence.firm_universe_firm_id_seq'::regclass),
    cik varchar(20) NOT NULL,
    ticker varchar(10),
    company_name varchar(200) NOT NULL,
    sector varchar(100),
    industry varchar(100),
    sic_code varchar(10),
    sp500_entry_date date,
    sp500_exit_date date,
    is_active boolean DEFAULT true,
    CONSTRAINT firm_universe_pkey PRIMARY KEY (firm_id),
    CONSTRAINT uq_firm_cik_period UNIQUE (cik, sp500_entry_date)
);

CREATE TABLE IF NOT EXISTS risk_emergence.news_articles_staging (
    title text,
    body text,
    publication_date text,
    language text,
    source text,
    sentiment text,
    is_duplicate text,
    data_type text,
    url text,
    _raw_line_number bigint,
    _source_file text
);

CREATE TABLE IF NOT EXISTS risk_emergence.news_articles (
    article_id bigint NOT NULL DEFAULT nextval('risk_emergence.news_articles_article_id_seq'::regclass),
    title text NOT NULL,
    body text NOT NULL,
    publication_date date NOT NULL,
    language varchar(10),
    source_raw varchar(100),
    sentiment_original double precision,
    is_duplicate_flag boolean DEFAULT false,
    data_type varchar(30),
    url text,
    source_id smallint,
    is_english boolean DEFAULT true,
    is_duplicate_verified boolean DEFAULT false,
    is_excluded boolean DEFAULT false,
    word_count integer,
    char_count integer,
    body_tsv tsvector,
    title_tsv tsvector,
    ingestion_batch varchar(50),
    ingested_at timestamptz DEFAULT now(),
    content_hash bytea,
    CONSTRAINT news_articles_pkey PRIMARY KEY (article_id),
    CONSTRAINT news_articles_source_id_fkey FOREIGN KEY (source_id)
        REFERENCES risk_emergence.news_sources(source_id)
);

CREATE TABLE IF NOT EXISTS risk_emergence.news_theme_annotations (
    annotation_id bigint NOT NULL DEFAULT nextval('risk_emergence.news_theme_annotations_annotation_id_seq'::regclass),
    article_id bigint NOT NULL,
    theme_id integer NOT NULL,
    human_label smallint,
    annotator varchar(50),
    sample_stratum varchar(50) NOT NULL,
    sample_wave integer NOT NULL DEFAULT 1,
    split varchar(20),
    notes text,
    annotated_at timestamp without time zone,
    CONSTRAINT news_theme_annotations_pkey PRIMARY KEY (annotation_id),
    CONSTRAINT unique_article_theme_annotation UNIQUE (article_id, theme_id, sample_wave)
    -- Live DB also contains CHECK constraint "valid_human_label".
    -- Its expression was not included in the metadata export and is not guessed.
);

CREATE TABLE IF NOT EXISTS risk_emergence.news_theme_scores (
    score_id bigint NOT NULL DEFAULT nextval('risk_emergence.news_theme_scores_score_id_seq'::regclass),
    article_id bigint NOT NULL,
    theme_id integer NOT NULL,
    score real NOT NULL,
    method varchar(30) NOT NULL,
    model_version varchar(50),
    computed_at timestamptz DEFAULT now(),
    CONSTRAINT news_theme_scores_pkey PRIMARY KEY (score_id),
    CONSTRAINT uq_article_theme_method_version UNIQUE (article_id, theme_id, method, model_version),
    CONSTRAINT news_theme_scores_article_id_fkey FOREIGN KEY (article_id)
        REFERENCES risk_emergence.news_articles(article_id),
    CONSTRAINT news_theme_scores_theme_id_fkey FOREIGN KEY (theme_id)
        REFERENCES risk_emergence.candidate_themes(theme_id)
);

CREATE TABLE IF NOT EXISTS risk_emergence.news_embeddings (
    article_id bigint NOT NULL,
    model_name varchar(50) NOT NULL,
    embedding real[] NOT NULL,
    computed_at timestamptz DEFAULT now(),
    CONSTRAINT news_embeddings_pkey PRIMARY KEY (article_id, model_name),
    CONSTRAINT news_embeddings_article_id_fkey FOREIGN KEY (article_id)
        REFERENCES risk_emergence.news_articles(article_id)
);

CREATE TABLE IF NOT EXISTS risk_emergence.news_novelty (
    article_id bigint NOT NULL,
    model_name varchar NOT NULL,
    novelty_score real NOT NULL,
    k_neighbors smallint NOT NULL,
    window_days smallint NOT NULL,
    search_method varchar NOT NULL,
    k_prime smallint,
    hnsw_m smallint,
    ef_construction smallint,
    ef_search smallint,
    computed_at timestamptz DEFAULT now(),
    CONSTRAINT news_novelty_pkey PRIMARY KEY (article_id, model_name, k_neighbors, window_days),
    CONSTRAINT news_novelty_article_fkey FOREIGN KEY (article_id)
        REFERENCES risk_emergence.news_articles(article_id)
    -- Live DB also contains CHECK constraints:
    -- news_novelty_k_check, news_novelty_score_check, news_novelty_window_check.
    -- Their expressions were not included in the metadata export and are not guessed.
);

CREATE TABLE IF NOT EXISTS risk_emergence.news_firm_mentions (
    mention_id bigint NOT NULL DEFAULT nextval('risk_emergence.news_firm_mentions_mention_id_seq'::regclass),
    article_id bigint NOT NULL,
    firm_id integer NOT NULL,
    mention_context varchar(20) NOT NULL DEFAULT 'body'::varchar,
    mention_count smallint DEFAULT 1,
    confidence real,
    CONSTRAINT news_firm_mentions_pkey PRIMARY KEY (mention_id),
    CONSTRAINT uq_article_firm UNIQUE (article_id, firm_id),
    CONSTRAINT news_firm_mentions_article_id_fkey FOREIGN KEY (article_id)
        REFERENCES risk_emergence.news_articles(article_id),
    CONSTRAINT news_firm_mentions_firm_id_fkey FOREIGN KEY (firm_id)
        REFERENCES risk_emergence.firm_universe(firm_id)
);

CREATE TABLE IF NOT EXISTS risk_emergence.news_indicators (
    indicator_id bigint NOT NULL DEFAULT nextval('risk_emergence.news_indicators_indicator_id_seq'::regclass),
    theme_id integer NOT NULL,
    period_start date NOT NULL,
    period_type varchar(10) NOT NULL DEFAULT 'month'::varchar,
    article_count integer,
    article_count_norm real,
    burst_state smallint,
    burst_intensity real,
    novelty_score real,
    semantic_drift real,
    n_firms_mentioned integer,
    n_sectors_mentioned integer,
    herfindahl_sector real,
    computed_at timestamptz DEFAULT now(),
    computation_notes text,
    CONSTRAINT news_indicators_pkey PRIMARY KEY (indicator_id),
    CONSTRAINT uq_theme_period UNIQUE (theme_id, period_start, period_type),
    CONSTRAINT news_indicators_theme_id_fkey FOREIGN KEY (theme_id)
        REFERENCES risk_emergence.candidate_themes(theme_id)
);

-- Sequence ownership ----------------------------------------------------------
ALTER SEQUENCE risk_emergence.candidate_themes_theme_id_seq OWNED BY risk_emergence.candidate_themes.theme_id;
ALTER SEQUENCE risk_emergence.firm_universe_firm_id_seq OWNED BY risk_emergence.firm_universe.firm_id;
ALTER SEQUENCE risk_emergence.news_articles_article_id_seq OWNED BY risk_emergence.news_articles.article_id;
ALTER SEQUENCE risk_emergence.news_firm_mentions_mention_id_seq OWNED BY risk_emergence.news_firm_mentions.mention_id;
ALTER SEQUENCE risk_emergence.news_indicators_indicator_id_seq OWNED BY risk_emergence.news_indicators.indicator_id;
ALTER SEQUENCE risk_emergence.news_sources_source_id_seq OWNED BY risk_emergence.news_sources.source_id;
ALTER SEQUENCE risk_emergence.news_theme_annotations_annotation_id_seq OWNED BY risk_emergence.news_theme_annotations.annotation_id;
ALTER SEQUENCE risk_emergence.news_theme_scores_score_id_seq OWNED BY risk_emergence.news_theme_scores.score_id;

-- Full-text trigger function --------------------------------------------------
CREATE OR REPLACE FUNCTION risk_emergence.news_articles_tsv_trigger()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.body_tsv  := to_tsvector('english', COALESCE(NEW.body, ''));
    NEW.title_tsv := to_tsvector('english', COALESCE(NEW.title, ''));
    NEW.word_count := array_length(string_to_array(trim(COALESCE(NEW.body, '')), ' '), 1);
    NEW.char_count := length(COALESCE(NEW.body, ''));
    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS trg_articles_tsv ON risk_emergence.news_articles;
CREATE TRIGGER trg_articles_tsv
BEFORE INSERT OR UPDATE OF body, title
ON risk_emergence.news_articles
FOR EACH ROW
EXECUTE FUNCTION risk_emergence.news_articles_tsv_trigger();

-- Non-constraint indexes ------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_articles_batch ON risk_emergence.news_articles USING btree (ingestion_batch);
CREATE INDEX IF NOT EXISTS idx_articles_body_tsv ON risk_emergence.news_articles USING gin (body_tsv);
CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON risk_emergence.news_articles USING btree (content_hash);
CREATE INDEX IF NOT EXISTS idx_articles_pubdate ON risk_emergence.news_articles USING btree (publication_date);
CREATE INDEX IF NOT EXISTS idx_articles_source_date ON risk_emergence.news_articles USING btree (source_id, publication_date);
CREATE INDEX IF NOT EXISTS idx_articles_title_tsv ON risk_emergence.news_articles USING gin (title_tsv);
CREATE INDEX IF NOT EXISTS idx_articles_valid ON risk_emergence.news_articles USING btree (publication_date, source_id)
    WHERE ((is_english = true) AND (is_duplicate_flag = false) AND (is_duplicate_verified = false) AND (is_excluded = false));
CREATE INDEX IF NOT EXISTS idx_firm_active ON risk_emergence.firm_universe USING btree (is_active) WHERE (is_active = true);
CREATE INDEX IF NOT EXISTS idx_firm_cik ON risk_emergence.firm_universe USING btree (cik);
CREATE INDEX IF NOT EXISTS idx_firm_sector ON risk_emergence.firm_universe USING btree (sector);
CREATE INDEX IF NOT EXISTS idx_firm_ticker ON risk_emergence.firm_universe USING btree (ticker);
CREATE INDEX IF NOT EXISTS idx_indicators_theme_period ON risk_emergence.news_indicators USING btree (theme_id, period_start);
CREATE INDEX IF NOT EXISTS idx_mentions_article ON risk_emergence.news_firm_mentions USING btree (article_id);
CREATE INDEX IF NOT EXISTS idx_mentions_firm ON risk_emergence.news_firm_mentions USING btree (firm_id);
CREATE INDEX IF NOT EXISTS idx_mentions_firm_article ON risk_emergence.news_firm_mentions USING btree (firm_id, article_id);
CREATE INDEX IF NOT EXISTS idx_theme_scores_article ON risk_emergence.news_theme_scores USING btree (article_id);
CREATE INDEX IF NOT EXISTS idx_theme_scores_theme ON risk_emergence.news_theme_scores USING btree (theme_id);
CREATE INDEX IF NOT EXISTS idx_theme_scores_theme_score ON risk_emergence.news_theme_scores USING btree (theme_id, score DESC);

-- Views ----------------------------------------------------------------------
CREATE OR REPLACE VIEW risk_emergence.v_valid_articles AS
SELECT a.article_id,
       a.title,
       a.body,
       a.publication_date,
       a.language,
       a.source_raw,
       a.sentiment_original,
       a.is_duplicate_flag,
       a.data_type,
       a.url,
       a.source_id,
       a.is_english,
       a.is_duplicate_verified,
       a.is_excluded,
       a.word_count,
       a.char_count,
       a.body_tsv,
       a.title_tsv,
       a.ingestion_batch,
       a.ingested_at,
       a.content_hash,
       s.source_code,
       s.source_name
FROM risk_emergence.news_articles a
LEFT JOIN risk_emergence.news_sources s ON a.source_id = s.source_id
WHERE a.is_english = true
  AND a.is_duplicate_flag = false
  AND a.is_duplicate_verified = false
  AND a.is_excluded = false;

CREATE OR REPLACE VIEW risk_emergence.v_coverage_by_source AS
SELECT s.source_code,
       s.source_name,
       date_trunc('month', a.publication_date::timestamptz)::date AS month,
       count(*) AS total_articles,
       count(*) FILTER (WHERE a.is_duplicate_flag = false
                          AND a.is_duplicate_verified = false
                          AND a.is_excluded = false) AS valid_articles
FROM risk_emergence.news_articles a
LEFT JOIN risk_emergence.news_sources s ON a.source_id = s.source_id
GROUP BY s.source_code, s.source_name, date_trunc('month', a.publication_date::timestamptz)
ORDER BY s.source_code, date_trunc('month', a.publication_date::timestamptz)::date;

CREATE OR REPLACE VIEW risk_emergence.v_gold_annotation_blind AS
SELECT a.annotation_id,
       a.article_id,
       a.theme_id,
       CASE a.theme_id
           WHEN 1 THEN 'Cybersecurity Risk'::text
           WHEN 2 THEN 'Climate / Environmental Risk'::text
           WHEN 3 THEN 'Pandemic Risk'::text
           WHEN 4 THEN 'Supply Chain Risk'::text
           WHEN 5 THEN 'AI-Related Risk'::text
           ELSE NULL::text
       END AS theme_name,
       n.title,
       n.body,
       a.human_label,
       a.annotator,
       a.notes,
       a.annotated_at
FROM risk_emergence.news_theme_annotations a
JOIN risk_emergence.news_articles n ON n.article_id = a.article_id
WHERE a.sample_wave = 1;

CREATE OR REPLACE VIEW risk_emergence.news_theme_scores_final AS
SELECT score_id, article_id, theme_id, score, method, model_version, computed_at
FROM risk_emergence.news_theme_scores
WHERE (method = 'keyword' AND model_version = 'seedwords_tsrank_v1')
   OR (method = 'weakly_supervised' AND model_version = 'weak_v3_continuous_frozen')
   OR (method = 'supervised_v3' AND model_version = 'supervised_v3_frozen');

CREATE OR REPLACE VIEW risk_emergence.v_theme_coverage AS
SELECT t.theme_name,
       date_trunc('month', a.publication_date::timestamptz)::date AS month,
       count(*) AS theme_articles,
       total.total AS total_articles,
       round(count(*)::numeric / NULLIF(total.total, 0)::numeric, 6) AS coverage_ratio
FROM risk_emergence.news_theme_scores ts
JOIN risk_emergence.news_articles a ON ts.article_id = a.article_id
JOIN risk_emergence.candidate_themes t ON ts.theme_id = t.theme_id
JOIN LATERAL (
    SELECT count(*) AS total
    FROM risk_emergence.v_valid_articles va
    WHERE date_trunc('month', va.publication_date::timestamptz)
        = date_trunc('month', a.publication_date::timestamptz)
) total ON true
WHERE ts.score >= 0.5::double precision
  AND a.is_english = true
  AND a.is_duplicate_flag = false
  AND a.is_excluded = false
GROUP BY t.theme_name, date_trunc('month', a.publication_date::timestamptz), total.total
ORDER BY t.theme_name, date_trunc('month', a.publication_date::timestamptz)::date;

COMMIT;
