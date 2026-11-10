# PostgreSQL database schema

This folder documents the PostgreSQL structure used by the thesis project **Emergence and Prediction of New Financial Risk Categories**.

## Environment

- DBMS: PostgreSQL 18
- Database used during development: `research`
- Project schema: `risk_emergence`
- Corpus period: 2014–2023
- The database contents are **not** distributed through GitHub because of their size. This folder contains the database structure only.

## Recreating the structure

Run `news_schema.sql` on a PostgreSQL database before running the ingestion and analysis scripts. The SQL file creates the `risk_emergence` schema, tables, sequences, constraints available from the metadata export, indexes, views, the full-text trigger function, and its trigger.

Example with `psql`:

```bash
psql -d research -f 05-sql/news_schema.sql
```

The database name itself is not hard-coded in the SQL file. Therefore, the same schema can be created in another PostgreSQL database if required.

## Main objects

| Object | Role in the pipeline |
|---|---|
| `news_articles_staging` | Raw landing table used during CSV ingestion. Original fields are kept as text together with `_raw_line_number` and `_source_file` for traceability. |
| `news_sources` | Newspaper/source reference table. |
| `news_articles` | Main cleaned and normalized article table. It also stores validation/deduplication flags, full-text vectors, text-length fields, ingestion metadata and `content_hash`. |
| `candidate_themes` | Reference table for the financial-risk themes used in thematic scoring. |
| `news_theme_annotations` | Human annotations used for development/evaluation of thematic classification. |
| `news_theme_scores` | Article-theme scores produced by the different scoring methods and model versions. |
| `news_embeddings` | Article-level document embeddings, keyed by article and embedding-pipeline identifier (`model_name`). Embeddings are stored as PostgreSQL `real[]` (`float4[]`), not pgvector. |
| `news_novelty` | Article-level novelty estimates and the parameters of the corresponding nearest-neighbour search. |
| `news_indicators` | Theme-period indicators. The current structure includes coverage, burst fields, aggregated novelty, semantic drift and firm/sector concentration fields. |
| `firm_universe` | Firm reference table prepared for firm-level/news-market extensions. |
| `news_firm_mentions` | Article-firm mention table prepared for entity/firm linkage. |

## Views

`v_valid_articles` is the operational corpus view. It retains articles satisfying the current validity conditions: English language, not flagged as exact duplicates, not flagged as verified near-duplicates, and not excluded.

`news_theme_scores_final` exposes only the frozen thematic-score versions currently used by the project:

- keyword: `seedwords_tsrank_v1`
- weak supervision: `weak_v3_continuous_frozen`
- supervised: `supervised_v3_frozen`

`v_coverage_by_source` provides monthly article counts by source, while `v_theme_coverage` is an analytical theme-coverage view. `v_gold_annotation_blind` joins the annotation sample to article text for the first annotation wave.

## Full-text processing

The function `risk_emergence.news_articles_tsv_trigger()` populates/updates:

- `body_tsv`
- `title_tsv`
- `word_count`
- `char_count`

The trigger `trg_articles_tsv` runs on inserts and on updates to article title/body. PostgreSQL English full-text search configuration is used for the `tsvector` fields.

## Indexes

The schema contains B-tree indexes for common article/date/source/theme lookups, GIN indexes on `body_tsv` and `title_tsv`, and a partial `idx_articles_valid` index matching the validity filters used by `v_valid_articles`.

## Important reproducibility note

This SQL file was reconstructed from metadata exported directly from the live `risk_emergence` schema on 2026-10-02 rather than from an older design document. The exports supplied table/column definitions, defaults, PK/FK/UNIQUE metadata, indexes, views, sequences, and the trigger-function body.

The metadata query used for constraints did **not** export the SQL expressions of four custom CHECK constraints that exist in the live database:

- `news_theme_annotations.valid_human_label`
- `news_novelty.news_novelty_k_check`
- `news_novelty.news_novelty_score_check`
- `news_novelty.news_novelty_window_check`

Their expressions have deliberately **not** been inferred or invented in `news_schema.sql`. Standard NOT NULL constraints are fully represented through the column definitions. If byte-for-byte structural equivalence with the live database is required, export these four CHECK definitions (or use `pg_dump --schema-only`) and add them to the SQL file.

The metadata export also did not include explicit foreign-key action clauses. The SQL therefore uses PostgreSQL's default FK behavior (`NO ACTION`) rather than asserting undocumented custom actions.

## Data and execution order

`news_schema.sql` creates the structure only. It does not include the original news files, the 2.2M+ cleaned articles, embeddings, novelty values, thematic scores or monthly indicators. Those objects are populated by the scripts under `03-code/` following the pipeline order documented in the repository.
