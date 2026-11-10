================================================================================
README — Struttura del repository news-embedding
Ultimo aggiornamento: giugno 2026
================================================================================


1. IL PROGETTO
================================================================================

Questo repository supporta il progetto di ricerca:

    "Emergence and Prediction of New Financial Risk Categories"

L'obiettivo è costruire un sistema che — a partire da un corpus di articoli
di stampa finanziaria — identifichi, tracci e misuri nel tempo categorie di
rischio emergenti (es. cyber risk, climate risk, pandemic risk, AI risk,
supply chain risk).

Il flusso generale prevede:

  1. Ingestione dei CSV del corpus news in un database PostgreSQL
  2. Normalizzazione, validazione e deduplicazione degli articoli
  3. Calcolo di score tematici per articolo (NLP / LLM)
  4. Generazione di embedding documentali
  5. Produzione di serie temporali e indicatori aggregati per tema

Il corpus di partenza comprende articoli 2014–2023 di cinque testate:
Financial Times (FTXX), Wall Street Journal (WSJX), New York Times (NYTX),
Los Angeles Times (LATX), USA Today (USAT).

Il database PostgreSQL si chiama "research", usa lo schema "risk_emergence",
e gira in locale (localhost:5432).


2. STRUTTURA DELLE CARTELLE
================================================================================

news-embedding/
│
├── 01-notebooks/        Jupyter notebooks di analisi ed esplorazione
├── 02-documents/        Documentazione del progetto (design, note di ricerca)
├── 03-code/             Script Python per la pipeline di ingestione
├── 04-sandbox/          Area di lavoro temporanea per test e sperimentazione
├── 05-sql/              Script SQL per la creazione dello schema PostgreSQL
├── 06-data/             Dati intermedi o di appoggio (CSV, file locali)
├── 07-papers/           Articoli accademici di riferimento
└── 08-other/            File accessori (presentazioni LaTeX, script di utilità)


3. DESCRIZIONE DEI FILE
================================================================================


── 01-notebooks/ ──────────────────────────────────────────────────────────────

  readme.txt
    Placeholder. La cartella è predisposta per i notebook Jupyter che saranno
    sviluppati nelle fasi analitiche successive (scoring, embedding, analisi
    delle serie temporali). Al momento non contiene notebook.


── 02-documents/ ──────────────────────────────────────────────────────────────

  db_design_news.docx
    Documento di progettazione del database per il modulo News. Descrive la
    struttura dello schema PostgreSQL, le tabelle, gli indici, i trigger e le
    viste. È il documento di riferimento per comprendere le scelte di design
    dell'infrastruttura dati.


── 03-code/ ───────────────────────────────────────────────────────────────────

  readme.txt
    Documentazione completa della pipeline di ingestione: scopo di ciascuno
    script, ordine di esecuzione, note su configurazione e limitazioni note.

  ingestion_manifest.csv
    File di configurazione della pipeline. Elenca i cinque CSV del corpus da
    caricare, con i campi: enabled (1/0), file_path, max_rows (limite debug),
    description. Determina quali file vengono processati dagli script Python.

  load_news_step_1_v1.py
    Step 1 della pipeline: legge i CSV elencati nel manifest e li carica nella
    staging table (risk_emergence.news_articles_staging). Usa psycopg2 con
    inserimento batch (execute_batch, chunk da 2000 righe). Registra numero
    riga originale e file sorgente per tracciabilità. Salta le righe malformate
    (numero di colonne diverso da 9).

  check_staging_table_v1.py
    Script di validazione della staging table — versione in memoria. Carica
    tutti i record con fetchall e verifica campo per campo: formato date,
    codici source, valori sentiment e is_duplicate, presenza di title/body,
    lingua. Produce un report con conteggi, esempi di record problematici e
    distribuzioni dei valori distinti. Adatto a dataset di dimensioni moderate.

  check_staging_table_v2.py
    Script di validazione della staging table — versione streaming. Stessa
    logica della v1, ma usa un cursore server-side con lettura a blocchi
    (fetchmany, 10000 righe per volta). Adatto a staging table di grandi
    dimensioni. Aggiunge contatore di avanzamento con percentuale ed ETA.

  load_news_step_2_v1.py
    Step 2 della pipeline: legge i record dalla staging table in streaming,
    li valida, normalizza e trasferisce nella tabella finale
    (risk_emergence.news_articles). Applica parsing delle date, mapping dei
    source_code, normalizzazione della lingua, calcolo del content_hash SHA-256
    per deduplicazione. Supporta un flag COMMIT_FINAL_INSERTS per eseguire in
    modalità dry-run (rollback) senza scrivere sul database.

  .vscode/settings.json
    Configurazione locale di VS Code: specifica Conda come gestore di ambienti
    e package manager Python per il workspace.


── 04-sandbox/ ────────────────────────────────────────────────────────────────

  readme.txt
    Placeholder. Cartella riservata a test, prove e sperimentazione informale.
    Al momento non contiene file operativi.


── 05-sql/ ────────────────────────────────────────────────────────────────────

  readme.txt
    Documentazione estesa del file news_schema.sql: descrive ogni oggetto
    creato, i prerequisiti di esecuzione, come verificare che lo script abbia
    funzionato e come si inserisce nel workflow complessivo del progetto.

  news_schema.sql
    Script DDL principale del modulo News. Crea nell'ordine: le estensioni
    PostgreSQL (uuid-ossp, pg_trgm), lo schema risk_emergence, la tabella di
    riferimento delle fonti (news_sources, pre-popolata con le 5 testate), la
    tabella principale degli articoli (news_articles) con i relativi indici
    e trigger per full-text search, la staging table (news_articles_staging),
    la tabella anagrafica imprese (firm_universe), la tabella delle menzioni
    articolo-impresa (news_firm_mentions), la tabella dei temi candidati
    (candidate_themes, pre-popolata con 5 temi), le tabelle per score tematici
    (news_theme_scores), embedding (news_embeddings) e indicatori aggregati
    (news_indicators), e tre viste di utilità (v_valid_articles,
    v_coverage_by_source, v_theme_coverage).


── 06-data/ ───────────────────────────────────────────────────────────────────

  readme.txt
    Placeholder. Cartella destinata a dati intermedi, CSV di appoggio o output
    parziali delle pipeline. Al momento non contiene file di dati.


── 07-papers/ ─────────────────────────────────────────────────────────────────

  readme.txt
    Placeholder. Cartella per gli articoli accademici di riferimento del progetto.

  Chen Y. et al. - Expected Returns and Large Language Models.pdf
    Articolo accademico che esplora l'uso dei Large Language Model per la
    previsione dei rendimenti attesi. Rilevante per la componente LLM-based
    del progetto (scoring tematico e sentiment degli articoli).

  Didisheim A. et al. - The Inefficient Pricing of News.pdf
    Articolo accademico sull'inefficienza del mercato nell'incorporare le
    informazioni contenute nelle news. Rilevante per il framework teorico del
    progetto sulla relazione tra flusso informativo e mercati finanziari.


── 08-other/ ──────────────────────────────────────────────────────────────────

  readme.txt
    Placeholder. Cartella per file accessori non classificati altrove.

  embedding_concepts_v0.tex
    Presentazione LaTeX (formato Beamer, 16:9) sui concetti di embedding per
    la ricerca statistica. Illustra le basi teoriche degli embedding testuali
    rilevanti per il progetto, con utilizzo di TikZ per le figure.

  template.tex
    Template LaTeX generico, base per la produzione di documenti e
    presentazioni nell'ambito del progetto.

  purge.bat
    Script batch Windows per la pulizia dei file ausiliari generati dalla
    compilazione LaTeX (rimuove .log, .aux, .toc, .gz, .nav, .out, .snm,
    .ps, .vrb e le cartelle _minted*).


4. ORDINE DI ESECUZIONE DELLA PIPELINE DI INGESTIONE
================================================================================

  1. Eseguire 05-sql/news_schema.sql sul database "research" per creare
     lo schema e tutte le tabelle.

  2. Configurare 03-code/ingestion_manifest.csv con i percorsi dei CSV
     e i flag enabled desiderati.

  3. Eseguire 03-code/load_news_step_1_v1.py per caricare i CSV in staging.

  4. Eseguire 03-code/check_staging_table_v2.py per validare la qualità
     dei dati in staging.

  5. Eseguire 03-code/load_news_step_2_v1.py per trasferire i record
     corretti nella tabella finale news_articles.

  I passi 3–4 possono essere iterati: se la validazione rivela problemi
  sistematici, si corregge e si ricarica la staging prima di procedere.


5. NOTE TECNICHE
================================================================================

- Database:    PostgreSQL, localhost:5432, database "research"
- Schema:      risk_emergence
- Separatore CSV: pipe (|), encoding UTF-8
- Python:      ambiente Conda, psycopg2
- Corpus:      5 testate, periodo 2014–2023
- Credenziali: attualmente in chiaro negli script; migrare a .env o
               ~/.pgpass prima di condividere il codice.

================================================================================
