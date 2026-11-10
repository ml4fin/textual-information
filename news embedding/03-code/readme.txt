================================================================================
README — Pipeline di ingestione news nel database PostgreSQL
================================================================================

Progetto: Emergence and Prediction of New Financial Risk Categories
Schema:   risk_emergence
Database: research (PostgreSQL, localhost)

Ultimo aggiornamento: Aprile 2026


PANORAMICA
----------

La pipeline carica gli articoli di notizie finanziarie dai file CSV
pipe-delimited nel database PostgreSQL, seguendo un processo in tre fasi:

    CSV originali  -->  staging table  -->  validazione  -->  tabella finale
       (Step 1)          (Check)              (Step 2)


ELENCO FILE
-----------

1. ingestion_manifest.csv
   Tipo: configurazione
   
   File di configurazione che elenca i CSV da caricare. Ogni riga
   corrisponde a un file sorgente con quattro campi:
   
     - enabled:     1 per caricare il file, 0 per ignorarlo
     - file_path:   percorso completo del CSV sorgente
     - max_rows:    limite righe per debug (vuoto = carica tutto)
     - description: nota descrittiva libera
   
   Attualmente elenca cinque file, uno per testata:
   FTXX (Financial Times), WSJX (Wall Street Journal),
   NYTX (New York Times), LATX (Los Angeles Times), USAT (USA Today).
   Periodo coperto: 2014-2023.


2. load_news_step_1_v1.py
   Tipo: script di ingestione (Step 1 — CSV → staging)
   
   Legge i file CSV elencati nel manifest e li carica nella staging
   table (risk_emergence.news_articles_staging). La staging table ha
   tutti i campi TEXT e nessun vincolo, per accettare qualsiasi dato
   grezzo senza errori di tipo.
   
   Caratteristiche principali:
     - Usa il modulo csv (non Pandas) per parsare i CSV riga per riga
     - Inserimento tramite execute_batch (psycopg2.extras), non riga per riga
     - Commit per chunk (BATCH_SIZE = 2000 righe)
     - Gestione del campo Body molto lungo (csv.field_size_limit aumentato)
     - Registra il numero di riga originale (_raw_line_number) e il nome
       del file sorgente (_source_file) per tracciabilità
     - Salta le righe con numero di colonne diverso da 9
     - Supporta due modalità: file singolo (load_csv_to_staging) o
       batch da manifest (load_manifest_to_staging)
     - La staging può essere svuotata (TRUNCATE) prima del caricamento
   
   Output: report con conteggio righe inserite e saltate per file.


3. check_staging_table_v1.py
   Tipo: script di validazione (Check — versione in memoria)
   
   Legge tutti i record dalla staging table e li valida campo per campo,
   producendo un report dettagliato sulla qualità dei dati. Carica tutti
   i record in memoria (fetchall).
   
   Controlli eseguiti:
     - Date: verifica che il formato sia parsabile (YYYY-MM-DD e varianti)
     - Source: verifica che il valore sia mappabile sulla tabella news_sources
     - Sentiment: verifica che sia numerico o vuoto
     - IsDuplicate: verifica che sia interpretabile come booleano
     - Title e Body: segnala record con campi vuoti
     - Language: conta i record con lingua riconducibile a inglese
   
   Output: conteggi per ciascun controllo, esempi di record problematici
   (fino a MAX_EXAMPLES_PER_CATEGORY per tipo), distribuzioni dei valori
   distinti per language, source e is_duplicate.
   
   Limitazione: carica tutti i record in memoria. Non adatto a staging
   table con milioni di righe.


4. check_staging_table_v2.py
   Tipo: script di validazione (Check — versione streaming)
   
   Stessa logica di validazione della v1, ma riscritta per lavorare in
   streaming tramite cursore server-side (fetchmany con FETCH_SIZE = 10000).
   Non carica tutti i record in memoria, quindi è adatta anche a staging
   table molto grandi.
   
   Aggiunge rispetto alla v1:
     - Cursore server-side con lettura a blocchi
     - Contatore di avanzamento con percentuale ed ETA stimata
     - Funzione update_distribution separata per maggiore leggibilità
     - Parametro MAX_DISTRIBUTION_ITEMS per limitare l'output
   
   Stesso output della v1 (report + esempi + distribuzioni), ma scalabile.


5. load_news_step_2_v1.py
   Tipo: script di trasferimento (Step 2 — staging → tabella finale)
   
   Legge i record dalla staging table in streaming (cursore server-side),
   li valida, normalizza e trasferisce nella tabella finale
   risk_emergence.news_articles.
   
   Trasformazioni applicate a ogni record:
     - Title e Body: strip; scarta se vuoti
     - Date: parsing multi-formato in oggetto date; scarta se non parsabile
     - Source: matching contro news_sources (source_raw → source_id);
       scarta se non mappabile
     - Language: normalizzazione + calcolo flag is_english
     - Sentiment: conversione in float (None se non numerico)
     - IsDuplicate: conversione in booleano; scarta se non interpretabile
     - content_hash: calcolo SHA-256 del body per deduplicazione esatta
     - ingestion_batch: derivato dal nome del file sorgente
   
   Caratteristiche:
     - Lettura streaming via cursore server-side (FETCH_SIZE = 10000)
     - Inserimento batch nella tabella finale (INSERT_BATCH_SIZE = 2000)
     - Flag COMMIT_FINAL_INSERTS per scegliere tra commit e rollback
       (utile in debug: esegui tutto senza scrivere)
     - Filtro opzionale per file sorgente (SOURCE_FILE_FILTER)
     - Report finale con conteggio processati / inseriti / scartati
     - Esempi di record scartati per ogni categoria di errore
     - Contatore di avanzamento con velocità ed ETA
   
   I campi word_count, char_count, body_tsv e title_tsv non sono calcolati
   da questo script: sono popolati automaticamente dal trigger PostgreSQL
   definito nello schema (trg_articles_tsv).


ORDINE DI ESECUZIONE
---------------------

   1. Creare lo schema e le tabelle tramite news_schema.sql
   2. Eseguire load_news_step_1_v1.py  (CSV → staging)
   3. Eseguire check_staging_table_v2.py (validazione qualità dati)
   4. Eseguire load_news_step_2_v1.py  (staging → tabella finale)

I passi 2 e 3 possono essere ripetuti iterativamente: se il check rivela
problemi sistematici (es. encoding, source non mappate), si corregge il
CSV o lo schema e si ricarica la staging.


CONFIGURAZIONE
--------------

Tutti gli script condividono gli stessi parametri di connessione,
attualmente definiti in chiaro nel codice:

   host:     localhost
   port:     5432
   database: research
   user:     postgres
   schema:   risk_emergence

Le credenziali andrebbero spostate in un file .env o in ~/.pgpass
prima di versionare il codice su Git.


NOTE
----

- I file CSV usano il separatore pipe (|) e encoding UTF-8.
  Sono noti artefatti di codifica (es. simbolo euro corrotto)
  che non vengono corretti da questa pipeline; un eventuale
  passaggio di pulizia encoding va aggiunto a monte dello Step 1.

- Lo script originale di ingestione (upload-news-into-pgsqldb.py)
  usava tabelle separate per testata e INSERT riga per riga.
  La pipeline attuale corregge entrambi i problemi: tabella unica
  con source_id e inserimento batch.

================================================================================
