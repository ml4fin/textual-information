# Importa strumenti per hashing del body.
import hashlib

# Importa datetime per il parsing delle date.
from datetime import datetime, date

# Importa tempo per contatore avanzamento ed ETA.
import time

# Importa csv solo per eventuali estensioni future e coerenza con gli step precedenti.
import csv

# Importa tipi usati nelle annotazioni.
from typing import Any, Dict, List, Optional, Tuple

# Importa psycopg2 per la connessione a PostgreSQL.
import psycopg2

# Importa execute_batch per gli insert batch nella tabella finale.
from psycopg2.extras import execute_batch


# ============================================================================
# CONFIGURAZIONE DEBUG / SVILUPPO
# ============================================================================

# Flag informativo: script pensato per esecuzione da IDE.
RUN_FROM_IDE = True

# Se valorizzato, trasferisce solo i record provenienti da questo file sorgente.
# Esempio: "FTXX_2014_2023.csv"
SOURCE_FILE_FILTER = None

# Numero di record letti per volta dalla staging.
FETCH_SIZE = 10000

# Numero di record inseriti per singolo batch nella tabella finale.
INSERT_BATCH_SIZE = 2000

# Ogni quanti record processati stampare il progresso.
PROGRESS_EVERY_N_ROWS = 50000

# Numero massimo di esempi problematici da conservare per categoria.
MAX_EXAMPLES_PER_CATEGORY = 5

# Se True, stampa alcuni esempi di record scartati.
PRINT_REJECTION_EXAMPLES = True

# Se True, esegue il commit finale; se False fa rollback a fine script.
# Molto utile in debug.
COMMIT_FINAL_INSERTS = True

# Parametri di connessione al database.
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "research",
    "user": "postgres",
    "password": "Unibo1234",
}

# Nome dello schema del progetto.
SCHEMA_NAME = "risk_emergence"

# Nome della staging table.
STAGING_TABLE = "news_articles_staging"

# Nome della tabella finale news.
NEWS_TABLE = "news_articles"

# Nome della tabella fonti.
SOURCES_TABLE = "news_sources"


# ============================================================================
# CONNESSIONE DATABASE
# ============================================================================

def get_connection():
    """
    Crea una connessione al database PostgreSQL.
    """
    # Restituisce una connessione PostgreSQL usando i parametri configurati.
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )


# ============================================================================
# VERIFICHE STRUTTURALI
# ============================================================================

def check_schema_and_tables(conn) -> None:
    """
    Verifica che le tabelle minime richieste esistano nello schema.
    """
    # Query che controlla la presenza delle tabelle necessarie.
    query = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema = %s
      AND table_name IN ('news_articles_staging', 'news_articles', 'news_sources')
    ORDER BY table_name;
    """

    # Apre un cursore per eseguire la query.
    with conn.cursor() as cur:
        # Esegue la query passando il nome dello schema.
        cur.execute(query, (SCHEMA_NAME,))
        rows = cur.fetchall()

    # Converte il risultato in un insieme di tuple.
    found = {(r[0], r[1]) for r in rows}

    # Definisce le tabelle richieste.
    required = {
        (SCHEMA_NAME, "news_articles_staging"),
        (SCHEMA_NAME, "news_articles"),
        (SCHEMA_NAME, "news_sources"),
    }

    # Individua eventuali tabelle mancanti.
    missing = required - found

    # Se manca qualcosa, interrompe l'esecuzione.
    if missing:
        raise RuntimeError(f"Tabelle mancanti nello schema {SCHEMA_NAME}: {missing}")

    # Messaggio di conferma.
    print("Verifica schema OK: tabelle richieste presenti.")


def get_staging_row_count(conn, source_file_filter: Optional[str] = None) -> int:
    """
    Conta i record presenti nella staging table.
    """
    # Query base di conteggio.
    query = f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{STAGING_TABLE}"
    params: tuple = ()

    # Se richiesto, filtra per file sorgente.
    if source_file_filter is not None:
        query += " WHERE _source_file = %s"
        params = (source_file_filter,)

    # Esegue la query di conteggio.
    with conn.cursor() as cur:
        cur.execute(query, params)
        count = cur.fetchone()[0]

    # Restituisce il totale.
    return count


def get_source_mapping(conn) -> Dict[str, int]:
    """
    Legge la mappatura source_raw/source_domain -> source_id da news_sources.
    """
    # Legge le colonne reali della tabella news_sources.
    query_cols = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position;
    """

    # Recupera i nomi di colonna disponibili.
    with conn.cursor() as cur:
        cur.execute(query_cols, (SCHEMA_NAME, SOURCES_TABLE))
        cols = [r[0] for r in cur.fetchall()]

    # Possibili colonne candidate che descrivono la sorgente grezza/domìnio.
    candidate_columns = [
        "source_domain",
        "domain",
        "source_raw",
        "source_code",
        "source_name",
    ]

    # Seleziona la prima colonna plausibile presente.
    source_col = None
    for candidate in candidate_columns:
        if candidate in cols:
            source_col = candidate
            break

    # Se non trova nessuna colonna compatibile, interrompe.
    if source_col is None:
        raise RuntimeError(
            "Impossibile identificare la colonna di news_sources da usare per il matching. "
            f"Colonne trovate: {cols}"
        )

    # Query che legge source_id e valore sorgente.
    query = f"""
    SELECT source_id, {source_col}
    FROM {SCHEMA_NAME}.{SOURCES_TABLE}
    WHERE {source_col} IS NOT NULL
    """

    # Esegue la query.
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    # Costruisce la mappa normalizzata source -> source_id.
    source_mapping = {
        str(row[1]).strip().lower(): row[0]
        for row in rows
        if row[1] is not None
    }

    # Stampa quale colonna è stata usata.
    print(f"Colonna usata in news_sources per il matching: {source_col}")

    # Restituisce la mappa completa.
    return source_mapping


# ============================================================================
# FUNZIONI DI NORMALIZZAZIONE / VALIDAZIONE
# ============================================================================

def is_blank(value: Optional[str]) -> bool:
    """
    Verifica se un valore è nullo, vuoto o composto solo da spazi.
    """
    # None è considerato vuoto.
    if value is None:
        return True

    # Dopo strip, stringa vuota -> True.
    return str(value).strip() == ""


def normalize_text_or_none(value: Optional[str]) -> Optional[str]:
    """
    Restituisce la stringa ripulita o None se vuota.
    """
    # Se il valore è vuoto, restituisce None.
    if is_blank(value):
        return None

    # Restituisce la stringa con strip.
    return str(value).strip()


def parse_date(value: Optional[str]) -> Optional[date]:
    """
    Converte la data testuale in oggetto date se possibile.
    """
    # Valore vuoto -> None.
    if is_blank(value):
        return None

    # Normalizza il testo.
    text = str(value).strip()

    # Elenco dei formati attesi.
    candidate_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]

    # Prova i vari formati.
    for fmt in candidate_formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass

    # Nessun parsing riuscito.
    return None


def parse_float_or_none(value: Optional[str]) -> Optional[float]:
    """
    Converte una stringa numerica in float; se non convertibile restituisce None.
    """
    # Valore vuoto -> None.
    if is_blank(value):
        return None

    # Normalizza il testo.
    text = str(value).strip()

    # Prova la conversione.
    try:
        return float(text)
    except ValueError:
        return None


def parse_duplicate_flag(value: Optional[str]) -> Optional[bool]:
    """
    Interpreta is_duplicate come booleano; se il valore non è riconosciuto restituisce None.
    """
    # Valore vuoto -> None.
    if is_blank(value):
        return None

    # Normalizza il testo.
    text = str(value).strip().lower()

    # Mappa dei valori accettati.
    truthy = {"true", "t", "1", "yes"}
    falsy = {"false", "f", "0", "no"}

    # Restituisce True per i valori positivi.
    if text in truthy:
        return True

    # Restituisce False per i valori negativi.
    if text in falsy:
        return False

    # Valore non riconosciuto.
    return None


def looks_english(value: Optional[str]) -> bool:
    """
    Verifica se il campo language suggerisce inglese.
    """
    # Valore vuoto -> False.
    if is_blank(value):
        return False

    # Normalizza il testo.
    text = str(value).strip().lower()

    # Restituisce True per i codici attesi.
    return text in {"en", "eng", "english"}


def compute_content_hash_bytes(body: str) -> bytes:
    """
    Calcola l'hash SHA-256 del body e lo restituisce come bytes.
    """
    # Converte il body in bytes UTF-8 e calcola SHA-256.
    return hashlib.sha256(body.encode("utf-8", errors="ignore")).digest()


# ============================================================================
# UTILITA' REPORT / PROGRESS
# ============================================================================

def add_example(examples_dict: Dict[str, List[dict]], category: str, record: dict) -> None:
    """
    Aggiunge un record agli esempi di una categoria, fino al massimo configurato.
    """
    # Crea la lista della categoria se non esiste ancora.
    if category not in examples_dict:
        examples_dict[category] = []

    # Aggiunge l'esempio solo se il limite non è stato raggiunto.
    if len(examples_dict[category]) < MAX_EXAMPLES_PER_CATEGORY:
        examples_dict[category].append(record)


def format_seconds(seconds: float) -> str:
    """
    Converte secondi in una stringa leggibile hh:mm:ss.
    """
    # Gestisce valori negativi/non disponibili.
    if seconds < 0:
        return "n/a"

    # Arrotonda ai secondi interi.
    total = int(seconds)

    # Calcola ore, minuti, secondi.
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    # Restituisce la stringa formattata.
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(processed: int, total: int, inserted: int, rejected: int, start_time: float) -> None:
    """
    Stampa il contatore di avanzamento in tempo reale.
    """
    # Calcola il tempo trascorso.
    elapsed = time.time() - start_time

    # Calcola la velocità media di processamento.
    speed = processed / elapsed if elapsed > 0 else 0.0

    # Calcola la percentuale completata.
    pct = (processed / total * 100.0) if total > 0 else 0.0

    # Calcola i record rimanenti.
    remaining = max(total - processed, 0)

    # Stima l'ETA.
    eta_seconds = (remaining / speed) if speed > 0 else -1

    # Stampa lo stato corrente.
    print(
        f"[PROGRESS] processati={processed:,}/{total:,} "
        f"({pct:6.2f}%) | inseriti={inserted:,} | scartati={rejected:,} "
        f"| velocita={speed:,.0f} rec/s | elapsed={format_seconds(elapsed)} "
        f"| ETA={format_seconds(eta_seconds)}"
    )


# ============================================================================
# PREPARAZIONE RECORD FINALE
# ============================================================================

def transform_staging_record(
    record: Dict[str, Any],
    source_mapping: Dict[str, int],
    rejection_examples: Dict[str, List[dict]],
) -> Optional[Tuple]:
    """
    Trasforma un record della staging nel record finale da inserire in news_articles.

    Restituisce:
    - una tuple pronta per INSERT se il record è valido;
    - None se il record va scartato.
    """
    # Normalizza title e body.
    title = normalize_text_or_none(record["title"])
    body = normalize_text_or_none(record["body"])

    # Se title è vuoto, scarta il record.
    if title is None:
        add_example(rejection_examples, "blank_title", record)
        return None

    # Se body è vuoto, scarta il record.
    if body is None:
        add_example(rejection_examples, "blank_body", record)
        return None

    # Converte la data.
    publication_date = parse_date(record["publication_date"])

    # Se la data non è valida, scarta il record.
    if publication_date is None:
        add_example(rejection_examples, "invalid_date", record)
        return None

    # Normalizza il campo source_raw.
    source_raw = normalize_text_or_none(record["source"])

    # Se source è vuoto, scarta il record.
    if source_raw is None:
        add_example(rejection_examples, "blank_source", record)
        return None

    # Cerca source_id tramite la mappa.
    source_key = source_raw.strip().lower()
    source_id = source_mapping.get(source_key)

    # Se la source non è mappabile, scarta il record.
    if source_id is None:
        add_example(rejection_examples, "unmapped_source", record)
        return None

    # Normalizza language.
    language = normalize_text_or_none(record["language"])

    # Calcola il flag is_english.
    is_english = looks_english(language)

    # Converte sentiment; se non convertibile, mette None.
    sentiment_original = parse_float_or_none(record["sentiment"])

    # Converte is_duplicate; se non riconosciuto, scarta il record.
    duplicate_flag = parse_duplicate_flag(record["is_duplicate"])
    if duplicate_flag is None:
        add_example(rejection_examples, "invalid_is_duplicate", record)
        return None

    # Normalizza data_type.
    data_type = normalize_text_or_none(record["data_type"])

    # Normalizza url.
    url = normalize_text_or_none(record["url"])

    # Ricava il batch di ingestione dal nome del file sorgente.
    ingestion_batch = normalize_text_or_none(record["_source_file"])

    # Calcola l'hash del body.
    content_hash = compute_content_hash_bytes(body)

    # Restituisce la tuple nel formato atteso dalla tabella finale.
    return (
        title,                  # title
        body,                   # body
        publication_date,       # publication_date
        language,               # language
        source_raw,             # source_raw
        sentiment_original,     # sentiment_original
        duplicate_flag,         # is_duplicate_flag
        data_type,              # data_type
        url,                    # url
        source_id,              # source_id
        is_english,             # is_english
        False,                  # is_duplicate_verified
        False,                  # is_excluded
        ingestion_batch,        # ingestion_batch
        content_hash,           # content_hash
    )


# ============================================================================
# INSERT NELLA TABELLA FINALE
# ============================================================================

def insert_batch_into_news(conn, batch_rows: List[Tuple]) -> None:
    """
    Inserisce un batch di record nella tabella finale news_articles.
    """
    # Query parametrica di insert nella tabella finale.
    sql = f"""
    INSERT INTO {SCHEMA_NAME}.{NEWS_TABLE}
    (
        title,
        body,
        publication_date,
        language,
        source_raw,
        sentiment_original,
        is_duplicate_flag,
        data_type,
        url,
        source_id,
        is_english,
        is_duplicate_verified,
        is_excluded,
        ingestion_batch,
        content_hash
    )
    VALUES
    (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """

    # Apre un cursore per l'inserimento batch.
    with conn.cursor() as cur:
        # Esegue l'insert batch.
        execute_batch(cur, sql, batch_rows, page_size=INSERT_BATCH_SIZE)


# ============================================================================
# CARICAMENTO STREAMING STAGING -> NEWS
# ============================================================================

def load_staging_into_news(
    conn,
    source_mapping: Dict[str, int],
    source_file_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Legge i record dalla staging in streaming e carica nella tabella finale solo quelli corretti.
    """
    # Conta i record da processare.
    total_records = get_staging_row_count(conn, source_file_filter=source_file_filter)

    # Inizializza il report.
    report = {
        "total_records": total_records,
        "processed_records": 0,
        "inserted_records": 0,
        "rejected_records": 0,
        "rejection_examples": {},
    }

    # Se non ci sono record, restituisce subito il report.
    if total_records == 0:
        return report

    # Costruisce la query di lettura dalla staging.
    query = f"""
    SELECT
        title,
        body,
        publication_date,
        language,
        source,
        sentiment,
        is_duplicate,
        data_type,
        url,
        _raw_line_number,
        _source_file
    FROM {SCHEMA_NAME}.{STAGING_TABLE}
    """
    params: tuple = ()

    # Applica il filtro per file sorgente se richiesto.
    if source_file_filter is not None:
        query += " WHERE _source_file = %s"
        params = (source_file_filter,)

    # Ordina per file e numero riga per rendere più leggibile il debug.
    query += " ORDER BY _source_file, _raw_line_number"

    # Nome del cursore server-side per streaming.
    cursor_name = "staging_to_news_cursor"

    # Batch di insert nella tabella finale.
    insert_batch: List[Tuple] = []

    # Registra il tempo di inizio.
    start_time = time.time()

    # Prossima soglia di stampa progresso.
    next_progress_threshold = PROGRESS_EVERY_N_ROWS

    # Usa un cursore server-side per non caricare tutto in memoria.
    with conn.cursor(name=cursor_name) as cur:
        # Esegue la query.
        cur.execute(query, params)

        # Imposta la dimensione interna dei blocchi.
        cur.itersize = FETCH_SIZE

        # Continua finché ci sono record.
        while True:
            # Legge il prossimo blocco di righe.
            rows = cur.fetchmany(FETCH_SIZE)

            # Se non ci sono più righe, esce dal ciclo.
            if not rows:
                break

            # Elabora il blocco corrente.
            for row in rows:
                # Costruisce il record come dizionario.
                record = {
                    "title": row[0],
                    "body": row[1],
                    "publication_date": row[2],
                    "language": row[3],
                    "source": row[4],
                    "sentiment": row[5],
                    "is_duplicate": row[6],
                    "data_type": row[7],
                    "url": row[8],
                    "_raw_line_number": row[9],
                    "_source_file": row[10],
                }

                # Incrementa il numero di record processati.
                report["processed_records"] += 1

                # Trasforma il record staging in record finale.
                final_row = transform_staging_record(
                    record=record,
                    source_mapping=source_mapping,
                    rejection_examples=report["rejection_examples"],
                )

                # Se il record non è valido, incrementa gli scarti.
                if final_row is None:
                    report["rejected_records"] += 1
                else:
                    # Se valido, lo aggiunge al batch di insert.
                    insert_batch.append(final_row)

                # Se il batch è pieno, lo inserisce nella tabella finale.
                if len(insert_batch) >= INSERT_BATCH_SIZE:
                    insert_batch_into_news(conn, insert_batch)
                    report["inserted_records"] += len(insert_batch)
                    insert_batch = []

                # Stampa il progresso alle soglie previste.
                if report["processed_records"] >= next_progress_threshold:
                    print_progress(
                        processed=report["processed_records"],
                        total=report["total_records"],
                        inserted=report["inserted_records"],
                        rejected=report["rejected_records"],
                        start_time=start_time,
                    )
                    next_progress_threshold += PROGRESS_EVERY_N_ROWS

    # Inserisce l'ultimo batch rimasto.
    if insert_batch:
        insert_batch_into_news(conn, insert_batch)
        report["inserted_records"] += len(insert_batch)

    # Stampa sempre l'ultima riga di avanzamento.
    print_progress(
        processed=report["processed_records"],
        total=report["total_records"],
        inserted=report["inserted_records"],
        rejected=report["rejected_records"],
        start_time=start_time,
    )

    # Restituisce il report finale.
    return report


# ============================================================================
# STAMPA REPORT FINALE
# ============================================================================

def print_examples(title: str, examples: List[dict]) -> None:
    """
    Stampa alcuni esempi di record scartati.
    """
    # Se non ci sono esempi, non stampa nulla.
    if not examples:
        return

    # Stampa il titolo della sezione.
    print("")
    print(title)

    # Stampa ogni esempio in forma sintetica.
    for ex in examples:
        print(
            f"  file={ex['_source_file']}, "
            f"line={ex['_raw_line_number']}, "
            f"date={ex['publication_date']}, "
            f"source={ex['source']}, "
            f"language={ex['language']}, "
            f"is_duplicate={ex['is_duplicate']}, "
            f"title_preview={repr((ex['title'] or '')[:60])}"
        )


def print_load_report(report: Dict[str, Any]) -> None:
    """
    Stampa il report finale del caricamento.
    """
    # Stampa intestazione.
    print("")
    print("=" * 80)
    print("REPORT CARICAMENTO STAGING -> NEWS")
    print("=" * 80)

    # Stampa i conteggi principali.
    print(f"Totale record letti      : {report['total_records']}")
    print(f"Totale record processati : {report['processed_records']}")
    print(f"Totale record inseriti   : {report['inserted_records']}")
    print(f"Totale record scartati   : {report['rejected_records']}")

    # Se richiesto, stampa anche esempi di scarto.
    if PRINT_REJECTION_EXAMPLES:
        print_examples("ESEMPI - title vuoti", report["rejection_examples"].get("blank_title", []))
        print_examples("ESEMPI - body vuoti", report["rejection_examples"].get("blank_body", []))
        print_examples("ESEMPI - date non valide", report["rejection_examples"].get("invalid_date", []))
        print_examples("ESEMPI - source vuote", report["rejection_examples"].get("blank_source", []))
        print_examples("ESEMPI - source non mappate", report["rejection_examples"].get("unmapped_source", []))
        print_examples("ESEMPI - is_duplicate non valido", report["rejection_examples"].get("invalid_is_duplicate", []))


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Punto di ingresso principale.

    Questa v1 legge i record corretti dalla staging e li inserisce
    nella tabella finale news_articles.
    """
    # Apre la connessione al database.
    with get_connection() as conn:
        # Disattiva l'autocommit per controllare esplicitamente commit/rollback.
        conn.autocommit = False

        # Conferma che la connessione è riuscita.
        print("Connessione al database riuscita.")

        # Verifica che le tabelle richieste esistano.
        check_schema_and_tables(conn)

        # Conta i record presenti in staging.
        total_in_staging = get_staging_row_count(
            conn,
            source_file_filter=SOURCE_FILE_FILTER,
        )

        # Se non ci sono record, termina.
        if total_in_staging == 0:
            print("La staging table non contiene record da trasferire.")
            return

        # Stampa quanti record saranno processati.
        print(f"Record in staging da trasferire: {total_in_staging:,}")

        # Se attivo, stampa il filtro sul file sorgente.
        if SOURCE_FILE_FILTER is not None:
            print(f"Filtro _source_file attivo: {SOURCE_FILE_FILTER}")

        # Legge la mappa source -> source_id.
        source_mapping = get_source_mapping(conn)

        # Esegue il caricamento streaming staging -> news.
        report = load_staging_into_news(
            conn=conn,
            source_mapping=source_mapping,
            source_file_filter=SOURCE_FILE_FILTER,
        )

        # Se richiesto, conferma le scritture.
        if COMMIT_FINAL_INSERTS:
            conn.commit()
            print("Commit finale eseguito.")
        else:
            conn.rollback()
            print("Rollback eseguito: nessun dato scritto in news_articles.")

        # Stampa il report finale.
        print_load_report(report)


# Esegue main solo se il file è lanciato direttamente.
if __name__ == "__main__":
    main()