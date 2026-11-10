# Importa il modulo standard per leggere CSV.
import csv

# Aumenta in modo robusto il limite massimo per i campi CSV.
# Serve perché i body delle news possono essere molto lunghi.
def increase_csv_field_size_limit() -> None:
    """
    Aumenta il limite massimo accettato dal modulo csv per un singolo campo.
    """
    import sys

    # Parte dal massimo intero disponibile sulla piattaforma.
    max_int = sys.maxsize

    # Riduce progressivamente il valore finché viene accettato.
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)


# Applica subito l'aumento del limite CSV.
increase_csv_field_size_limit()

# Importa datetime per il parsing delle date.
from datetime import datetime

# Importa strumenti per misurare il tempo e stimare avanzamento/ETA.
import time

# Importa tipi usati nelle annotazioni.
from typing import Any, Dict, List, Optional, Tuple

# Importa psycopg2 per la connessione a PostgreSQL.
import psycopg2


# ============================================================================
# CONFIGURAZIONE DEBUG / SVILUPPO
# ============================================================================

# Flag informativo: script pensato per esecuzione da IDE.
RUN_FROM_IDE = True

# Se valorizzato, valida solo i record provenienti da questo file sorgente.
# Esempio: "FTXX_2014_2023.csv"
SOURCE_FILE_FILTER = None

# Numero massimo di esempi problematici da conservare per categoria.
MAX_EXAMPLES_PER_CATEGORY = 5

# Se True, stampa anche le distribuzioni dei valori distinti più frequenti.
PRINT_VALUE_DISTRIBUTIONS = True

# Numero massimo di valori distinti da stampare per ciascuna distribuzione.
MAX_DISTRIBUTION_ITEMS = 30

# Numero di righe lette per volta dal database.
FETCH_SIZE = 10000

# Ogni quanti record aggiornare il contatore di avanzamento.
PROGRESS_EVERY_N_ROWS = 50000

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

# Nome della tabella delle fonti normalizzate.
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

        # Recupera tutte le righe restituite.
        rows = cur.fetchall()

    # Costruisce l'insieme delle tabelle trovate.
    found = {(r[0], r[1]) for r in rows}

    # Definisce l'insieme delle tabelle richieste.
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

    # Se richiesto, applica il filtro sul file sorgente.
    if source_file_filter is not None:
        query += " WHERE _source_file = %s"
        params = (source_file_filter,)

    # Esegue la query di conteggio.
    with conn.cursor() as cur:
        cur.execute(query, params)
        count = cur.fetchone()[0]

    # Restituisce il numero di record.
    return count


def get_allowed_sources(conn) -> set:
    """
    Legge i valori ammessi per le fonti dalla tabella news_sources.
    """
    # Query che legge l'elenco delle colonne della tabella news_sources.
    query_cols = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position;
    """

    # Legge le colonne realmente presenti.
    with conn.cursor() as cur:
        cur.execute(query_cols, (SCHEMA_NAME, SOURCES_TABLE))
        cols = [r[0] for r in cur.fetchall()]

    # Possibili nomi della colonna che contiene il dominio/source grezzo.
    candidate_columns = [
        "source_domain",
        "domain",
        "source_raw",
        "source_code",
        "source_name",
    ]

    # Seleziona la prima colonna candidata trovata.
    selected_col = None
    for candidate in candidate_columns:
        if candidate in cols:
            selected_col = candidate
            break

    # Se non trova nessuna colonna plausibile, interrompe l'esecuzione.
    if selected_col is None:
        raise RuntimeError(
            "Impossibile identificare la colonna di news_sources "
            f"da usare per il matching. Colonne trovate: {cols}"
        )

    # Query che legge i valori distinti ammessi.
    query_values = f"""
    SELECT DISTINCT {selected_col}
    FROM {SCHEMA_NAME}.{SOURCES_TABLE}
    WHERE {selected_col} IS NOT NULL
    """

    # Esegue la query.
    with conn.cursor() as cur:
        cur.execute(query_values)
        rows = cur.fetchall()

    # Normalizza i valori per un confronto robusto.
    allowed_sources = {str(r[0]).strip().lower() for r in rows if r[0] is not None}

    # Stampa la colonna scelta per il matching.
    print(f"Colonna usata in news_sources per il matching: {selected_col}")

    # Restituisce i valori ammessi.
    return allowed_sources


# ============================================================================
# FUNZIONI DI VALIDAZIONE SINGOLO CAMPO
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


def is_valid_date_string(value: Optional[str]) -> bool:
    """
    Verifica se la data è parseabile in uno dei formati attesi.
    """
    # Una data vuota non è valida.
    if is_blank(value):
        return False

    # Normalizza la stringa.
    text = str(value).strip()

    # Elenco dei formati candidati.
    candidate_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]

    # Prova a fare il parsing con ciascun formato.
    for fmt in candidate_formats:
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            pass

    # Nessun formato compatibile trovato.
    return False


def is_valid_numeric_string(value: Optional[str]) -> bool:
    """
    Verifica se il valore può essere convertito in float.
    """
    # Valore vuoto ammesso come missing.
    if is_blank(value):
        return True

    # Normalizza il testo.
    text = str(value).strip()

    # Prova la conversione numerica.
    try:
        float(text)
        return True
    except ValueError:
        return False


def is_valid_duplicate_flag(value: Optional[str]) -> bool:
    """
    Verifica se il campo is_duplicate è interpretabile come booleano.
    """
    # Valore vuoto ammesso in questa fase.
    if is_blank(value):
        return True

    # Normalizza il testo.
    text = str(value).strip().lower()

    # Insieme dei valori ritenuti validi.
    valid_values = {"true", "false", "t", "f", "1", "0", "yes", "no"}

    # Restituisce True se riconosciuto.
    return text in valid_values


def looks_english(value: Optional[str]) -> bool:
    """
    Verifica se il campo language suggerisce inglese.
    """
    # Valore vuoto -> non inglese.
    if is_blank(value):
        return False

    # Normalizza il testo.
    text = str(value).strip().lower()

    # Restituisce True per i codici/nomi attesi dell'inglese.
    return text in {"en", "eng", "english"}


# ============================================================================
# UTILITA' REPORT
# ============================================================================

def add_example(examples_dict: Dict[str, List[dict]], category: str, record: dict) -> None:
    """
    Aggiunge un record agli esempi di una categoria, fino al massimo configurato.
    """
    # Crea la lista della categoria se non esiste ancora.
    if category not in examples_dict:
        examples_dict[category] = []

    # Aggiunge l'esempio solo se non è già stato raggiunto il limite.
    if len(examples_dict[category]) < MAX_EXAMPLES_PER_CATEGORY:
        examples_dict[category].append(record)


def update_distribution(distribution: Dict[str, int], key: Optional[str]) -> None:
    """
    Aggiorna una distribuzione di frequenze.
    """
    # Normalizza None in stringa vuota.
    norm_key = "" if key is None else str(key).strip()

    # Incrementa il conteggio del valore osservato.
    distribution[norm_key] = distribution.get(norm_key, 0) + 1


def print_examples(title: str, examples: List[dict]) -> None:
    """
    Stampa alcuni esempi di record problematici.
    """
    # Se non ci sono esempi, non stampa nulla.
    if not examples:
        return

    # Stampa un titolo leggibile.
    print("")
    print(title)

    # Stampa ciascun esempio in forma sintetica.
    for ex in examples:
        print(
            f"  file={ex['_source_file']}, "
            f"line={ex['_raw_line_number']}, "
            f"date={ex['publication_date']}, "
            f"source={ex['source']}, "
            f"language={ex['language']}, "
            f"is_duplicate={ex['is_duplicate']}, "
            f"sentiment={ex['sentiment']}, "
            f"title_preview={repr((ex['title'] or '')[:60])}"
        )


def print_distribution(title: str, distribution: Dict[str, int], max_items: int = 30) -> None:
    """
    Stampa una distribuzione di frequenze ordinata per numerosità decrescente.
    """
    # Se la distribuzione è vuota, non stampa nulla.
    if not distribution:
        return

    # Ordina per conteggio decrescente e poi per chiave.
    sorted_items = sorted(distribution.items(), key=lambda x: (-x[1], x[0]))

    # Stampa il titolo della distribuzione.
    print("")
    print(title)

    # Stampa solo i primi max_items valori.
    for key, count in sorted_items[:max_items]:
        print(f"  {repr(key)} -> {count}")

    # Se ci sono altri valori, lo segnala.
    if len(sorted_items) > max_items:
        print(f"  ... altri {len(sorted_items) - max_items} valori non mostrati")


def format_seconds(seconds: float) -> str:
    """
    Converte un numero di secondi in una stringa leggibile.
    """
    # Gestisce casi non significativi.
    if seconds < 0:
        return "n/a"

    # Arrotonda ai secondi interi.
    total = int(seconds)

    # Calcola ore, minuti e secondi.
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    # Restituisce la stringa formattata.
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(processed: int, total: int, start_time: float) -> None:
    """
    Stampa un contatore di avanzamento con percentuale, velocità ed ETA.
    """
    # Calcola il tempo trascorso.
    elapsed = time.time() - start_time

    # Evita divisioni per zero.
    speed = processed / elapsed if elapsed > 0 else 0.0

    # Calcola la percentuale completata.
    pct = (processed / total * 100.0) if total > 0 else 0.0

    # Calcola i record ancora da elaborare.
    remaining = max(total - processed, 0)

    # Stima il tempo rimanente.
    eta_seconds = (remaining / speed) if speed > 0 else -1

    # Stampa la riga di avanzamento.
    print(
        f"[PROGRESS] processati={processed:,}/{total:,} "
        f"({pct:6.2f}%) | velocita={speed:,.0f} rec/s "
        f"| elapsed={format_seconds(elapsed)} "
        f"| ETA={format_seconds(eta_seconds)}"
    )


# ============================================================================
# INIZIALIZZAZIONE REPORT
# ============================================================================

def initialize_report(total_records: int) -> Dict[str, Any]:
    """
    Inizializza la struttura del report di validazione.
    """
    # Crea il dizionario base dei contatori.
    return {
        "total_records": total_records,
        "valid_dates": 0,
        "invalid_dates": 0,
        "mapped_sources": 0,
        "unmapped_sources": 0,
        "valid_sentiment": 0,
        "invalid_sentiment": 0,
        "valid_is_duplicate": 0,
        "invalid_is_duplicate": 0,
        "blank_title": 0,
        "blank_body": 0,
        "english_like": 0,
        "non_english_like": 0,
        "examples": {},
        "language_distribution": {},
        "source_distribution": {},
        "duplicate_distribution": {},
    }


# ============================================================================
# VALIDAZIONE STREAMING
# ============================================================================

def process_record(
    record: Dict[str, Any],
    allowed_sources: set,
    report: Dict[str, Any],
) -> None:
    """
    Applica i controlli di qualità a un singolo record e aggiorna il report.
    """
    # Estrae i campi principali del record.
    title = record["title"]
    body = record["body"]
    publication_date = record["publication_date"]
    language = record["language"]
    source = record["source"]
    sentiment = record["sentiment"]
    is_duplicate = record["is_duplicate"]

    # Aggiorna le distribuzioni dei valori distinti.
    update_distribution(report["language_distribution"], language)
    update_distribution(report["source_distribution"], source)
    update_distribution(report["duplicate_distribution"], is_duplicate)

    # Controlla la validità della data.
    if is_valid_date_string(publication_date):
        report["valid_dates"] += 1
    else:
        report["invalid_dates"] += 1
        add_example(report["examples"], "invalid_dates", record)

    # Normalizza la source per il confronto con news_sources.
    source_norm = "" if source is None else str(source).strip().lower()

    # Controlla se la source è mappabile.
    if source_norm != "" and source_norm in allowed_sources:
        report["mapped_sources"] += 1
    else:
        report["unmapped_sources"] += 1
        add_example(report["examples"], "unmapped_sources", record)

    # Controlla la validità del sentiment.
    if is_valid_numeric_string(sentiment):
        report["valid_sentiment"] += 1
    else:
        report["invalid_sentiment"] += 1
        add_example(report["examples"], "invalid_sentiment", record)

    # Controlla la validità del flag is_duplicate.
    if is_valid_duplicate_flag(is_duplicate):
        report["valid_is_duplicate"] += 1
    else:
        report["invalid_is_duplicate"] += 1
        add_example(report["examples"], "invalid_is_duplicate", record)

    # Conta i title vuoti.
    if is_blank(title):
        report["blank_title"] += 1
        add_example(report["examples"], "blank_title", record)

    # Conta i body vuoti.
    if is_blank(body):
        report["blank_body"] += 1
        add_example(report["examples"], "blank_body", record)

    # Conta se la lingua sembra inglese.
    if looks_english(language):
        report["english_like"] += 1
    else:
        report["non_english_like"] += 1


def stream_validate_staging(
    conn,
    allowed_sources: set,
    source_file_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Valida la staging in streaming, senza caricare tutto in memoria.
    """
    # Conta prima quanti record verranno elaborati.
    total_records = get_staging_row_count(conn, source_file_filter=source_file_filter)

    # Se non ci sono record, restituisce un report vuoto ma coerente.
    report = initialize_report(total_records=total_records)
    if total_records == 0:
        return report

    # Costruisce la query per leggere i record necessari alla validazione.
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

    # Se richiesto, limita l'analisi a un file specifico.
    if source_file_filter is not None:
        query += " WHERE _source_file = %s"
        params = (source_file_filter,)

    # Ordina i record per rendere il debug più leggibile.
    query += " ORDER BY _source_file, _raw_line_number"

    # Crea un cursore server-side per leggere a blocchi dal database.
    # Il nome del cursore consente a PostgreSQL di non inviare tutto insieme.
    cursor_name = "staging_validation_cursor"

    # Registra il tempo di inizio per il contatore di avanzamento.
    start_time = time.time()

    # Contatore dei record processati.
    processed = 0

    # Ultima soglia alla quale è stato stampato il progresso.
    next_progress_threshold = PROGRESS_EVERY_N_ROWS

    # Usa un cursore named per streaming lato server.
    with conn.cursor(name=cursor_name) as cur:
        # Esegue la query sul cursore server-side.
        cur.execute(query, params)

        # Imposta quanti record recuperare internamente per volta.
        cur.itersize = FETCH_SIZE

        # Continua finché ci sono righe da leggere.
        while True:
            # Legge il prossimo blocco di righe.
            rows = cur.fetchmany(FETCH_SIZE)

            # Se non ci sono più righe, esce dal ciclo.
            if not rows:
                break

            # Elabora tutte le righe del blocco corrente.
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

                # Valida il singolo record aggiornando il report.
                process_record(record, allowed_sources, report)

                # Incrementa il numero di record processati.
                processed += 1

                # Stampa il progresso alle soglie prefissate.
                if processed >= next_progress_threshold:
                    print_progress(processed, total_records, start_time)
                    next_progress_threshold += PROGRESS_EVERY_N_ROWS

    # Stampa sempre una riga finale di completamento.
    print_progress(processed, total_records, start_time)

    # Restituisce il report completo.
    return report


# ============================================================================
# STAMPA REPORT
# ============================================================================

def print_validation_report(report: Dict[str, Any]) -> None:
    """
    Stampa in modo leggibile il report di validazione.
    """
    # Stampa intestazione del report.
    print("")
    print("=" * 80)
    print("REPORT VALIDAZIONE STAGING")
    print("=" * 80)

    # Stampa i conteggi principali.
    print(f"Totale record analizzati         : {report['total_records']}")
    print(f"Date valide                     : {report['valid_dates']}")
    print(f"Date non valide                 : {report['invalid_dates']}")
    print(f"Source mappate                  : {report['mapped_sources']}")
    print(f"Source non mappate              : {report['unmapped_sources']}")
    print(f"Sentiment validi                : {report['valid_sentiment']}")
    print(f"Sentiment non validi            : {report['invalid_sentiment']}")
    print(f"is_duplicate validi             : {report['valid_is_duplicate']}")
    print(f"is_duplicate non validi         : {report['invalid_is_duplicate']}")
    print(f"Title vuoti                     : {report['blank_title']}")
    print(f"Body vuoti                      : {report['blank_body']}")
    print(f"Language riconducibile a inglese: {report['english_like']}")
    print(f"Language non inglese/ignota     : {report['non_english_like']}")

    # Stampa esempi di record problematici.
    print_examples("ESEMPI - date non valide", report["examples"].get("invalid_dates", []))
    print_examples("ESEMPI - source non mappate", report["examples"].get("unmapped_sources", []))
    print_examples("ESEMPI - sentiment non validi", report["examples"].get("invalid_sentiment", []))
    print_examples("ESEMPI - is_duplicate non validi", report["examples"].get("invalid_is_duplicate", []))
    print_examples("ESEMPI - title vuoti", report["examples"].get("blank_title", []))
    print_examples("ESEMPI - body vuoti", report["examples"].get("blank_body", []))

    # Se richiesto, stampa anche le distribuzioni.
    if PRINT_VALUE_DISTRIBUTIONS:
        print_distribution(
            "DISTRIBUZIONE language",
            report["language_distribution"],
            max_items=MAX_DISTRIBUTION_ITEMS,
        )
        print_distribution(
            "DISTRIBUZIONE source",
            report["source_distribution"],
            max_items=MAX_DISTRIBUTION_ITEMS,
        )
        print_distribution(
            "DISTRIBUZIONE is_duplicate",
            report["duplicate_distribution"],
            max_items=MAX_DISTRIBUTION_ITEMS,
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Punto di ingresso principale.

    Questa versione valida i dati presenti in staging in modalità streaming,
    quindi è adatta anche a milioni di record.
    """
    # Apre la connessione al database.
    with get_connection() as conn:
        # Conferma che la connessione è stata aperta.
        print("Connessione al database riuscita.")

        # Verifica che le tabelle richieste siano presenti.
        check_schema_and_tables(conn)

        # Conta i record presenti in staging da validare.
        total_in_staging = get_staging_row_count(
            conn,
            source_file_filter=SOURCE_FILE_FILTER,
        )

        # Se non ci sono record da validare, termina subito.
        if total_in_staging == 0:
            print("La staging table non contiene record da validare.")
            return

        # Stampa quanti record saranno analizzati.
        print(f"Record in staging da validare: {total_in_staging:,}")

        # Se attivo, stampa il filtro per file sorgente.
        if SOURCE_FILE_FILTER is not None:
            print(f"Filtro _source_file attivo: {SOURCE_FILE_FILTER}")

        # Legge i valori ammessi di source dalla tabella news_sources.
        allowed_sources = get_allowed_sources(conn)

        # Avvia la validazione streaming con contatore di progresso.
        report = stream_validate_staging(
            conn=conn,
            allowed_sources=allowed_sources,
            source_file_filter=SOURCE_FILE_FILTER,
        )

        # Stampa il report finale.
        print_validation_report(report)


# Esegue main solo se il file viene lanciato direttamente.
if __name__ == "__main__":
    main()