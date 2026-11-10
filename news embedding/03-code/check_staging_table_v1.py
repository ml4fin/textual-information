# Importa il modulo standard per leggere file CSV.
import csv

# Aumenta in modo robusto il limite massimo per i campi CSV.
# Serve perché il campo Body può essere molto lungo.
def increase_csv_field_size_limit() -> None:
    """
    Aumenta il limite massimo consentito dal modulo csv per un singolo campo.
    """
    import sys

    # Parte dal massimo intero disponibile sulla piattaforma.
    max_int = sys.maxsize

    # Continua a ridurre il valore finché csv.field_size_limit lo accetta.
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)


# Applica subito l'aumento del limite per i campi CSV.
increase_csv_field_size_limit()

# Importa espressioni regolari per validare date e numeri.
import re

# Importa datetime per il parsing delle date.
from datetime import datetime

# Importa Path per gestire percorsi file in modo robusto.
from pathlib import Path

# Importa tipi usati nelle annotazioni.
from typing import Any, Dict, List, Optional

# Importa psycopg2 per la connessione a PostgreSQL.
import psycopg2


# ============================================================================
# CONFIGURAZIONE DEBUG / SVILUPPO
# ============================================================================

# Flag informativo: lo script è pensato per essere lanciato da IDE.
RUN_FROM_IDE = True

# Se valorizzato, limita la validazione ai record di questo file sorgente.
# Esempio: "FTXX_2014_2023.csv"
SOURCE_FILE_FILTER = None

# Numero massimo di esempi problematici da stampare per ciascuna categoria.
MAX_EXAMPLES_PER_CATEGORY = 5

# Se True, stampa anche le distribuzioni dei valori distinti.
PRINT_VALUE_DISTRIBUTIONS = True

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
    # Restituisce una connessione usando i parametri configurati.
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

    # Converte il risultato in un insieme di tuple (schema, tabella).
    found = {(r[0], r[1]) for r in rows}

    # Definisce l'insieme delle tabelle richieste.
    required = {
        (SCHEMA_NAME, "news_articles_staging"),
        (SCHEMA_NAME, "news_articles"),
        (SCHEMA_NAME, "news_sources"),
    }

    # Individua le tabelle mancanti.
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
    # Query base per contare le righe in staging.
    query = f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{STAGING_TABLE}"
    params: tuple = ()

    # Se è richiesto un filtro sul file sorgente, lo aggiunge alla query.
    if source_file_filter is not None:
        query += " WHERE _source_file = %s"
        params = (source_file_filter,)

    # Esegue la query di conteggio.
    with conn.cursor() as cur:
        cur.execute(query, params)
        count = cur.fetchone()[0]

    # Restituisce il numero totale di record.
    return count


def get_staging_records(conn, source_file_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Legge i record dalla staging table e li restituisce come lista di dizionari.
    """
    # Query per leggere i campi rilevanti ai fini della validazione.
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

    # Se richiesto, limita i record a un file sorgente specifico.
    if source_file_filter is not None:
        query += " WHERE _source_file = %s"
        params = (source_file_filter,)

    # Ordina i record per file e numero riga per una lettura più intuitiva.
    query += " ORDER BY _source_file, _raw_line_number"

    # Apre un cursore per leggere i dati.
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    # Costruisce i dizionari record per record.
    records = []
    for row in rows:
        records.append(
            {
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
        )

    # Restituisce la lista completa dei record letti.
    return records


def get_allowed_sources(conn) -> set:
    """
    Legge i valori ammessi per le fonti dalla tabella news_sources.

    La funzione prova prima a identificare automaticamente la colonna
    che contiene il dominio sorgente; in caso contrario usa una lista
    di possibili nomi.
    """
    # Legge le colonne della tabella news_sources.
    query_cols = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position;
    """

    with conn.cursor() as cur:
        cur.execute(query_cols, (SCHEMA_NAME, SOURCES_TABLE))
        cols = [r[0] for r in cur.fetchall()]

    # Possibili nomi candidati per la colonna che contiene il dominio/raw source.
    candidate_columns = [
        "source_domain",
        "domain",
        "source_raw",
        "source_code",
        "source_name",
    ]

    # Seleziona la prima colonna candidata effettivamente presente.
    selected_col = None
    for candidate in candidate_columns:
        if candidate in cols:
            selected_col = candidate
            break

    # Se non trova nessuna colonna attesa, interrompe l'esecuzione.
    if selected_col is None:
        raise RuntimeError(
            "Impossibile identificare la colonna della tabella news_sources "
            f"che contiene il valore sorgente. Colonne trovate: {cols}"
        )

    # Query che legge i valori distinti ammessi dalla tabella news_sources.
    query_values = f"""
    SELECT DISTINCT {selected_col}
    FROM {SCHEMA_NAME}.{SOURCES_TABLE}
    WHERE {selected_col} IS NOT NULL
    """

    with conn.cursor() as cur:
        cur.execute(query_values)
        rows = cur.fetchall()

    # Normalizza i valori in lowercase e strip per confronto robusto.
    allowed_sources = {str(r[0]).strip().lower() for r in rows if r[0] is not None}

    # Stampa quale colonna è stata utilizzata.
    print(f"Colonna usata in news_sources per il matching: {selected_col}")

    # Restituisce l'insieme dei valori sorgente ammessi.
    return allowed_sources


# ============================================================================
# FUNZIONI DI VALIDAZIONE SINGOLO CAMPO
# ============================================================================

def is_blank(value: Optional[str]) -> bool:
    """
    Verifica se un valore è nullo, vuoto o composto solo da spazi.
    """
    # True se il valore è None.
    if value is None:
        return True

    # True se, dopo strip, la stringa è vuota.
    return str(value).strip() == ""


def is_valid_date_string(value: Optional[str]) -> bool:
    """
    Verifica se la data è parseabile in uno dei formati attesi.
    """
    # Se il valore è vuoto, la data non è valida.
    if is_blank(value):
        return False

    # Normalizza il valore in stringa senza spazi esterni.
    text = str(value).strip()

    # Prova i formati più probabili per i CSV di news.
    candidate_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]

    # Tenta il parsing con ciascun formato.
    for fmt in candidate_formats:
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            pass

    # Se nessun formato funziona, restituisce False.
    return False


def is_valid_numeric_string(value: Optional[str]) -> bool:
    """
    Verifica se il valore può essere convertito in numero float.
    """
    # Valori vuoti sono considerati validi come "missing", non come errore.
    if is_blank(value):
        return True

    # Normalizza la stringa.
    text = str(value).strip()

    # Prova la conversione a float.
    try:
        float(text)
        return True
    except ValueError:
        return False


def is_valid_duplicate_flag(value: Optional[str]) -> bool:
    """
    Verifica se il campo is_duplicate contiene un valore interpretabile come booleano.
    """
    # I valori vuoti sono ammessi in questa fase.
    if is_blank(value):
        return True

    # Normalizza il valore.
    text = str(value).strip().lower()

    # Insieme dei valori accettati.
    valid_values = {"true", "false", "t", "f", "1", "0", "yes", "no"}

    # Restituisce True se il valore è riconosciuto.
    return text in valid_values


def looks_english(value: Optional[str]) -> bool:
    """
    Verifica se il campo language suggerisce che il record sia in inglese.
    """
    # Un valore vuoto non è inglese.
    if is_blank(value):
        return False

    # Normalizza il valore.
    text = str(value).strip().lower()

    # Valori tipici considerati inglese.
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


def print_examples(title: str, examples: List[dict]) -> None:
    """
    Stampa gli esempi di record problematici per una categoria.
    """
    # Se non ci sono esempi, non stampa nulla.
    if not examples:
        return

    # Stampa un titolo leggibile.
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
            f"sentiment={ex['sentiment']}, "
            f"title_preview={repr((ex['title'] or '')[:60])}"
        )


def print_distribution(title: str, distribution: Dict[str, int]) -> None:
    """
    Stampa una distribuzione di frequenze ordinata per numerosità decrescente.
    """
    # Se la distribuzione è vuota, non stampa nulla.
    if not distribution:
        return

    # Stampa il titolo della distribuzione.
    print("")
    print(title)

    # Ordina per conteggio decrescente e poi per chiave.
    sorted_items = sorted(distribution.items(), key=lambda x: (-x[1], x[0]))

    # Stampa ogni coppia valore-conteggio.
    for key, count in sorted_items:
        print(f"  {repr(key)} -> {count}")


# ============================================================================
# VALIDAZIONE STAGING
# ============================================================================

def validate_staging_records(records: List[Dict[str, Any]], allowed_sources: set) -> Dict[str, Any]:
    """
    Esegue i controlli di qualità sui record della staging table.
    """
    # Numero totale di record analizzati.
    total_records = len(records)

    # Inizializza i contatori del report.
    report = {
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
    }

    # Dizionario per conservare esempi di record problematici.
    examples: Dict[str, List[dict]] = {}

    # Distribuzione dei valori distinti del campo language.
    language_distribution: Dict[str, int] = {}

    # Distribuzione dei valori distinti del campo source.
    source_distribution: Dict[str, int] = {}

    # Distribuzione dei valori distinti del campo is_duplicate.
    duplicate_distribution: Dict[str, int] = {}

    # Itera su tutti i record estratti dalla staging.
    for record in records:
        # Legge i campi principali del record.
        title = record["title"]
        body = record["body"]
        publication_date = record["publication_date"]
        language = record["language"]
        source = record["source"]
        sentiment = record["sentiment"]
        is_duplicate = record["is_duplicate"]

        # Normalizza alcuni valori per le distribuzioni.
        language_key = "" if language is None else str(language).strip()
        source_key = "" if source is None else str(source).strip()
        duplicate_key = "" if is_duplicate is None else str(is_duplicate).strip()

        # Aggiorna la distribuzione dei valori di language.
        language_distribution[language_key] = language_distribution.get(language_key, 0) + 1

        # Aggiorna la distribuzione dei valori di source.
        source_distribution[source_key] = source_distribution.get(source_key, 0) + 1

        # Aggiorna la distribuzione dei valori di is_duplicate.
        duplicate_distribution[duplicate_key] = duplicate_distribution.get(duplicate_key, 0) + 1

        # Controlla se la data è valida.
        if is_valid_date_string(publication_date):
            report["valid_dates"] += 1
        else:
            report["invalid_dates"] += 1
            add_example(examples, "invalid_dates", record)

        # Controlla se la source è mappabile rispetto a news_sources.
        source_norm = "" if source is None else str(source).strip().lower()
        if source_norm in allowed_sources and source_norm != "":
            report["mapped_sources"] += 1
        else:
            report["unmapped_sources"] += 1
            add_example(examples, "unmapped_sources", record)

        # Controlla se il sentiment è numerico o vuoto ammesso.
        if is_valid_numeric_string(sentiment):
            report["valid_sentiment"] += 1
        else:
            report["invalid_sentiment"] += 1
            add_example(examples, "invalid_sentiment", record)

        # Controlla se il flag is_duplicate è interpretabile.
        if is_valid_duplicate_flag(is_duplicate):
            report["valid_is_duplicate"] += 1
        else:
            report["invalid_is_duplicate"] += 1
            add_example(examples, "invalid_is_duplicate", record)

        # Conta i titoli vuoti.
        if is_blank(title):
            report["blank_title"] += 1
            add_example(examples, "blank_title", record)

        # Conta i body vuoti.
        if is_blank(body):
            report["blank_body"] += 1
            add_example(examples, "blank_body", record)

        # Conta se la lingua sembra inglese oppure no.
        if looks_english(language):
            report["english_like"] += 1
        else:
            report["non_english_like"] += 1

    # Aggiunge gli esempi al report finale.
    report["examples"] = examples

    # Aggiunge le distribuzioni al report finale.
    report["language_distribution"] = language_distribution
    report["source_distribution"] = source_distribution
    report["duplicate_distribution"] = duplicate_distribution

    # Restituisce il report completo.
    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    """
    Stampa in modo leggibile il report di validazione.
    """
    # Stampa intestazione.
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
    print_examples(
        "ESEMPI - date non valide",
        report["examples"].get("invalid_dates", []),
    )
    print_examples(
        "ESEMPI - source non mappate",
        report["examples"].get("unmapped_sources", []),
    )
    print_examples(
        "ESEMPI - sentiment non validi",
        report["examples"].get("invalid_sentiment", []),
    )
    print_examples(
        "ESEMPI - is_duplicate non validi",
        report["examples"].get("invalid_is_duplicate", []),
    )
    print_examples(
        "ESEMPI - title vuoti",
        report["examples"].get("blank_title", []),
    )
    print_examples(
        "ESEMPI - body vuoti",
        report["examples"].get("blank_body", []),
    )

    # Se richiesto, stampa anche le distribuzioni dei valori distinti.
    if PRINT_VALUE_DISTRIBUTIONS:
        print_distribution(
            "DISTRIBUZIONE language",
            report["language_distribution"],
        )
        print_distribution(
            "DISTRIBUZIONE source",
            report["source_distribution"],
        )
        print_distribution(
            "DISTRIBUZIONE is_duplicate",
            report["duplicate_distribution"],
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Punto di ingresso principale.

    Questa versione esegue i controlli di qualità sui dati già presenti
    nella staging table, senza ancora trasferirli nella tabella finale.
    """
    # Apre la connessione al database.
    with get_connection() as conn:
        print("Connessione al database riuscita.")

        # Verifica che le tabelle richieste siano presenti.
        check_schema_and_tables(conn)

        # Conta i record presenti in staging.
        total_in_staging = get_staging_row_count(
            conn,
            source_file_filter=SOURCE_FILE_FILTER,
        )

        # Se non ci sono record da validare, termina.
        if total_in_staging == 0:
            print("La staging table non contiene record da validare.")
            return

        # Stampa quanti record saranno analizzati.
        print(f"Record in staging da validare: {total_in_staging}")

        # Se attivo, stampa il filtro per file sorgente.
        if SOURCE_FILE_FILTER is not None:
            print(f"Filtro _source_file attivo: {SOURCE_FILE_FILTER}")

        # Legge i record dalla staging table.
        records = get_staging_records(
            conn,
            source_file_filter=SOURCE_FILE_FILTER,
        )

        # Legge i valori sorgente ammessi dalla tabella news_sources.
        allowed_sources = get_allowed_sources(conn)

        # Esegue la validazione dei record estratti.
        report = validate_staging_records(records, allowed_sources)

        # Stampa il report finale di validazione.
        print_validation_report(report)


# Esegue main solo se il file viene lanciato direttamente.
if __name__ == "__main__":
    main()