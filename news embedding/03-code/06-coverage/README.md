Procedura di calcolo:

Il calcolo originale è stato effettuato direttamente in PostgreSQL attraverso query SQL.

La procedura ha previsto l'aggregazione mensile dei punteggi Weak Supervision positivi, il conteggio degli articoli validi per ciascun mese e il calcolo della Coverage normalizzata.

I risultati sono stati successivamente inseriti nella tabella `risk_emergence.news_indicators` mediante una transazione SQL con operazione `INSERT ... ON CONFLICT DO UPDATE`.

L'operazione ha aggiornato le osservazioni già presenti e inserito quelle mancanti, preservando i valori di Semantic Drift precedentemente calcolati.

