L’obiettivo è ottenere un corpus affidabile, eliminando i duplicati e verificando la qualità dei dati.

Ordine di esecuzione:
1. load_news_step1.py : carica gli articoli originali nella tabella temporanea del database PostgreSQL.
2. check_staging_table_v2.py : verifica la correttezza dei dati importati
3. load_news_step2_v1.py : normalizzazione e trasferimento degli articoli nella tabella definitiva.
4. deduplicate_news_minhash_v2.py : dentifica gli articoli quasi duplicati attraverso MinHash e Locality-Sensitive Hashing (LSH), utilizzando una verifica finale basata sulla similarità di Jaccard.

Gli altri script sono stati utilizzati per verificare la qualità del processo e non costituiscono passaggi obbligatori della pipeline principale:

- check_duplicate.py: controlla le informazioni sui duplicati presenti nei dati originali.
- inspect_preprocessing_anomalies.py: individua eventuali anomalie nei testi e nei dati preprocessati.
- audit_preprocessing_quality.py: verifica la qualità complessiva del corpus e la gestione dei duplicati.
- audit_news_pipeline.py: esegue un controllo generale delle tabelle e dei risultati del preprocessing.
