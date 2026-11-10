Lo script `novelty_calibration_benchmark_v2.py` è stato utilizzato per confrontare diverse configurazioni dell'indicatore, considerando finestre temporali di 30, 90 e 365 giorni e differenti numeri di vicini semantici.

Gli script `faiss_hnsw_smoke_test_v1.py`, `novelty_faiss_benchmark_v1.py`, `novelty_faiss_multitarget_benchmark_v1.py`, `novelty_faiss_temporal_benchmark_v1.py` e `novelty_faiss_filter_depth_benchmark_v2.py` documentano i test effettuati per valutare l'efficienza e l'accuratezza della ricerca dei vicini attraverso FAISS.

Infine, `novelty_semantic_audit_v1.py` e `novelty_temporal_stability_v1.py` sono stati utilizzati per verificare la coerenza semantica dell'indicatore e la sua stabilità nel tempo.


Riproduzione dei risultati finali:
Lo script principale è `novelty_faiss_production_v1.py`, che calcola l'indicatore di novelty per gli articoli del corpus utilizzando gli embeddings già disponibili nel database PostgreSQL.

Il file `library.py` è uno script ausiliario per il controllo delle librerie utilizzate e non fa parte della pipeline di calcolo dell'indicatore.
