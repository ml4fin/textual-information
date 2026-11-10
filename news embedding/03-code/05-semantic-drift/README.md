Gli script `semantic_drift_centroids_pilot_v1.py` e `semantic_drift_centroids_percentile_pilot_v1.py` sono stati utilizzati per confrontare diverse strategie di costruzione dei centroidi tematici, considerando differenti criteri di selezione degli articoli.

Successivamente, `semantic_drift_weak_centroids_validation_v1.py`, `semantic_drift_centroid_split_half_validation_v1.py` e `semantic_drift_effective_support_validation_v1.py` hanno permesso di verificare la stabilità e l'affidabilità dei centroidi, anche in relazione al numero di articoli disponibili per ciascun tema e mese.

Gli script `semantic_drift_temporal_reference_pilot_v1.py` e `semantic_drift_reference_depth_validation_v1.py` sono stati utilizzati per confrontare diverse finestre temporali e verificare la disponibilità di osservazioni storiche sufficienti per costruire un riferimento affidabile.


Riproduzione dei risultati finali:

Lo script principale è `semantic_drift_production_v1.py`, che calcola l'indicatore di Semantic Drift utilizzando gli embeddings e i punteggi tematici già disponibili nel database PostgreSQL.
