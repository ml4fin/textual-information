Il processo inizia con `prepare_supervised_cv.py`, che prepara il dataset annotato per la validazione incrociata.

Gli script `train_supervised_baseline.py`, `train_supervised_v2_theme_interactions.py`,
`train_final_supervised_model.py`, `tune_supervised_model.py` e `tune_supervised_threshold.py` documentano lo sviluppo delle prime versioni del modello e la ricerca delle configurazioni più appropriate.

La versione V3 è stata sviluppata attraverso `train_supervised_v3_hybrid.py`, che introduce una rappresentazione ibrida delle caratteristiche testuali.

Successivamente, `tune_supervised_v3_threshold.py` è stato utilizzato per selezionare la soglia decisionale, mentre `train_final_supervised_v3_candidate.py` ha permesso di addestrare il modello candidato definitivo.

Prima dell'applicazione al corpus, lo script `score_supervised_v3_diagnostic_1000.py` è stato utilizzato per verificare il comportamento del modello su un campione di articoli.

Infine, `freeze_supervised_v3.py` ha permesso di validare e congelare il modello e la relativa configurazione.

Per ricostruire il modello definitivo, l'ordine principale è:
1. `prepare_supervised_cv.py`: prepara il dataset di sviluppo.
2. `train_supervised_v3_hybrid.py`: sviluppa e valuta il modello V3.
3. `tune_supervised_v3_threshold.py`: seleziona la soglia decisionale.
4. `train_final_supervised_v3_candidate.py`: addestra il modello candidato.
5. `freeze_supervised_v3.py`: valida e congela il modello definitivo.
6. `score_supervised_v3_full_corpus.py`: applica il modello congelato all'intero corpus di articoli validi.
