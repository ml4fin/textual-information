Lo script `evaluate_gold.py` è stato utilizzato per valutare le prestazioni rispetto al dataset annotato durante lo sviluppo.

Successivamente, `analyze_gold_errors.py` ha permesso di approfondire gli errori di classificazione, individuando i principali casi di falsi positivi e falsi negativi.

Lo script `evaluate_final_weak_supervision.py` documenta invece la valutazione della versione Weak Supervision V3 sul dataset di sviluppo.

La valutazione definitiva utilizza un insieme di articoli annotati manualmente, validato e congelato prima del confronto finale tra i metodi. Per riprodurre la valutazione, gli script principali sono:
1. `extract_final_test_existing_scores.py`: estrae dal database i punteggi Keyword-Based e Weak Supervision già calcolati per gli articoli del test set.
2. `score_final_test_supervised_v3.py`: applica il modello Supervised V3 Frozen agli articoli del test set.
3. `score_final_test_weak_v3_frozen.py`: produce le predizioni della versione Weak Supervision V3 congelata.
4. `evaluate_final_keyword_weak_v232.py`: valuta e confronta i risultati Keyword-Based e Weak Supervision.
5. `evaluate_final_supervised_v3.py`: calcola le metriche finali del modello supervisionato V3.
6. `evaluate_final_weak_v3.py`: valuta le predizioni della versione Weak Supervision V3 congelata.

Lo script `final_test_weak_continuous_evaluation.py` è dedicato alla valutazione specifica dei punteggi continui prodotti dalla Weak Supervision V3 e al loro confronto con le
annotazioni del test set finale.
