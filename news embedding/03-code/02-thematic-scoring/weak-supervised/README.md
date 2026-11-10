Gli script `analyze_ai_context.py`, `analyze_ai_local_context.py`, `analyze_rule_discrimination.py`, `discover_missing_theme_signals.py`, `inspect_A_errors.py` e `inspect_cooccurrence_triggers.py` sono stati utilizzati per analizzare il comportamento delle regole, individuare possibili errori e migliorare il riconoscimento dei segnali tematici.

Gli script `simulate_cooccurrence_rules.py`, `simulate_rule_changes.py`, `simulate_strong_weak_signals.py`, `simulaye_ai_rules.py` e `compe_simulate_proximity_rules.py` documentano invece le simulazioni condotte durante lo sviluppo. Non sono necessari per riprodurre i punteggi finali.

La configurazione delle regole V3 è definita in `weak_supervision_v3_config.py`. 
Gli script `build_weak_v3_label_functions.py`, `build_weak_v3_continuous_evidence.py` e
`build_weak_v3_continuous.py` sono stati utilizzati per costruire e analizzare le evidenze tematiche sul dataset di sviluppo.

Successivamente, `compare_weak_v3_continuous_candidates.py` ha permesso di confrontare diverse formule di scoring continuo, mentre `freeze_weak_supervision_v3.py` ha documentato il congelamento della configurazione V3.

Per applicare il metodo definitivo al corpus, i file principali sono:
1. `weak_supervision_v3_config.py`: contiene le regole linguistiche e i segnali tematici della versione V3.
2. `weak_supervision_v3_continuous_config.py`: definisce i parametri della versione continua definitiva e la formula di scoring selezionata.
3. `validate_weak_v3_continuous_article_engine.py`: verifica che l'implementazione ottimizzata dello scoring riproduca i punteggi della configurazione congelata. È un controllo di validazione.
4. `score_weak_v3_continuous_full_corpus.py`: è lo script principale di produzione e applica il metodo Weak Supervision V3 Continuous a tutti gli articoli validi del corpus.
