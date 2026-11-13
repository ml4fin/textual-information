# Codice usato per ricostruire quantitativamente i risultati della tesi Valleri

Data: 2026-11-13. Gli script non modificano alcun file originale: leggono i dati e scrivono solo in `ECB_OUT` (default `./out`). Linguaggio: Python 3.11+. Dipendenze: `numpy`, `pandas`, `scipy`, `statsmodels`.
Contesto e risultati: `../controllo_replicabilita.md`, `../inventario.md` §8, `../domande_aperte.md`.

## Preparazione
1. Estrarre lo zip `Tesi Valleri/git-hub/ECB-monetary-surprises-extraction-llm--main.zip` (Z1, coerente con la tesi) in una cartella; contiene `data/` e due notebook.
2. Impostare le variabili d'ambiente (o modificare `_common.py`):
   - `ECB_REPO` = cartella estratta `ECB-monetary-surprises-extraction-llm--main`
   - `ECB_DATI` = `Tesi Valleri/dati` (serve `all_ECB_speeches.csv`; per lo script 02 anche le cartelle `accounts` e `press_conferences`, altrimenti usa gli elenchi in `liste/`)
   - `ECB_OUT` = cartella di output (default `./out`)
3. Eseguire da questa cartella: `python 01_lp_replication.py` ecc. (ognuno è indipendente).

## File
| File | Cosa fa | Input | Output | Esito 2026-11-13 |
|---|---|---|---|---|
| `_common.py` | Percorsi comuni (variabili d'ambiente) | – | – | – |
| `01_lp_replication.py` | Riesegue le celle 1, 3 e 6 del notebook `ECB_LocalProjections.ipynb` (caricamento, dataset mensile, LP congiunta hawkish/dovish con HAC e Wald su R = [1, −1], controlli `hicp_yoy`, `gdp_yoy`, `gdp_gap`, 3 ritardi) e confronta con `lp_results_hicp.csv` | `data/lp_dataset.csv`, notebook LP, `data/lp_results_hicp.csv` | `out/lp_rerun.csv`, `out/monthly.pkl`, stampa | **Replica esatta**: differenze ≈ 1e-15 su β⁺, β⁻, p di Wald; nobs identici; h=9: β⁺ = −0,857, β⁻ = +0,480, p = 0,0089, nobs = 212; 14 orizzonti con p < 0,10 |
| `02_coverage_check.py` | Mappa le 312 riunioni ai documenti dichiarati dalla pipeline (Accounts post-2015, conferenze stampa pre-2015), esamina le date nominali dei bollettini e ricalcola i discorsi nella finestra [t−28g, t−10g] confrontandoli con `speech_n_processed` | `data/ecb_surprises_metadata.json`, `all_ECB_speeches.csv`, elenchi file (`liste/` o cartelle) | `out/cov.pkl`, stampa | Pre-2015: 0 documenti mancanti; post-2015: 86/87 Accounts trovati per data (1 = file senza data nel nome); bollettini giorno 28 in 230/312; discorsi 5,19 (JSON) vs 5,63 (ricalcolo), 20 riunioni senza discorsi |
| `03_surprise_stats.py` | Statistiche della sorpresa DFR per sotto-periodo, test pre/post 2015 (t, Levene), verifica aritmetica dell'Agent III (E[Δi] = Σ p·bin) e confronto JSON ↔ CSV delle aspettative | `data/surprises_timeseries.csv`, `data/ecb_surprises_metadata.json` | `out/json_vs_csv_expected.csv`, stampa | Media −1,12 pb, sd 13,54; pre/post −3,49 vs +5,00 (t = −5,17); errore aritmetico 0; JSON ≠ CSV in 225/312 riunioni (1999–2014), max 14,59 pb |
| `04_modal_accuracy_baseline.py` | Accuratezza della moda a priori (`modal_outcome`) rispetto al bin dell'esito effettivo, quota di hold e baseline "sempre hold" | `data/surprises_timeseries.csv` | stampa | Accuratezza 77,2% (pre 76,9%, post 78,2%); hold effettivi 81,4%; 5 esiti non classificabili nei 5 bin |
| `05_speaker_weight_check.py` | Analisi **statica** del notebook della pipeline (non lo esegue): righe con `speaker`, colonne del CSV, nomi non riconosciuti dalla regola `name.lower() in speaker.lower()` | `ECB_Pipeline_MASTER.ipynb`, `all_ECB_speeches.csv` | stampa | Il codice usa `speaker`, il CSV ha `speakers`; 10 nomi su 29 non riconosciuti **[inferenza: il bug va confermato eseguendo il pipeline]** |
| `liste/accounts_filenames.txt` | Elenco dei 90 nomi file degli Accounts (senza estensione) | estratto dall'elenco OneDrive del 2026-10-01 | – | – |
| `liste/pressconf_dates.txt` | Date (AAAAMMGG) delle 267 conferenze stampa | idem | – | – |

## Cosa NON è stato ricostruito (e perché)
- **Stadio LLM** (Agent IM, IB: chiamate a Gemini 2.5 Flash): richiede chiave API e ~15 ore **[inferenza]**; non rieseguito. Si è verificato solo l'aritmetica degli Agent II/III dagli output salvati.
- **Costruzione di `lp_dataset.csv`** (merge HICP, PIL, disoccupazione): codice e dati grezzi assenti.
- **Figure e tabelle descrittive del cap. 4** della tesi: non ricalcolate.
- Gli indici delle celle in `01` (1, 3, 6) valgono per il notebook di Z1; con Z2 vanno ricontrollati.

## Note di riproducibilità
- Gli script sono stati rieseguiti il 2026-11-13 in un ambiente con `pandas`, `scipy`, `statsmodels`. Se i numeri cambiano con altre versioni delle librerie, confrontare con `lp_results_hicp.csv`.
- I file in `liste/` servono solo se le cartelle `accounts` e `press_conferences` non sono scaricate (i file OneDrive sono "solo cloud").
