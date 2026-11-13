# Controllo di replicabilità – Tesi Valleri (ECB Communications as Monetary Shocks)

Legenda: **[doc]** = scritto nei documenti/file; **[inferenza]** = mia deduzione.
Fonti: tesi (`papers/ECB Communications...pdf`), cartella dati OneDrive `ecb-communications/Tesi Valleri/`, repository GitHub `enricovalleri-01/ECB-monetary-surprises-extraction-llm-` (snapshot `git-hub/ECB-monetary-surprises-extraction-llm--main.zip`, coerente con GitHub).

## 1. Esito sintetico
- **Dati**: quasi tutti presenti; manca la parte macro grezza (HICP = Harmonised Index of Consumer Prices, PIL, disoccupazione) e il codice che costruisce `lp_dataset.csv`.
- **Codice su GitHub**: sì, due notebook (`ECB_Pipeline_MASTER.ipynb`, `ECB_LocalProjections.ipynb`), ma con problemi che impediscono l'esecuzione "as is".
- **Stadio Local Projections (LP)**: replicato esattamente a partire da `data/lp_dataset.csv` (numeri della Tabella 5.1 ritrovati) [doc: rieseguito].
- **Stadio LLM (Gemini 2.5 Flash)**: replicabile solo in modo approssimato (modello ospitato, chiave API vuota nel codice, ~15 ore di calcolo) [inferenza].

## 2. Dati: cosa c'è
| Elemento | Stato | Fonte |
|---|---|---|
| Accounts (90 PDF), Economic Bulletins (87), Monthly Bulletins (192), Press conferences (267 HTML) | presenti; tutti i 312 meeting mappabili a documenti (verifica su campione per i file non aperti) | `dati/` |
| Speeches BCE | 2.995 righe, 147 senza testo → 2.848 usate | `dati/all_ECB_speeches.csv` |
| Decisioni di policy 1999–2025 | presenti (313 righe; nome file corrotto) | `data/` repo |
| Serie delle sorprese (312×23), metadata JSON con output degli agenti | presenti | `data/` repo |
| Dataset LP (312×27) e risultati | presenti | `data/` repo |
| Serie macro grezze + codice di merge | **mancanti** | – |

Correzione a un mio rilievo precedente: gli Economic Bulletins 2015–2025 sono completi; il "bollettino mancante autunno 2025" era un'inferenza errata.

## 3. Codice: cosa c'è e cosa non funziona
- Errore di sintassi: `SPEAKER_ROLES = {` duplicato (cella 5 nel repo, cella 6 nella versione vecchia).
- Percorsi fissi Colab/Drive (`/content/drive/MyDrive/ECB_Research/ECB_Project`); `GEMINI_API_KEY = ''`.
- `filter_speeches` legge `row['speaker']`, ma la colonna del CSV è `speakers` e non c'è alcun rename: **probabilmente** tutti i discorsi ricevono il peso di default 0,40, rendendo inefficaci i pesi per ruolo descritti in tesi [inferenza; non verificabile senza l'esecuzione originale]. Il matching dei nomi (`name.lower() in speaker.lower()`) fallisce inoltre per alcuni nomi.
- README non allineato al contenuto.
- La pagina `data/` di GitHub non è consultabile direttamente (robots.txt); il suo contenuto è dedotto dallo zip.
- Lo zip `...-pipeline-main.zip` (23 maggio) è una versione più vecchia; quella coerente con la tesi è quella del 5 luglio.

## 4. Discrepanze tesi ↔ dati/codice
1. **Segno del coefficiente "dovish"**: la convenzione `shock_neg = surprise.clip(upper=0)` rende l'interpretazione del coefficiente incoerente con il testo.
2. **Accuracy modale**: ricalcolo 77% (pre-2015: 76,9%) contro il 68% in tesi; la strategia "sempre hold" darebbe 81,4% (tesi: "circa metà" hold).
3. **Controllo `gdp_gap`** nelle LP non compare nell'equazione 5.1 della tesi.
4. **Discorsi per meeting**: 5,2 (campo `speech_n_processed` del JSON) e 5,6 (ricalcolo dal CSV, finestra [t−28g, t−10g]) contro ≈8 dichiarati; 20 riunioni senza alcun discorso.
5. **Riunione del 22/01/2015** assente dalla serie (312 su 313 decisioni).
6. **Date dei bollettini**: nominali (giorno 28 in 230 casi su 312), non la data reale di pubblicazione.
7. **Timing HICP** rispetto al mese del meeting.
8. **Metadata JSON vs CSV (nuovo)**: le aspettative `expected_change_bp` nel JSON e nel CSV delle sorprese coincidono dal 2015 in poi, ma **differiscono in quasi tutti i meeting 1999–2014**, fino a 14,6 pb (es. 2014-11-06: CSV −11,58 vs JSON −1,94; 2005-04-07: 13,31 vs −1,28). Nel CSV i valori hanno multipli di 0,0025, compatibili con una media di 4 esecuzioni [inferenza]. Non è chiaro quale versione abbia prodotto i numeri in tesi.
9. Statistiche della serie (DFR, sorprese meccaniche, pb): media −1,12, dev. std 13,54; era Duisenberg/Trichet −5,53, Draghi +6,54, Lagarde +3,81. Differenza pre/post 2015: −3,49 vs +5,00 (t = −5,17, p < 0,001) [doc: calcolato da `surprises_timeseries.csv`]. Aritmetica dell'Agent III (atteso dalla distribuzione posteriore) verificata: errore 0.

## 5. Per replicare davvero servirebbe
Download delle serie ECB Data Portal (HICP, PIL, output gap, disoccupazione) e script di merge; chiave API Gemini; correzione del bug di sintassi e della colonna `speakers`; chiarimento dall'autore su: versione dei dati (JSON vs CSV), convenzione di segno, esclusione di gennaio 2015, pesi dei relatori.

## 6. Codice usato per questo controllo
Vedi `code/README.md` (script 01–05, rieseguiti il 2026-11-13).
