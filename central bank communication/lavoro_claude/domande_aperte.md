# Domande aperte sulla tesi Valleri (da risolvere nei lavori successivi)

Documento per il prossimo laureando. Non sono domande rivolte all'autore: sono punti che chi riprende il lavoro deve risolvere, ricostruendo i fatti dai dati e dal codice. Per ciascuno: **cosa non torna**, **dove guardare**, **come risolverlo**. Origine delle evidenze: `controllo_replicabilita.md`, `inventario.md` §8, script in `code/`. Ciò che è inferenza è marcato **[inferenza]**.

Sigle: BCE = Banca Centrale Europea; LP = Local Projections; HICP = Harmonised Index of Consumer Prices (indice armonizzato dei prezzi al consumo); DFR = Deposit Facility Rate (tasso sui depositi presso la BCE); HAC = heteroskedasticity and autocorrelation consistent (errori standard robusti); LLM = Large Language Model; API = Application Programming Interface; pb = punti base.

## A. Quale versione dei dati ha prodotto i numeri della tesi?
1. **JSON e CSV delle aspettative differiscono prima del 2015.** `expected_change_bp` in `ecb_surprises_metadata.json` e in `surprises_timeseries.csv` coincide dal 2015 ma differisce in 225 riunioni su 312 (1999–2014), fino a 14,6 pb (es. 2014-11-06: −11,58 vs −1,94). Quale dei due file è l'input delle LP e delle statistiche in tesi? *Dove*: `code/03_surprise_stats.py` → `out/json_vs_csv_expected.csv`. *Come*: ricalcolare la media delle sorprese con entrambe le versioni e vedere quale dà −1,1 pb (valore in tesi).
2. **I valori del CSV sono multipli di 0,0025.** Sono la media di 4 esecuzioni dell'LLM? **[inferenza]** *Come*: cercare nel notebook della pipeline un ciclo su più run/seed; altrimenti rieseguire 4 volte un sottoinsieme di riunioni.
3. **Z1 e Z2 (due repository) danno risultati diversi** (media −1,12 vs −1,28 pb) ma `lp_results_hicp.csv` è identico. Le LP sono state rigenerate dopo l'ultima esecuzione della pipeline? *Come*: rieseguire `code/01` su `lp_dataset.csv` di Z2 e confrontare con quello di Z1.
4. **Il file `ecb_policy_decisions_FULL_1999_2025.csv(`** ha il nome corrotto e 313 righe (una in più delle sorprese). Quale è l'estrapolazione ufficiale dei tassi e da dove proviene?

## B. Costruzione delle variabili macro (non ricostruibile oggi)
5. **Nessun codice o dato grezzo per HICP, PIL, output gap, disoccupazione.** `lp_dataset.csv` è un risultato finale. *Come*: scaricare le serie dall'ECB Data Portal, ricostruire `lp_dataset.csv` e confrontare colonna per colonna.
6. **Timing HICP rispetto alla riunione**: il valore `hicp_yoy` del mese della riunione è già noto al momento dello shock? Verificare allineamento mensile (publication lag) per evitare look-ahead.
7. **`gdp_gap` tra i controlli** (`['hicp_yoy','gdp_yoy','gdp_gap']`, lags=3) non compare nell'equazione 5.1 della tesi. Quale è corretta? Come si ottiene l'output gap (filtro? stima ex post?) e c'è look-ahead?

## C. Convenzioni e specificazione delle Local Projections
8. **Segno del coefficiente dovish.** `shock_neg = surprise.clip(upper=0)` assume valori ≤ 0, quindi β⁻ = +0,480 (h=9) significa che uno shock dovish *abbassa* l'HICP (stesso segno dell'effetto hawkish: −0,857). La tesi lo interpreta come "risposta più debole" usando i moduli (rapporto 1,8:1). Chiarire l'interpretazione economica e se è un puzzle (price puzzle?). *Dove*: `code/01`.
9. **Test di Wald** (R = [1, −1]): con shock dovish ≤ 0 verifica β⁺ = β⁻ che è l'ipotesi di linearità corretta; con la convenzione in modulo servirebbe R = [1, 1]. Verificare che il testo di tesi sia coerente.
10. **Test multipli**: 25 orizzonti × test di Wald non indipendenti; solo 14 su 25 con p < 0,10 e nessuna correzione. Servono controllo del false discovery rate o test congiunto.
11. **HAC con lunghi orizzonti** (maxlags = max(h,1)): l'inferenza a h > 12 è affidabile con ~200 osservazioni? Provare Newey-West con bandwidth più ampie e bootstrap.
12. **Campione effettivo**: 312 riunioni vs 324 mesi nel dataset mensile, nobs 177–224 per orizzonte (disoccupazione mancante in 20 righe). Quali mesi mancano e perché?

## D. Valutazione dell'LLM
13. **Accuratezza della moda**: ricalcolo 77,2% (pre-2015 76,9%) vs 68% in tesi; la baseline "sempre hold" farebbe 81,4%. Come si definisce l'accuratezza (bin diversi? esclusi i 5 esiti intermedi)? Il modello batte la baseline? *Dove*: `code/04`.
14. **Quota di "hold" ex post 81,4%**, mentre la tesi parla di "circa la metà": a quale quota si riferisce?
15. **Outcome leakage / look-ahead**: Gemini 2.5 Flash conosce gli esiti dei tassi BCE? Test con documenti pre-cutoff vs post-cutoff, o con modelli cronologicamente coerenti (cfr. Didisheim et al., 2026).
16. **Rieseguibilità**: la pipeline con `temperature=0.1` non è deterministica. Quanto varia la serie delle sorprese tra esecuzioni? Serve un esperimento con N run per quantificare la varianza e confrontarla con la dimensione delle sorprese.

## E. Corpus e pre-processing
17. **Pesi dei relatori.** Il codice legge `row['speaker']` ma il CSV ha `speakers` (ricalcolo: 10 nomi su 29 non riconosciuti per accenti/iniziali). **[inferenza]** Tutti i discorsi hanno peso di default 0,40. *Come*: rieseguire `weighted` su un sottoinsieme con colonna rinominata e vedere se cambia la serie.
18. **Discorsi per riunione**: 5,2 (JSON) o 5,6 (ricalcolo) vs ≈8 in tesi; 20 riunioni senza discorsi; 147–148 righe senza testo; mancano i governatori delle banche centrali nazionali. Quale è la finestra corretta (28 giorni prima, blackout di 10 giorni)?
19. **Date dei bollettini nominali** (giorno 28 in 230 casi su 312): usare la data reale di pubblicazione per evitare look-ahead.
20. **Break di gennaio 2015** (Accounts invece di Intro Statements): la differenza di media pre/post (−3,49 vs +5,00 pb, t = −5,2) può essere dovuta al cambio di fonte, non alla politica. Come separare fonte, ZLB (zero lower bound) e QE (quantitative easing)? Una prova: applicare la pipeline Accounts anche a un periodo pre-2015 se disponibile.
21. **Riunione del 22/01/2015** esclusa dalla serie (312 su 313): reinserirla ora che l'Account è disponibile (file senza data nel nome).
22. **Nomi anomali nei PDF Accounts** (`22-05-2020`, `25-06-2020`, `28-08-25.pdf` che contiene la riunione di giugno 2025, `28-08-25 p2`): mappa nome→riunione corretta.
23. **Regola di estrazione dell'Intro Statement** (frase "at your disposal for questions"): fallisce per 1999-01-07 e in formati più recenti? Misurare quante estrazioni sono vuote o troncate.

## F. Replicabilità del software
24. **Errore di sintassi** `SPEAKER_ROLES = {` duplicato, percorsi Colab/Drive fissi, `GEMINI_API_KEY` vuota, README non allineato (cita `mlf_surprise_mech`, assente). Rendere la pipeline eseguibile con un `requirements.txt`, percorsi relativi e variabili d'ambiente.
25. **Test di regressione**: dopo ogni modifica, confrontare le uscite con `lp_results_hicp.csv` (script `code/01`).
26. **Contenuto della pagina `data/` di GitHub** non verificato direttamente (blocco robots.txt): confrontarla con Z1.

## G. Estensioni da non confondere con la replica
27. Benchmark di mercato per la BCE (Euribor caplet); panel core/periferia; entropia posteriore (trasparenza Draghi vs Lagarde); filone embedding (Didisheim et al.); altre banche centrali. Da affrontare solo dopo aver chiuso i punti A–F.

## Ordine di lavoro suggerito
A1 → A3 (quale versione) → B5 (dati macro) → C8–C9 (convenzione di segno) → D13 (baseline) → E17–E19 (corpus) → F24 (codice eseguibile).
