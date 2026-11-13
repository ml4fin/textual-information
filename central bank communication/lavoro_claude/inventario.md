# Inventario della cartella `03-llm-monetary-policy`

Data: 2026-10-01. Modalità: sola lettura. Nessun file originale è stato spostato, rinominato, modificato o eliminato. Unico file creato: questo inventario, in `lavoro_claude/`.

**Versione 2 (estesa).** Questa versione include anche le tre cartelle puntate dai collegamenti `.lnk` (accesso concesso il 2026-10-01). Le sezioni 3, 4, 5, 6 e 7 sono riscritte; §0, §1.2 e §2.1 sono aggiornati.

**Versione 3 (2026-11-13).** Aggiunta la §8 (verifica di replicabilità della tesi: dati, codice, discrepanze) e corretti alcuni punti precedenti (vedi §8.4). Il dettaglio è in `controllo_replicabilita.md`, le domande aperte in `domande_aperte.md`, gli script in `code/` (descritti in `code/README.md`).

Convenzioni:
- Ogni informazione è seguita dal file di provenienza tra parentesi quadre, es. `[P2]`.
- Ciò che è scritto nel documento è riportato senza marcatura. Le mie deduzioni sono marcate **[inferenza]**.
- "non indicato" = non ricavabile dal documento (o dalla parte che ho letto).
- Codici file: **P1** = `papers/Baglioni A. - Monetary Policy Implementation.pdf`; **P2** = `papers/ECB Communications as Monetary Shocks...pdf` (tesi Valleri); **P3** = `papers/Fernandez-Fuertes R. - Monetary Policy Shocks A New Hop LLM and Central Bank Communication.pdf`; **L1/L2/L3** = i tre collegamenti `.lnk`.

## 0. Come ho suddiviso il lavoro e limiti di copertura

La cartella è piccola (6 file, 1 sottocartella), quindi l'ho trattata in un solo passaggio: elenco ricorsivo, metadati e checksum (md5) dei PDF, estrazione del testo con `pdftotext` (tutti e tre i PDF hanno testo estraibile, nessuna scansione), lettura dei collegamenti `.lnk` come stringhe.

**Aggiornamento: la cartella collegata contiene solo 3 PDF e 3 collegamenti, ma i collegamenti puntano a tre cartelle che ora ho aperto** (E1 `01-embedding-regression`, E2 `ecb-communications`, E3 `News Embedding`; vedi §3). Complessivamente ho censito **658 file preesistenti**: 6 nella cartella collegata, 6 in E1, 5 in E3 e 641 in E2. Il grosso (E2) è il materiale della tesi di Valleri: corpus BCE, due repository di codice e un docx.

**Metodo di lavoro per sottocartella:** (1) cartella collegata: lettura dei 3 PDF; (2) E1 e E3: elenco, md5, lettura di abstract e prime pagine; (3) E2: elenco, conteggi per sottocartella, analisi completa del CSV dei discorsi, del docx e dei due zip; lettura a campione di HTML e PDF (i file OneDrive erano solo-cloud e ho dovuto scaricarne alcuni; vedi §3). I file di E2 non aperti sono stati comunque elencati e confrontati per nome e dimensione.

Profondità di lettura (non ho letto tutto parola per parola):
- **P3** (80 pp.): letti in dettaglio abstract, introduzione, §2.1–2.6 (parte), §3 Dati, §4.1, §4.3, §4.5 (inizio), §4.6 (inizio e Tabella 9), §5 Conclusione. Appendici e figure solo scorse (titoli e alcune tabelle).
- **P2** (64 pp.): letti abstract, indice, cap. 1, §2.3–2.5, cap. 3 (quasi intero), cap. 5, cap. 6 (intero). Cap. 4 (analisi descrittiva) e Appendice B (prompt) solo scorsi.
- **P1** (274 pp., libro): letti solo frontespizio e indice. Il contenuto dei capitoli non è stato letto.

---

## 1. PAPER

### 1.1 Fernández-Fuertes R. — *Monetary Policy Shocks: A New Hope. Large Language Models and Central Bank Communication* — `papers/Fernandez-Fuertes R. - ... .pdf` [P3]

| Campo | Contenuto |
|---|---|
| Tipo / argomento (1 riga) | PDF, working paper (80 pp., A4, testo estraibile, non protetto, 4.723.088 B). Costruzione di *surprise* di politica monetaria Fed da comunicazioni ufficiali tramite sistema multi-agente LLM [P3]. |
| Titolo | Monetary Policy Shocks: A New Hope – Large Language Models and Central Bank Communication [P3] |
| Autori | Rubén Fernández-Fuertes, Dept. of Finance, Bocconi University [P3] |
| Anno | 2025: "WORKING PAPER N. 257, OCTOBER 2025"; prima versione 12 ottobre 2025, questa versione 29 ottobre 2025 [P3]. Collana editoriale non indicata nel testo estratto; i ringraziamenti citano il Baffi Centre [P3] (**[inferenza]** serie Baffi/Bocconi). |
| Domanda di ricerca | Si possono costruire *surprise* di politica monetaria meno rumorose di quelle di mercato (alta frequenza) estraendo direttamente le aspettative condizionali dalle comunicazioni pubbliche della Fed, anziché "ripulire" ex post le misure di mercato? [P3, intro] |
| Banche centrali e periodo | Federal Reserve (FOMC). Periodo **riportato in modo incoerente nel paper**: 1996–2025 (§1, Tab. 1: 1996-01/2025-03, 265 riunioni); "256 riunioni (1996–2025)" (§1); "2000–2023" (§2.1, implementazione); "269 riunioni (1996–2025)" nelle figure IRF; "223 riunioni (1996–2023)" in Tab. 3 [P3]. Da chiarire. |
| Corpus di comunicazione | Beige Book (≈2 settimane prima della riunione), Minutes della riunione precedente (rilasciati ≈3 settimane dopo), comunicati FOMC storici (Statements) [P3, §2.1]. Non usa discorsi, conferenze stampa né Greenbook. |
| Metodo di misura | **LLM**: pipeline sequenziale a 4 agenti (Agent IM: Minutes → stance/forward guidance; Agent IB: Beige Book → score in [−1,1] per inflazione, occupazione, crescita, consumi; Agent II: sintesi → distribuzione di probabilità prior sulle azioni della Fed; Agent III: confronto con la decisione realizzata → *surprise* e "salience") [P3, §2]. Modello: GPT-4.1-mini (cutoff giugno 2024) [P3, §2.1]. Vincoli anti look-ahead: cutoff temporali per documento, ancoraggio temporale nei prompt, controlli automatici, test "out-of-knowledge-cutoff"; stabilità su 17 esecuzioni (dev. std. mediana cross-run 4,2 bp per aspettative e 5,4 bp per *surprise*) [P3, §2.6]. |
| Modello econometrico / previsione | (i) Regressione predittiva OLS di Δi_t su i_{t−1} e score del Beige Book (Tab. 1, 265 riunioni); (ii) regressioni di predicibilità su 6 predittori di Bauer & Swanson (Tab. 3, 223 riunioni); (iii) **Local projections** (Jordà 2005) con errori Newey-West; confronto OLS-LP su campione pre-crisi 1996:01–2008:10 (108 oss.) e **2SLP-IV** su campione pieno (tasso Fed strumentato dalla *surprise*); (iv) strategia di trading su spread curva 1M–10Y (regola: posizione = −segno(s) se |s| > 67° percentile mobile su 60 riunioni) [P3, §4]. |
| Variabili dipendenti | Δ tasso Fed (Tab. 1); *surprise* stessa (Tab. 3); nelle IRF: tasso Fed, disoccupazione, produzione industriale, S&P 500, livello prezzi/CPI, PIL reale, spread 5Y–1M e 10Y–1M, path atteso e term premium 5Y/10Y [P3, Fig. 8–11 e testo §1]. Elenco completo delle IRF: non verificato su tutte le figure. |
| Controlli | Replica la specifica di Aruoba & Drechsel (2024): 4 ritardi di produzione industriale, CPI, disoccupazione, excess bond premium, S&P 500, tasso Fed; 0 ritardi dello shock [P3, §4.5.1]. Per il trading: costi di transazione 5 bp/trade [P3, Tab. 9]. |
| Dati esterni di confronto | R&R aggiornati (Acosta 2023); surprise di mercato di Jarociński & Karadi; futures FF1–FF4, ED1–ED4, SOFR (LSEG DataScope, TickData; dati a 1 minuto, finestra 30'); shock ripuliti di Bauer & Swanson (2023a) e Miranda-Agrippino & Ricco (2021); macro da FRED [P3, §3]. |
| Risultato principale (3 righe) | 1) Le *surprise* narrative spiegano il 52% della variazione del tasso (contro 15–17% delle misure di mercato) e mostrano errore di misura minimo. 2) Una *surprise* restrittiva di 25 bp produce disinflazione persistente (nessun *price puzzle*), calo di PIL/produzione, aumento di disoccupazione e due canali sulla curva (compressione iniziale dello spread, poi steepening da term premium); bande di confidenza ampie. 3) Strategia flattener/steepener: rendimento cumulato 43,7% (intro) e totale 19,2%/26,1% con holding 180/300 giorni nella Tab. 9 "equal-weight benchmark", oltre il doppio delle alternative di mercato [P3]. **[inferenza]** 43,7% e Tab. 9 sono probabilmente varianti diverse (duration-hedged vs equal-weight): da verificare. |
| Limiti dichiarati | Non ho trovato una sezione "Limitations" nelle parti lette. Dichiarati nel testo: bande di confidenza ampie; campione pre-crisi di 108 oss. con potere statistico limitato e IV inaffidabile; predicibilità moderata delle *surprise* (R² = 0,164) attribuita a informazione di mercato nel blackout (S&P 500 = 72% della predicibilità), lasciata a ricerche future; variabilità stocastica dell'LLM (mitigata con 17 run); il modello è addestrato fino a metà 2024 [P3, §2.6, §4.3, §4.5.1]. |
| Incoerenze interne | Numero di riunioni (256/265/269/223), periodo (1996–2025 vs 2000–2023), cifra di rendimento (43,7% vs Tab. 9) [P3]. |

### 1.2 Altri paper (cartella E1, non di comunicazione delle banche centrali)

I due paper citati nelle istruzioni di progetto (**Chen et al.** e **Didisheim et al.**) e altri due (**Tang et al.**, **Wang et al.**) si trovano in E1 `01-embedding-regression` (e tre di essi anche in E3 come duplicati con nomi diversi). Sono lavori su embedding LLM di **notizie finanziarie e rendimenti azionari**, non su comunicazione di banche centrali: elencati al §4.2 con scheda breve (letti solo frontespizio e abstract). Non ho compilato per essi la scheda completa del §1.1.

---

## 2. TESI PRECEDENTI

### 2.1 Valleri E. — *ECB Communications as Monetary Shocks: A Multi-Agent LLM Framework and Asymmetric Transmission Analysis* — `papers/ECB Communications as Monetary Shocks...pdf` [P2]

Nota: sta nella cartella `papers/` ma è una **tesi magistrale** (M.Sc. Applied Economics and Markets, Università di Bologna, relatore Prof. Umberto Cherubini, a.a. 2025/2026, autore Enrico Valleri) [P2, frontespizio]. PDF 64 pp., LaTeX, testo estraibile, 927.504 B, creato il 27 giugno 2026 [P2, metadati].

| Campo | Contenuto |
|---|---|
| Domanda di ricerca | (a) Le comunicazioni ufficiali BCE permettono di costruire una misura di *surprise* "market neutral"? (b) La trasmissione sull'inflazione HICP è asimmetrica fra *surprise* restrittive (hawkish) e espansive (dovish)? [P2, abstract, cap. 1, §5.3] |
| Metodo | Estensione all'ECB del framework multi-agente di Fernández-Fuertes (2025) [P2, §3.1]. Pipeline a 4 stadi: due agenti LLM (IM: Accounts/Intro Statement; IB: Bollettini + discorsi) + due moduli deterministici (II: update softmax della prior; III: calcolo aritmetico delle surprise). LLM: Google Gemini 2.5 Flash, temperatura 0,1, chunk da 12.000 caratteri con overlap 600 [P2, §3.4–3.5]. Distribuzione su 5 bin {≥+50, +25, 0, −25, ≤−50} bp [P2, nota 2 p. 19]. Pesi a priori per ruolo dell'oratore (Presidente 1,00; VP 0,85; Chief Economist 0,80; Executive Board 0,70; Governatori NCB 0,55) [P2, §3.2]. Surprise calcolate per MRO, DFR, MLF; DFR è il tasso primario [P2, §3.3.1]. Econometria: **Local Projections** h=1..24 mesi con componenti hawkish/dovish separate, errori HAC Newey-West (bandwidth = h), test di Wald di simmetria β⁺ = −β⁻ [P2, §5.1]. |
| Dati | 312 riunioni del Consiglio direttivo, marzo 1999–dicembre 2025. Documenti: 90 PDF di Accounts (2015–2025); 267 trascrizioni HTML di conferenze stampa (1999–2026; 178 pre-2015), da cui si estrae l'Introductory Statement; 87 Economic Bulletin (2015–2025) e 192 Monthly Bulletin (1999–2014); 2.848 discorsi (1997–2026) dal database BCE, circa 8 per riunione dopo i filtri (lookback 28 giorni, blackout 10 giorni). Macro da ECB Data Portal: HICP (serie HICP.M.U2.N.000000.4D0.ANR), PIL (MNA.Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY, trimestrale con forward-fill), disoccupazione (LFSI.M.I10.S.UNEHRT.TOTAL0.15_74.T, da gen. 2000) [P2, §3.3]. |
| Risultati | Surprise con media vicina a zero (−1,1 bp), volatilità minima sotto Draghi, massime nei rialzi da 75 bp di settembre/ottobre 2022. A h=9 mesi una surprise hawkish di 25 bp riduce l'HICP di 0,857 pp, quella dovish la aumenta di 0,480 pp (≈56%, rapporto ≈1,8:1); Wald p = 0,009; asimmetria significativa al 10% in 14 orizzonti su 24 (concentrati in h=1–16) [P2, abstract, §5.2–5.3, Tab. 5.1]. Robustezza: con dummy post-2015 −0,910; escludendo 2022–2023 −1,103 [P2, §3.6, §5.2]. |
| Limiti dichiarati | (1) **Break strutturale a gennaio 2015** (Accounts al posto degli Intro Statement): F-test significativo (p=0,014/0,006; media pre −3,7 bp vs post +5,0 bp), non separabile dal cambio di regime (ZLB, QE). (2) Pesi di aggregazione a priori, non stimabili senza inferenza circolare. (3) **Bias temporale/outcome leakage** dell'LLM (Gemini addestrato su dati oltre il campione; non testabile senza un modello addestrato pre-1999). Inoltre: possibile *bad control* (PIL contemporaneo), HAC inaffidabile per h ≥ 18, stima "meeting-month" non pienamente mensile, i 24 test di Wald non sono indipendenti, bin che non coprono −10 bp e +75 bp (≤3 osservazioni), PIL in vintage rivisto, stime da leggere come associazioni condizionali e non come effetti causali [P2, §3.5, §5.1, §5.3, §6.2]. |
| Sviluppi futuri suggeriti | (i) Misura complementare di mercato dalle opzioni su Euribor (caplet, Black 1976; dati Puigvert-Gutiérrez & de Vincent-Humphreys 2010 per 1999–2010, Bloomberg/Refinitiv dopo) per scomporre la parte già prezzata; (ii) LP panel su paesi core vs periferici; (iii) analisi della trasparenza per era presidenziale tramite l'entropia posteriore di Agent II; (iv) specifica con PIL ritardato senza dummy; test congiunto di Wald o bande simultanee; (v) diagnosi di leakage via correlazione fra accuratezza modale e notorietà della riunione [P2, §3.5, §5.1, §5.3, §6.3]. |
| Parti riutilizzabili | Architettura e mappatura documenti per sotto-periodo (Tab. 3.1); prompt degli agenti IM e IB (Appendice B, p. 49–52, non letti per intero); regole di preprocessing (chunking, estrazione dell'Intro Statement tramite la frase "We are now at your disposal for questions"; correzione dei 11 casi di data effettiva dei tassi); tabelle di robustezza 3.2–3.4 e 5.2; checkpoint JSON per riunione [P2]. **Codice, notebook e dati NON sono nella cartella collegata, ma sono in E2 `ecb-communications/Tesi Valleri/`** (corpus, 2 zip di repository, serie derivate): vedi §3.1 e §4.1. P2 rinvia a un "replication code" [P2, nota 1 p. 34] e dichiara Python su Google Colab con documenti su Google Drive [P2, §3.5]. |
| Verifiche con i file di E2 | **Confermato:** 90 Accounts, 87 Economic Bulletin, 192 Monthly Bulletin, 267 conferenze (178 pre-2015), 312 riunioni, media −1,1 bp (coincide col repo Z1), LP h=9 −0,857/+0,480 (coincide con `lp_results_hicp.csv`). **Differenze:** 2.848 discorsi = 2.995 righe del CSV meno 147 senza testo (non dichiarato in P2); il CSV non contiene discorsi di governatori NCB, ma P2 assegna loro peso 0,55 [P2 §3.2; E2]; il repo Z2 (più vecchio) ha media −1,28 bp. Vedi §3.1, §4.1. |
| Incoerenze interne | Abstract/§3.4: pipeline con "due agenti LLM + due moduli deterministici", mentre P3 usa quattro agenti LLM; ciò è una scelta dichiarata dalla tesi, non un errore. "267 trascrizioni 1999–2026" a fronte di campione che termina a dicembre 2025. 90 Accounts vs 87 riunioni post-2015 (spiegata nel testo). Tab. 2.1 attribuisce a P3 il periodo 2000–2024, mentre P3 dichiara 1996–2025 [P2, P3]. |

Altre tesi precedenti: **nessuna** trovata nella cartella. Il file P2 è l'unica.

---

## 3. CARTELLE DI DATI E CARTELLE COLLEGATE (estensione ai `.lnk`)

Le tre cartelle puntate dai collegamenti sono state aperte (accesso concesso il 2026-10-01). Percorso breve: **E1** = `01-embedding-regression` (Google Drive, sorella della cartella collegata); **E2** = `ecb-communications` (OneDrive di ateneo, contiene solo la sottocartella `Tesi Valleri/`); **E3** = `News Embedding` (OneDrive di ateneo, struttura di cartelle quasi vuota).

**Come ho trattato i file di E2 (importante).** I file di E2 su OneDrive sono "solo cloud": dal mio ambiente risultavano elencabili ma non leggibili ("Invalid argument"). Per leggerli ho dovuto farli scaricare sul PC (staging), con effetto collaterale solo di scaricamento locale (nessuna modifica ai file). Per non scaricare ~700 MB ho scelto la lettura **a campione**:
- letti per intero: CSV dei discorsi, docx, 2 zip, tesi PDF (md5);
- campione di 5 HTML di conferenze stampa, 8 PDF di Accounts (inclusi tutti quelli con nome anomalo) e 4 PDF di bollettini (2 Economic, 2 Monthly);
- per gli altri file: solo nome, data e dimensione (da elenco), non il contenuto. Perciò conteggi e anomalie dei restanti ~620 file qui sotto sono **da nomi e dimensioni**, non da lettura del testo.

### 3.1 E2 · `ecb-communications/Tesi Valleri/` (641 file, ~0,74 GB) — materiale della tesi di Valleri

| Sottocartella | Contenuto | N. file | Dimensione | Formato |
|---|---|---|---|---|
| `dati/accounts/` | Accounts of the monetary policy meeting (BCE) | 90 PDF | 29,3 MB | PDF con testo (campione) |
| `dati/economic_bulletins/` | Economic Bulletin `eb2015NN.en.pdf` … `eb202507.en.pdf` | 87 PDF | 162,7 MB | PDF con testo (campione) |
| `dati/monthly_bulletins/` | Monthly Bulletin `mb199901en.pdf` … `mb201412en.pdf` | 192 PDF | 452,4 MB | PDF con testo (campione) |
| `dati/press_conferences/` | Conferenze stampa `pressconf_YYYYMMDD.html` | 267 HTML | 39,7 MB | HTML UTF-8 |
| `dati/all_ECB_speeches.csv` | Discorsi membri del Consiglio | 1 CSV | 57,1 MB | CSV, separatore `\|`, UTF-8, CRLF |
| `docs/` | `Aggiornamento_Tesi_Valleri.docx` + copia del PDF della tesi | 1 docx, 1 PDF | 20 KB; 0,93 MB | |
| `git-hub/` | 2 zip del repository di codice | 2 zip | 1,2 MB; 1,7 MB | |

Banca centrale: BCE (solo). Periodo, frequenza, lingua e problemi per sottoinsieme:

**Accounts (90 PDF).** Periodo 2015–2025. Frequenza: 8 per anno. Lingua: inglese. Nomi file in formato `gg-mm-aa` che indica la **data di pubblicazione** [inferenza dal confronto nome/contenuto]. Controllati a campione 8 file [P2, E2]. Problemi:
- `Account of the monetary policy meeting.pdf` non ha data nel nome; il contenuto è la riunione del 21–22 gennaio 2015 (stampa del 15/01/26). Il docx dichiara che quel meeting era stato escluso "per assenza di Accounts" [E2/docs/Aggiornamento_Tesi_Valleri.docx]: **il file esiste**, quindi la dichiarazione del docx è superata o il file è stato aggiunto dopo (**[inferenza]**).
- `22-05-2020.pdf` (riunione 29–30 aprile 2020) e `25-06-2020.pdf` (riunione 3–4 giugno 2020) usano l'anno a 4 cifre invece di 2: stesso formato incoerente.
- File "p2": `9-04-20 p2.pdf` (riunione straordinaria del 18 marzo 2020) e `28-08-25 p2.pdf` (riunione 23–24 luglio 2025) sono documenti diversi, non duplicati (md5 distinti, contenuti diversi).
- **Incoerenza nome/contenuto:** `28-08-25.pdf` contiene "Meeting of 25 June 2025", mentre `28-08-25 p2.pdf` contiene la riunione del 23–24 luglio 2025. Se il nome indica la data di pubblicazione, `28-08-25.pdf` non dovrebbe riferirsi a giugno (**[inferenza]**: possibile errore di nome o di sovrapposizione); da verificare.
- Conteggio per anno (dal nome): 2015: 6+1 senza data, 2016–2019: 8, 2020: 9, 2021: 9, 2022–2024: 8, 2025: 9. 90 file vs 87 riunioni post-2015 [P2]: la differenza è spiegata dai 2 "p2" e dal file senza data [inferenza].
- Date nelle prime pagine: `08-10-15.pdf` riporta "8 October 2015" ma riguarda la riunione del 2–3 settembre 2015, coerente con "data di pubblicazione".

**Economic Bulletins (87 PDF).** Periodo: `eb201501` … `eb202507` (2015–luglio 2025); 8 per anno, 7 nel 2025. Lingua: inglese. Campione: 2 file (76 e 126 pagine), con testo, non cifrati. Il campione termina a luglio 2025 mentre l'analisi della tesi arriva a dicembre 2025: **[inferenza]** il bollettino per le riunioni autunnali 2025 potrebbe mancare (da verificare). Nomi: `eb201501` è "Issue 1 (February 2015)" [E2]; nessun duplicato per dimensione.

**Monthly Bulletins (192 PDF).** Periodo: gennaio 1999–dicembre 2014 (12 per anno × 16 anni = 192, completo). Lingua: inglese. Campione: 2 file (92 e 216 pagine), con testo. Nessun duplicato per dimensione. Licenza: non indicata (documenti pubblici BCE).

**Press conferences (267 HTML).** Date da `pressconf_19990107` a `pressconf_20260205`. Per anno: 1999: 10, 2000: 12, 2001–2003: 9, 2004–2006: 11, 2007–2014: 12, 2015–2025: 8, 2026: 1. Pre-2015: **178 file**, coerente con P2. Frequenza: mensile (fino al 2014), poi 8/anno. Lingua: inglese. Campione di 5: 3 hanno la frase "We are now at your disposal for questions" usata da P2 per troncare la parte Q&A, **2 no**: `pressconf_19990107.html` (solo introductory statement, senza Q&A) e `pressconf_20260205.html` (titolo "Monetary policy statement (with Q&A)", con formulazione diversa). **[inferenza]** la regola di estrazione di P2 può non funzionare per le conferenze senza Q&A e per i formati recenti; da testare su tutti i 267 file. Due file con la stessa dimensione (`20070412` e `20090507`, 138.315 B) hanno md5 diverso: falso positivo, non sono duplicati. Il file del 2026-02-05 è **oltre** la fine del campione (dic. 2025) [P2].

**`all_ECB_speeches.csv`.** Formato CSV con separatore `|` (pipe), UTF-8, righe CRLF. **2.995 righe × 5 colonne**: `date`, `speakers`, `title`, `subtitle`, `contents`. Periodo 1997-02-07 – 2026-02-26 (ordinato in senso decrescente), frequenza: eventi, ~100–150 per anno dal 2004; 19 nel 1997, 17 nel 2026. Lingua: prevalentemente inglese; presenti titoli in tedesco (es. "Die Geldpolitik der EZB in der Corona-Krise"). Dimensione 57,1 MB. Mediana ≈ 2.649 parole per discorso, max 19.623. Fonte: database BCE (dichiarato in P2; licenza non indicata). Problemi:
- **147 righe con `contents` mancante** (NaN) + 1 vuota = 148 senza testo, concentrate dal 2018 in poi (2018: 3, 2019: 5, 2020: 13, 2021: 17, 2022: 17, 2023: 28, 2024: 28, 2025: 31, 2026: 4; 2007: 1). 2.995 − 147 = **2.848**, cifra citata da P2 per i discorsi: **[inferenza]** P2 usa solo le righe con testo; il dato non è dichiarato.
- 193 righe con meno di 100 parole.
- `speakers` mancante in 7 righe; 2 righe con due oratori separati da virgola (es. "Christine Lagarde,Philip R. Lane").
- 29 nomi distinti: appaiono solo membri dell'Executive Board/Presidenti (Trichet, Draghi, Lagarde, Duisenberg, Lane, Schnabel, Panetta, Cipollone, …). Controllati senza esito: Weidmann, Visco, Knot, Villeroy, Nagel, Rehn. **[inferenza]** il database non contiene discorsi di governatori delle banche centrali nazionali, mentre P2 dichiara pesi per "NCB Governors = 0,55" [P2, §3.2]: il peso potrebbe non avere effetto.
- 1 coppia data+titolo duplicata (2012-05-08, "Address at the event marking the first signatures to the T2S Framework Agreement": Praet e Draghi, due righe); 222 titoli ripetuti (discorsi diversi con titolo uguale, es. "The euro area in April 2025"; e stesso discorso in lingue diverse: es. titolo in tedesco duplicato in date diverse). Nessuna riga completamente duplicata e nessun contenuto duplicato.
- 1 carattere di sostituzione (U+FFFD) nei testi; nessuna mojibake "Ã".
- Differenza tra P2 (1997–2026 per i discorsi) e il resto del campione (1999–2025): vedi §2.1.

**`docs/Aggiornamento_Tesi_Valleri.docx`** (20 KB, italiano, aprile 2026): relazione di avanzamento dello studente. Contenuti: obiettivo, architettura pipeline a 4 agenti, tabella dati (90 Accounts; 87 Economic Bulletins; 192 Monthly Bulletins; 267 conferenze; ~2.800 discorsi; 313 riunioni di policy), stato (pipeline eseguita sul 2015–2025, 87 riunioni: 47 no_surprise, 30 hawkish, 10 dovish), piano econometrico in 3 TIER: (1) LP asimmetriche; (2) panel LP core vs periferia (Italia, Spagna, Grecia, Portogallo vs Germania, Francia); (3) Draghi vs Lagarde (entropia, forward guidance Odissea/Delfica). **Differenze rispetto alla tesi finale [P2]:** il TIER 2 (panel) è dichiarato in P2 solo come sviluppo futuro; il TIER 3 diventa uno sviluppo futuro (entropia posteriore); il campione 1999–2014 era "da completare" nel docx.

**`docs/ECB Communications as Monetary Shocks...pdf`** = **duplicato esatto** di P2 (md5 `a93576f7…`, 927.504 B). Quindi la tesi esiste in due copie: nella cartella collegata (`papers/`) e in E2.

**`git-hub/` (2 zip, repository di codice).** Dati e codice descritti al §4.

### 3.2 E1 · `01-embedding-regression` (6 file, 17 MB) — letteratura e seminario sugli embedding

Cartella piatta su Google Drive. Tutti i PDF hanno testo estraibile e non sono cifrati. Non sono dati ma letteratura/materiali (vedi anche §4).

### 3.3 E3 · `News Embedding` (OneDrive, 5 file, 17 MB)

Struttura numerata (01-notebooks, 02-lecture-notes, 03-code, 04-working, 05-pictures, 06-data, 07-papers). **Cinque cartelle sono vuote** (01, 03, 04, 05, 06). Contenuto: `02-lecture-notes/` (2 file) e `07-papers/` (3 PDF, **duplicati con nomi diversi** di tre PDF di E1). **Nessun dato** in `06-data`. Il template LaTeX e `purge.bat` sono materiale di supporto.

**Riassunto dati:** l'unico set di dati testuali reali è in E2 (corpus BCE) più le serie derivate nei repository zip (surprise, dataset LP). Non ci sono dati di mercato, né dati Fed, né HICP/PIL scaricati come file separati (le serie macro compaiono solo dentro `lp_dataset.csv`).

---

## 4. CODICE E ALTRO

### 4.1 Repository di codice (2 zip in E2, `Tesi Valleri/git-hub/`)

| | `ECB-monetary-surprises-extraction-llm--main.zip` (**Z1**) | `ECB-monetary-surprises-extraction-llm-pipeline-main.zip` (**Z2**) |
|---|---|---|
| Dimensione | 1.229.603 B | 1.688.211 B |
| Date interne dei file | 5 luglio 2026 | 23 maggio 2026 |
| N. file | 14 | 17 (aggiunge 3 PDF di figure) |
| Contenuto | 2 notebook, README, 5 file in `data/`, 6 figure PNG | 2 notebook, README, 5 file in `data/`, 9 figure (6 PNG + 3 PDF) |
| Note | Un file ha nome corrotto: `ecb_policy_decisions_FULL_1999_2025.csv(` (con una parentesi finale); README aggiornato | README più vecchio (descrive 4 "agenti LLM") |

**Z1 e Z2 sono quasi-duplicati ma NON equivalenti:**
- I due notebook differiscono (LP: Z2 ha 22 celle di codice, tutte con output salvato; Z1 ne ha 16 e nessun output salvato; pipeline: 14 celle in Z2 con 1 output, 13 in Z1).
- `surprises_timeseries.csv`: stesso elenco di 312 riunioni (1999-03-04 – 2025-12-18, nessuna data duplicata), ma **i valori differiscono in ~225 righe** per le colonne di output dell'LLM (es. `dfr_surprise_mech`, `expected_change_bp`, `entropy`, `p_hold`; in 199–225 delle 312 righe a seconda della colonna); Z2 ha due colonne in più (`t`, `post2015`). Il README di Z2 descrive anche `mlf_surprise_mech`, che **non esiste in nessuno dei due CSV** (l'ultima colonna è `mro_surprise_mech`).
- Le statistiche cambiano: media della *surprise* meccanica DFR **−1,12 bp (Z1)** contro **−1,28 bp (Z2)**; Z1 ha 79 hawkish/98 dovish/135 no_surprise, Z2 73/102/137. Il README di Z2 riporta −1,3 bp. **P2 riporta −1,1 bp**, quindi **Z1 (più recente) è coerente con la tesi**; Z2 è una versione precedente (**[inferenza]**).
- Invece `lp_results_hicp.csv` è identico (β⁺(h=9) = −0,857, β⁻ = +0,480, Wald p = 0,0089, nobs = 212) in entrambe, uguale a P2. **[inferenza]** i risultati LP non sono stati riestratti dopo la nuova esecuzione del pipeline, oppure sono stati copiati: da chiarire con l'autore.
- Dei 312 record, `dfr_surprise_norm` è mancante in 14 (Z1) / 13 (Z2) e `fg_type` in 50; in `lp_dataset.csv` `unemp_rate` manca in 20 righe; `ecb_policy_decisions_FULL_1999_2025.csv` ha 313 righe e 226–227 valori mancanti nelle colonne `bulletin*`/`account*` (i campi dei documenti si compilano solo dal 2015) [Z2, E2].

| File (Z2/Z1) | Scopo | Linguaggio | Librerie | Dipendenze da dati specifici |
|---|---|---|---|---|
| `ECB_Pipeline_MASTER.ipynb` | Pipeline a 4 stadi: Agent IM/IB (chiamate Gemini), Agent II/III (aritmetica, softmax) | Python (Colab) | pdfplumber, BeautifulSoup4, google-genai, pydantic, numpy, pandas, scipy.special, tqdm | Percorso Google Drive `/content/drive/MyDrive/ECB_Research/ECB_Project`; cartelle di dati di E2; chiave API Gemini (non trovata in chiaro nel codice: verificato su pattern `AIza`). |
| `ECB_LocalProjections.ipynb` | Local Projections asimmetriche su HICP (hawkish/dovish, Wald, IRF) | Python (Colab) | statsmodels, linearmodels, scipy, matplotlib, pandas, numpy | Stessi percorsi Drive; `data/lp_dataset.csv`; output in `/output/`. |
| `data/surprises_timeseries.csv` | 312 riunioni × 23–25 colonne: punteggi, entropia, surprise MRO/DFR | CSV | — | — |
| `data/ecb_policy_decisions_FULL_1999_2025.csv` | 313 riunioni: tassi MRO/DFR/MLF e variazioni | CSV | — | — |
| `data/lp_dataset.csv` | 312 × 27 colonne: input LP (hicp_yoy, unemp_rate, gdp_yoy) | CSV | — | — |
| `data/lp_results_hicp.csv` | 25 × 13: risultati LP, β⁺/β⁻, Wald per h=0..24 | CSV | — | — |
| `data/ecb_surprises_metadata.json` | 312 record con output completo degli agenti (`agent_im`, `agent_ib`, `agent_ii`, `surprises`), 1,9 MB | JSON | — | — |
| `Figures/*` | Figure IRF, Wald, distribuzione, serie, era | PNG/PDF | — | — |

**[inferenza]** Il codice è riutilizzabile solo con ambiente Colab/Drive, chiave Gemini e i dati di E2: i percorsi sono fissi nel notebook.

### 4.2 Altro materiale

| Percorso | Tipo | Dimensione | Argomento (1 riga) |
|---|---|---|---|
| E1 `Chen Y. et al. - Expected Returns and Large Language Models.pdf` | PDF, paper, 78 pp. | 7,56 MB | Chen, Kelly, Xiu: embedding LLM (ChatGPT, LLaMA) di notizie finanziarie per prevedere i rendimenti azionari cross-section; 16 mercati, 13 lingue, portafogli con Sharpe dopo costi, underreaction [E1, abstract]. Creazione PDF feb. 2026. |
| E1 `Didisheim A. et al. - The Inefficient Pricing of News.pdf` | PDF, paper, 72 pp. | 7,89 MB | Didisheim, Kelly, Pourmohammadi, Tian (11 aprile 2026): le notizie sono prevedibili da caratteristiche degli stock; i "news shocks" (parte non prevedibile) raddoppiano il potere predittivo a 18 mesi [E1, abstract]. |
| E1 `Tang E. et al. - Understanding Embedding Models for Regression.pdf` | PDF, paper, 22 pp. | 1,36 MB | Tang, Yang, Song: embedding LLM come feature in regressione, continuità di Lipschitz, ruolo di dimensione modello [E1, abstract]. |
| E1 `Wang L. et al. - Improving Text Embeddings with LLMs.pdf` | PDF, paper arXiv 2401.00368v3, 20 pp. | 0,51 MB | Wang et al. (Microsoft): metodo con dati sintetici e fine-tuning contrastivo di LLM decoder-only per embedding di testo; MTEB/BEIR [E1, abstract]. |
| E1 `embedding_concepts_v0.tex` | LaTeX Beamer (38 frame) | 32,6 KB | Seminario "Embeddings for Statistical Research": astrazione statistica, BoW, embedding statici/contestuali, pooling, regolarizzazione, news shocks, pipeline e failure modes [E1, `\title`]. |
| E1 `purge.bat` | script batch Windows | 133 B | Cancella file ausiliari LaTeX (`*.log`, `*.aux`, …). **Identico** a `News Embedding/02-lecture-notes/purge.bat` (md5 `1f63bb…`). |
| E3 `02-lecture-notes/template.tex` | LaTeX Beamer | 2,6 KB | Template di presentazione (tema CambridgeUS), con gestione U+FFFD [E3]. |
| E3 `07-papers/2411.14708v3.pdf` | PDF | 1,36 MB | **Duplicato esatto** di `Tang E. et al...` di E1 (md5 `90160e58…`). |
| E3 `07-papers/ssrn-4416687.pdf` | PDF | 7,56 MB | **Duplicato esatto** di `Chen Y. et al...` di E1 (md5 `301af774…`). |
| E3 `07-papers/ssrn-6540399.pdf` | PDF | 7,89 MB | **Duplicato esatto** di `Didisheim A. et al...` di E1 (md5 `1e646161…`). |

Per i quattro paper di E1 ho letto soltanto frontespizio, abstract e le prime pagine; **non ho compilato la scheda completa** richiesta per i paper di comunicazione delle banche centrali (corpus, modello econometrico, variabili, limiti), perché sono lavori su notizie finanziarie e rendimenti azionari, non su comunicazione di banche centrali: se li vuoi in scheda, vanno letti per intero.

### 4.3 Duplicati, quasi-duplicati e file illeggibili (aggiornato)

| Caso | Dettaglio | Tipo |
|---|---|---|
| Tesi Valleri | `papers/ECB Communications...pdf` ≡ `ecb-communications/Tesi Valleri/docs/ECB Communications...pdf` | **duplicato esatto** (md5 `a93576f7…`) |
| Chen / Didisheim / Tang | E1 ≡ E3 `07-papers` (nomi diversi: `ssrn-4416687.pdf`, `ssrn-6540399.pdf`, `2411.14708v3.pdf`) | **duplicati esatti** (3 coppie) |
| `purge.bat` | E1 ≡ E3 `02-lecture-notes` | **duplicato esatto** |
| Z1 vs Z2 | repository diversi con output del pipeline diversi | **quasi-duplicati non equivalenti** |
| `pressconf_20070412` vs `pressconf_20090507` | stessa dimensione, md5 diverso | falso positivo (non duplicati) |
| Accounts "p2" | `9-04-20 p2`, `28-08-25 p2` | documenti diversi, non duplicati |
| Z1 `…FULL_1999_2025.csv(` | nome file corrotto (parentesi finale) | anomalia di nome |
| Illeggibili | PDF/HTML non scaricati (≈620 file di E2 non aperti, solo elencati) | **non verificati** (non illeggibili) |
| Scansioni, PDF protetti | nessuno tra quelli aperti (nessun PDF cifrato; tutti con strato testo) | — |

Nota: P1 (Baglioni) è nella cartella collegata e non ha copie altrove.

---

## 5. TABELLE RIEPILOGATIVE (aggiornate)

### 5.1 Numero di file per tipo e per cartella

| Cartella | PDF paper/tesi/libri | PDF dati (Accounts, bollettini) | HTML | CSV | docx | zip | .tex | .bat | .lnk | Totale |
|---|---|---|---|---|---|---|---|---|---|---|
| Cartella collegata (radice) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `papers/` | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| `lavoro_claude/` (creata da me) | – | – | – | – | – | – | – | – | – | 1 (inventario.md) |
| E1 `01-embedding-regression` | 4 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 6 |
| E3 `News Embedding` (tot.) | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 5 |
| E2 `ecb-communications/…/dati/accounts` | 0 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 90 |
| E2 `…/dati/economic_bulletins` | 0 | 87 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 87 |
| E2 `…/dati/monthly_bulletins` | 0 | 192 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 192 |
| E2 `…/dati/press_conferences` | 0 | 0 | 267 | 0 | 0 | 0 | 0 | 0 | 0 | 267 |
| E2 `…/dati` (radice) | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| E2 `…/docs` | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| E2 `…/git-hub` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 |
| **Totale preesistente** | 11 | 369 | 267 | 1 | 1 | 2 | 2 | 2 | 3 | **658** |

Dentro i due zip: Z1 14 file, Z2 17 file (non contati nel totale, che conta gli zip come file singoli). Cartelle vuote in E3: 5.

### 5.2 Matrice banca centrale × periodo (con dati effettivamente presenti)

Legenda: **DATI** = documenti/dati nel filesystem; **D** = descritto solo nei documenti (nessun dato); **q** = solo trattazione qualitativa; **L** = serie derivata (surprise) nei repository zip; **–** = nulla.

| Banca centrale | 1996–98 | 1999–2007 | 2008–2014 | 2015–2019 | 2020–2021 | 2022–2025 | 2026 |
|---|---|---|---|---|---|---|---|
| BCE – comunicati/conferenze stampa | DATI (discorsi dal 1997; conferenze dal 1999-01) | DATI (conferenze, Monthly Bulletin) | DATI | DATI | DATI | DATI | 1 conferenza (feb 2026), 17 discorsi |
| BCE – Accounts delle riunioni | – | – (non esistono prima del 2015) | – | DATI (2015–2019) | DATI (con anomalie di nome) | DATI (fino a 2025) | – |
| BCE – bollettini | – | DATI (Monthly) | DATI (Monthly fino a dic. 2014) | DATI (Economic) | DATI | DATI (fino a lug. 2025) | – |
| BCE – discorsi (CSV) | DATI (65 righe 1997–98) | DATI | DATI | DATI | DATI (testo mancante in molti dal 2020) | DATI (testo mancante in 28–31/anno) | 17 righe |
| BCE – *surprise* LLM (serie derivata) | – | L (dal 1999-03) | L | L | L | L (fino a 2025-12-18) | – |
| BCE – decisioni di policy (tassi) | – | L (1999–) | L | L | L | L | – |
| Fed (Beige Book, Minutes, Statements) | D (P3) | D (P3) | D (P3) | D (P3) | D (P3) | D (P3, a 2025) | – |
| Fed – dati di mercato/shock (FF, ED, R&R, BS, MAR) | D (P3) | D | D | D | D | D | – |
| BoE | – | – | q (P1) | q (P1) | q (P1) | q (P1) | – |
| BoJ | – | – | q (P1) | q (P1) | q (P1) | q (P1) | – |
| Altre / news finanziarie (E1) | – | – | – | – | – | – | solo paper |

Lacune evidenti: **nessun dato grezzo della Fed né di altre banche centrali** nel filesystem esaminato; nessun dato di mercato (futures, OIS, caplet Euribor); Accounts BCE assenti prima del 2015 per costruzione; testo dei discorsi mancante dal 2018 in poi per ~147 record.

---

## 6. I 10 elementi più rilevanti per definire un target di tesi (aggiornato)

Ogni voce indica se è **nel documento/nei file** o una mia **[inferenza]**.

1. **Pipeline multi-agente (P3) già implementata per la BCE (P2 + repository Z1/Z2 + corpus in E2)**: esiste una base riutilizzabile completa (corpus 1999–2025, codice Colab, serie di 312 *surprise*), ma la copia di codice e dati va **riconciliata** (Z1 ≠ Z2; LP identico). Fonte: E2, P2.
2. **Riproducibilità e consistenza dei dati BCE**: il corpus ha anomalie concrete e verificabili (nomi file Accounts, `28-08-25.pdf` che contiene la riunione di giugno 2025, 147 discorsi senza testo, assenza di discorsi dei governatori NCB, regola di estrazione dell'Intro Statement che fallisce su 1999-01 e su feb. 2026). Una tesi di "auditing/ricostruzione" è fattibile. Fonte: E2 [inferenza dove indicato].
3. **Outcome leakage / look-ahead dell'LLM**: P2 lo dichiara "limite aperto"; P3 propone test out-of-knowledge-cutoff. Con il codice e il corpus, un test sistematico con modelli di cutoff diversi è realizzabile. Fonte: P2 §3.5, P3 §2.6, Z1/Z2.
4. **Break gennaio 2015** (Accounts al posto degli Intro Statement): non separabile da ZLB/QE. Fonte: P2 §3.6, E2.
5. **Benchmark di mercato per la BCE** (Euribor caplet): proposta di P2 non realizzata; nessun dato di mercato nei file. Fonte: P2 §6.3 [inferenza sull'assenza].
6. **Estensione panel core/periferia (TIER 2 del docx)**: era nel piano di aprile 2026, in P2 solo come sviluppo futuro; dati macro per paese non nei file. Fonte: E2 docx, P2 §6.3.
7. **Entropia posteriore / trasparenza Draghi vs Lagarde (TIER 3)**: l'entropia è già nel CSV (`entropy`, `p_hold`); analisi non svolta. Fonte: Z1/Z2, docx, P2 §6.3.
8. **Filone embedding** (E1/E3: Chen-Kelly-Xiu, Didisheim et al., Tang et al., Wang et al. + seminario `embedding_concepts_v0.tex`): alternativa/complemento agli agenti LLM scalari, già avviata come materiale ma **senza codice, dati o notebook** (E3 ha 5 cartelle vuote). **[inferenza]** target possibile: "news shocks" per comunicazione BCE usando embedding e purging. Fonte: E1, E3.
9. **Estensione a una terza banca centrale (Fed già su P3, BoE/BoJ solo P1)**: richiede raccolta di corpora; nessuno nei file. **[inferenza]**.
10. **Asimmetria hawkish/dovish**: risultato di P2 (≈1,8:1 a h=9) con limiti dichiarati (test di Wald non indipendenti, bad control, HAC a lunghi orizzonti); replicabile con più robustezza, ma i risultati LP sono identici tra Z1 e Z2 pur con surprise diverse: **verificare quale versione genera quali numeri**. Fonte: P2 §5, Z1/Z2. **[v3]** Lo stadio LP di Z1 è stato rieseguito e coincide con `lp_results_hicp.csv` (differenza ~1e-15), ma ci sono incoerenze di convenzione di segno e di controlli (§8.3).

---

## 7. Dubbi e cose da chiarire (aggiornato)

1. **Target della tesi**: livello (triennale/magistrale), studente, e se deve estendere P2/E2 (BCE), il filone embedding (E1/E3) o altro. Gli input non lo dicono.
2. **Z1 vs Z2**: quale repository è quello definitivo? Z1 (5 luglio 2026) è coerente con P2 (−1,1 bp); Z2 (23 maggio) no. I risultati LP sono identici nei due: sono stati rigenerati?
3. **`28-08-25.pdf`** (contiene la riunione del 25 giugno 2025) e `28-08-25 p2.pdf` (luglio 2025): errore di nomi o di data? E i quattro file con nomi anomali (`22-05-2020`, `25-06-2020`, file senza data, "p2")?
4. **Accounts di gennaio 2015**: il file esiste (senza data nel nome) ma il docx di aprile 2026 diceva che mancava; il pipeline lo ha usato? P2 conta 87 riunioni post-2015.
5. **Discorsi**: 2.995 righe vs 2.848 usate (le 147 senza testo?). Il database contiene solo Executive Board: il peso 0,55 dei governatori NCB in P2 è applicato a qualcosa?
6. **Regola di estrazione dell'Intro Statement** (frase "at your disposal for questions"): come tratta le conferenze senza Q&A (es. 1999-01-07) e i formati 2025–26?
7. ~~**Bollettini autunno 2025**: gli Economic Bulletin finiscono a luglio 2025~~ **[CORRETTO in v3: i bollettini 2015–2025 sono completi; il dubbio era un'inferenza errata. Resta aperto solo il fatto che le date dei bollettini nel JSON sono nominali (giorno 28), vedi §8.3.]**
8. **Cartelle E3 vuote** (`01-notebooks`, `03-code`, `04-working`, `05-pictures`, `06-data`): sono placeholder o i file non sono sincronizzati su OneDrive?
9. **Dati Fed / di mercato** (P3, benchmark BCE): vuoi che li cerchi e li catalogo o fanno parte di un altro progetto?
10. **Profondità**: vuoi la scheda completa dei 4 paper di E1 e una verifica "integrale" dei ~620 file di E2 che ho solo elencato (richiede scaricare ~700 MB da OneDrive sul PC)?

---

## 8. Verifica di replicabilità della tesi (v3, 2026-11-13)

Confronto incrociato tra tesi [P2], cartella dati E2 e repository GitHub `enricovalleri-01/ECB-monetary-surprises-extraction-llm-` (snapshot Z1 in E2, coerente con GitHub; la pagina `data/` di GitHub non era consultabile direttamente per robots.txt). Dettaglio: `controllo_replicabilita.md`. Script: `code/`.

### 8.1 Dati: disponibilità per stadio della tesi
| Stadio | Input necessari | Presenti? | Fonte |
|---|---|---|---|
| Corpus documenti | Accounts, Economic/Monthly Bulletin, conferenze stampa, discorsi | Sì (312 riunioni mappabili a documenti; 86/87 post-2015 trovano l'Account per data, l'87° è il file senza data nel nome) | E2, `code/02_coverage_check.py` |
| Agent IM/IB/II/III | Chiamate a Gemini 2.5 Flash | Output salvati nel JSON (312 record); **rieseguire** richiede chiave API (vuota nel codice) e ~15 ore **[inferenza]**; l'LLM ospitato non garantisce lo stesso risultato | Z1 |
| Serie delle sorprese | `surprises_timeseries.csv` | Sì, ma non coincide col JSON pre-2015 (§8.3) | Z1 |
| Dati macro per LP | HICP, PIL, disoccupazione grezzi + script di merge | **No**: c'è solo il risultato `lp_dataset.csv` | Z1 |
| LP e test di Wald | `lp_dataset.csv` + notebook | Sì: **replicato esattamente** | `code/01_lp_replication.py` |

### 8.2 Codice su GitHub
Sì: `ECB_Pipeline_MASTER.ipynb` (pipeline) e `ECB_LocalProjections.ipynb` (LP). Non eseguibili "as is": errore di sintassi (`SPEAKER_ROLES = {` duplicato), percorsi Colab/Drive fissi, `GEMINI_API_KEY = ''`; manca il codice che costruisce `lp_dataset.csv`.

### 8.3 Discrepanze tesi ↔ dati/codice
1. JSON vs CSV: `expected_change_bp` identico dal 2015, **diverso in 225/312 riunioni (1999–2014), fino a 14,6 bp**; nel CSV i valori sono multipli di 0,0025 (compatibili con una media di 4 esecuzioni **[inferenza]**) [`code/03`].
2. Accuratezza della moda ricalcolata 77,2% (pre-2015 76,9%) contro 68% in tesi; baseline "sempre hold" 81,4%; 5 esiti effettivi non classificabili nei 5 bin (`other`) [`code/04`].
3. Convenzione di segno: `shock_neg = surprise.clip(upper=0)` (valori ≤ 0): β⁻ = +0,480 a h=9 implica che uno shock dovish **riduce** l'HICP; il confronto 1,8:1 usa i valori assoluti [`code/01`].
4. Controllo `gdp_gap` nelle LP assente dall'equazione 5.1.
5. Discorsi per riunione: 5,19 (JSON) / 5,63 (ricalcolo) vs ≈8 in tesi; 20 riunioni senza discorsi [`code/02`].
6. Bug probabile sui pesi dei relatori: il codice legge `speaker`, il CSV ha `speakers`; inoltre 10 nomi su 29 non sarebbero riconosciuti per accenti/iniziali (es. "Philip R. Lane", "Benoît Cœuré") [`code/05`] — **[inferenza]** statica.
7. Date dei bollettini nominali (giorno 28 in 230 casi su 312); timing HICP rispetto al mese della riunione.
8. Riunione del 22/01/2015 esclusa (312 su 313 decisioni): il suo Account esiste ed è il file senza data nel nome.

### 8.4 Correzioni rispetto alle versioni precedenti
- Economic Bulletin 2015–2025 **completi** (non manca l'autunno 2025).
- Conteggio dei discorsi con testo: 2.847 (2.995 righe − 148 vuote) nel ricalcolo dello script 02; in precedenza avevo scritto 2.848 (147 NaN): la differenza è una riga con testo vuoto ma non NaN.
