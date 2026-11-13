"""02 - Copertura documentale: ogni riunione (312) ha il documento che il pipeline dichiara di aver usato?

Metodo: legge data/ecb_surprises_metadata.json (record agent_im / agent_ib) e confronta
 - pre-2015: account_pub_date dell'Agent IM  <-> conferenze stampa presenti in dati/press_conferences
 - post-2015: account_pub_date <-> nomi dei PDF in dati/accounts (data di pubblicazione nel nome)
 - bulletin_date dell'Agent IB (giorno del mese tipico = 28: data NOMINALE, non di pubblicazione)
 - n. discorsi nella finestra [t-28g, t-10g] dal CSV dei discorsi (con e senza testo) vs speech_n_processed
Elenchi: nomi file Accounts e date delle conferenze si leggono dalle cartelle in DATI, o in alternativa da ./liste/.
Output: out/cov.pkl + stampa. Esito 2026-11-13: tutte le 312 riunioni mappabili; discorsi ~5.6/riunione (tesi: ~8).
"""
import json, os, re
import pandas as pd
from datetime import timedelta
from _common import DATA, DATI, OUT

# Elenchi dei documenti: dalle cartelle in DATI se presenti, altrimenti dai file in ./liste
# (estratti dagli elenchi di file di OneDrive il 2026-10-01; nomi senza estensione per gli Accounts).
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(DATI + "/accounts") and os.path.isdir(DATI + "/press_conferences"):
    acc_names = [f[:-4] for f in os.listdir(DATI + "/accounts") if f.lower().endswith(".pdf")]
    pcs = [re.search(r"(\d{8})", f).group(1) for f in os.listdir(DATI + "/press_conferences") if re.search(r"\d{8}", f)]
else:
    acc_names = open(HERE + "/liste/accounts_filenames.txt").read().split("\n")
    acc_names = [a for a in acc_names if a]
    pcs = open(HERE + "/liste/pressconf_dates.txt").read().split()
pcset = set(pd.to_datetime(pcs, format="%Y%m%d"))
accdates = {}
for a in acc_names:
    m = re.match(r"^(\d{1,2})-(\d{2})-(\d{2,4})", a)   # gg-mm-aa oppure gg-mm-aaaa
    if m:
        d_, mo_, y_ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y_ = y_ + 2000 if y_ < 100 else y_
        accdates[pd.Timestamp(y_, mo_, d_)] = a

meta = json.load(open(DATA + "ecb_surprises_metadata.json"))
rows = []
for r in meta:
    im, ib = r["agent_im"], r["agent_ib"]
    rows.append(dict(
        meeting=pd.Timestamp(r["meeting_date"]),
        im_doc=pd.Timestamp(im["meeting_analyzed"]) if im.get("meeting_analyzed") else pd.NaT,
        im_pub=pd.Timestamp(im["account_pub_date"]) if im.get("account_pub_date") else pd.NaT,
        bull=pd.Timestamp(ib["bulletin_date"]) if ib.get("bulletin_date") else pd.NaT,
        nsp=ib.get("speech_n_processed")))
d = pd.DataFrame(rows)
d["post"] = d.meeting >= pd.Timestamp("2015-01-01")
print("record", len(d), "pre", (~d.post).sum(), "post", d.post.sum())
pre, post = d[~d.post], d[d.post]
print("pre-2015: documento IM non presente tra le conferenze:", (~pre.im_pub.isin(pcset)).sum())
print("post-2015: documento IM presente tra gli Accounts:", post.im_pub.isin(accdates.keys()).sum(), "/", len(post))
print(post[~post.im_pub.isin(accdates.keys())][["meeting", "im_doc", "im_pub"]].to_string())
print("bulletin_date: giorno del mese piu' frequente:", d.bull.dt.day.value_counts().head(3).to_dict(),
      "| nulli:", d.bull.isna().sum())
lag = (d.meeting - d.bull).dt.days
print("giorni riunione-bollettino: min/mediana", lag.min(), lag.median(), "| <10 gg:", (lag < 10).sum())

sp = pd.read_csv(DATI + "/all_ECB_speeches.csv", sep="|")
sp["d"] = pd.to_datetime(sp["date"])
sp["has"] = sp["contents"].fillna("").str.strip().ne("")
spt = sp[sp.has]
print("discorsi nel CSV:", len(sp), "| con testo:", len(spt), "| senza testo:", (~sp.has).sum())
cnt = lambda m, df: int(((df.d >= m - timedelta(days=28)) & (df.d <= m - timedelta(days=10))).sum())
d["n_csv_all"] = d.meeting.map(lambda m: cnt(m, sp))
d["n_csv_text"] = d.meeting.map(lambda m: cnt(m, spt))
d["nsp"] = pd.to_numeric(d.nsp)
print("discorsi/riunione: media nel JSON", round(d.nsp.mean(), 2), "| ricalcolo da CSV (con testo)", round(d.n_csv_text.mean(), 2))
print("coincidenza esatta JSON vs CSV (con testo):", (d.nsp == d.n_csv_text).sum(), "/", len(d))
print("riunioni senza alcun discorso (JSON):", (d.nsp == 0).sum())
d.to_pickle(OUT + "/cov.pkl")
