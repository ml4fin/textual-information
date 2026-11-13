"""01 - Replica esatta dello stadio Local Projections (LP) della tesi (Tabella 5.1 / cap. 5).

Metodo: esegue SENZA modifiche le celle del notebook ECB_LocalProjections.ipynb di Z1
(cella 1 = caricamento dati, cella 3 = dataset mensile, cella 6 = stima LP congiunta hawkish/dovish)
sul file data/lp_dataset.csv, poi confronta con data/lp_results_hicp.csv.
Unica modifica: DATA_PATH puntato al file locale e riscalatura surprise/25 (come nel notebook).
Dipendenze: numpy, pandas, scipy, statsmodels.
Output: out/lp_rerun.csv, out/monthly.pkl, stampa a video del confronto (max diff beta, p-value Wald, nobs).
Esito atteso (verificato 2026-11-13): differenze ~0 su beta_pos, beta_neg, p Wald e nobs.
NB: gli indici delle celle (1, 3, 6) valgono per il notebook di Z1; per Z2 vanno riverificati.
"""
import json
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2
from _common import REPO, DATA, OUT

j = json.load(open(REPO + "/ECB_LocalProjections.ipynb"))
src = lambda i: "".join(j["cells"][i]["source"])
ns = {"np": np, "pd": pd, "sm": sm, "stats": stats, "chi2": chi2}
exec("DATA_PATH=%r" % (DATA + "lp_dataset.csv"), ns)
exec(src(1), ns)
exec(src(3), ns)
exec("""
monthly['surprise_n']=monthly['surprise']/25.0
monthly['shock_pos_n']=monthly['shock_pos']/25.0
monthly['shock_neg_n']=monthly['shock_neg']/25.0
""", ns)
exec(src(6), ns)
res = ns["lp_hicp_joint"]
ref = pd.read_csv(DATA + "lp_results_hicp.csv")
print(res[["h", "beta_pos", "beta_neg", "se_pos", "se_neg", "wald_pval", "nobs"]].round(4).to_string(index=False))
m = res.merge(ref[["h", "beta_pos", "beta_neg", "wald_pval", "nobs"]], on="h", suffixes=("", "_ref"))
print("max abs diff beta_pos", (m.beta_pos - m.beta_pos_ref).abs().max(),
      "beta_neg", (m.beta_neg - m.beta_neg_ref).abs().max(),
      "wald p", (m.wald_pval - m.wald_pval_ref).abs().max(),
      "nobs", (m.nobs - m.nobs_ref).abs().max())
mo = ns["monthly"]
print("righe mensili", len(mo), "| shock_neg min/max", mo.shock_neg.min(), mo.shock_neg.max(),
      "| shock_pos min/max", mo.shock_pos.min(), mo.shock_pos.max())
print("mesi con shock_pos>0:", (mo.shock_pos > 0).sum(), "| mesi con shock_neg<0:", (mo.shock_neg < 0).sum())
print("orizzonti asimmetrici (p<0.10):", int(res.asymmetric.sum()))
print("NOTA segno: shock_neg <= 0 (clip upper=0): beta_neg>0 implica che uno shock dovish (negativo) ABBASSA l'HICP.")
res.to_csv(OUT + "/lp_rerun.csv", index=False)
mo.to_pickle(OUT + "/monthly.pkl")
