"""03 - Statistiche della serie delle sorprese e coerenza interna dei dati.

(a) Statistiche della sorpresa meccanica DFR (dfr_surprise_mech, punti base) per sotto-periodi
    (Duisenberg/Trichet fino a ott. 2011, Draghi nov. 2011-ott. 2019, Lagarde da nov. 2019) e break gen. 2015
    (t-test sulla differenza di medie pre/post, Levene sulle varianze).
(b) Aritmetica dell'Agent III: E[delta i] = sum_k p_k * bin_k ricalcolata dalla posterior (agent_ii) vs valore salvato.
(c) Coerenza JSON <-> CSV: expected_change_bp in ecb_surprises_metadata.json vs surprises_timeseries.csv.
    Esito 2026-11-13: (b) errore 0; (c) uguali dal 2015, DIVERSI in quasi tutte le riunioni 1999-2014 (fino a 14.6 bp).
Output: out/json_vs_csv_expected.csv (differenze per riunione) + stampa.
"""
import json
import pandas as pd
from scipy import stats
from _common import DATA, OUT

s = pd.read_csv(DATA + "surprises_timeseries.csv", parse_dates=["meeting_date"])
meta = json.load(open(DATA + "ecb_surprises_metadata.json"))
x = s.dfr_surprise_mech
eras = [("Full", s.meeting_date >= "1999-01-01"),
        ("Duis/Trich", s.meeting_date <= "2011-10-31"),
        ("Draghi", (s.meeting_date >= "2011-11-01") & (s.meeting_date <= "2019-10-31")),
        ("Lagarde", s.meeting_date >= "2019-11-01")]
for n, k in eras:
    v = x[k]
    print(f"{n:10s} N={k.sum():3d} media={v.mean():6.2f} sd={v.std():5.2f} min={v.min():6.1f} max={v.max():5.1f} "
          f"no_surprise(|x|<=5)={(v.abs() <= 5).sum()} hawkish={(v > 5).sum()} dovish={(v < -5).sum()}")
pre, post = x[s.meeting_date < "2015-01-01"], x[s.meeting_date >= "2015-01-01"]
t = stats.ttest_ind(pre, post)
print("pre/post 2015: medie", round(pre.mean(), 2), round(post.mean(), 2), "n", len(pre), len(post),
      "| t", round(t.statistic, 3), "p", round(t.pvalue, 5), "| Levene p", round(stats.levene(pre, post).pvalue, 3))

bins = {"hike_large": 50, "hike_standard": 25, "hold": 0, "cut_standard": -25, "cut_large": -50}
rows = []
for r in meta:
    pd_ = r["agent_ii"]["posterior_distribution"]
    e_calc = sum(pd_[k] * v for k, v in bins.items() if k in pd_)
    e_json = r["agent_ii"]["distribution_moments"]["expected_change_bp"]
    rows.append((r["meeting_date"], e_calc, e_json, r["surprises"]["dfr_surprise"]["expected_bp"]))
d = pd.DataFrame(rows, columns=["meeting_date", "e_calc", "e_json", "e_json_surprises"])
print("(b) max |E calcolata - E salvata|:", (d.e_calc - d.e_json).abs().max().round(4),
      "| max |E agent_ii - E surprises|:", (d.e_json - d.e_json_surprises).abs().max().round(4))
d["e_csv"] = d.meeting_date.map(s.set_index(s.meeting_date.dt.strftime("%Y-%m-%d")).expected_change_bp)
d["diff_csv_json"] = d.e_csv - d.e_json
print("(c) max |CSV - JSON| =", d.diff_csv_json.abs().max().round(2), "bp;",
      "riunioni con |diff|>0.01:", (d.diff_csv_json.abs() > 0.01).sum(), "/", len(d),
      "| ultima con differenza:", d[d.diff_csv_json.abs() > 0.01].meeting_date.max())
d.to_csv(OUT + "/json_vs_csv_expected.csv", index=False)
print(s[s.meeting_date.between("2014-12-01", "2015-03-31")][["meeting_date", "dfr_actual_bp", "expected_change_bp", "dfr_surprise_mech"]])
