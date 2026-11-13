"""04 - Accuratezza della moda a priori vs esito effettivo e baseline "sempre hold".

Metodo: confronta modal_outcome (bin modale dell'Agent II, colonna di surprises_timeseries.csv) con il bin
dell'esito effettivo ricavato da dfr_actual_bp: >=50 hike_large, >=25 hike_standard, 0 hold, (-50,-25] cut_standard, <=-50 cut_large.
Riporta accuratezza totale, pre/post 2015, quota di 'hold' effettivi (baseline sempre-hold) e quota di 'hold' previsti.
Confronto con la tesi: 68% pre-2015 (tesi) vs risultato ricalcolato qui (vedi stampa). I bin intermedi (es. -10 bp, +15 bp) 
sono classificati 'other' e contano come errore: se la tesi li accorpa in un bin diverso, il risultato cambia (da verificare).
"""
import pandas as pd
from _common import DATA

s = pd.read_csv(DATA + "surprises_timeseries.csv", parse_dates=["meeting_date"])
def bin_of(x):
    if x >= 50: return "hike_large"
    if x >= 25: return "hike_standard"
    if x == 0: return "hold"
    if x <= -50: return "cut_large"
    if x <= -25: return "cut_standard"
    return "other"
s["act_bin"] = s.dfr_actual_bp.map(bin_of)
s["post"] = s.meeting_date >= "2015-01-01"
print("righe", len(s), "| esiti effettivi:", s.act_bin.value_counts().to_dict())
print("modal_outcome:", s.modal_outcome.value_counts().to_dict())
acc = (s.modal_outcome == s.act_bin)
print("accuratezza moda totale:", round(acc.mean(), 3), "| pre-2015:", round(acc[~s.post].mean(), 3), "| post-2015:", round(acc[s.post].mean(), 3))
hold = (s.dfr_actual_bp == 0)
print("quota hold effettivi (baseline 'sempre hold'): totale", round(hold.mean(), 3),
      "| pre", round(hold[~s.post].mean(), 3), "| post", round(hold[s.post].mean(), 3))
print("quota hold previsti dalla moda:", round((s.modal_outcome == "hold").mean(), 3))
