"""05 - Verifica statica del possibile bug sui pesi dei relatori (speaker weights) nel notebook della pipeline.

Metodo (NON esegue il pipeline): estrae dal notebook ECB_Pipeline_MASTER.ipynb le righe che usano 'speaker'/'speakers',
verifica la colonna effettiva nel CSV dei discorsi e riporta quanti nomi del CSV NON verrebbero riconosciuti dalla regola
`name.lower() in speaker.lower()` usata da get_speaker_weight (fallback DEFAULT_WEIGHTS['unknown']=0.40).
Esito 2026-11-13: il codice legge row['speaker'] (con guardia 'speaker' in row.index) ma il CSV ha la colonna 'speakers'
e nessun rename -> verosimilmente peso 0.40 a tutti. E' un'INFERENZA statica: va confermata rieseguendo il pipeline.
"""
import json, re
import pandas as pd
from _common import REPO, DATI

nb = json.load(open(REPO + "/ECB_Pipeline_MASTER.ipynb"))
code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
print("--- righe del notebook con 'speaker' ---")
for i, line in enumerate(code.splitlines()):
    if re.search(r"speaker", line, re.I):
        print(f"{i:5d}: {line.strip()[:140]}")
sp = pd.read_csv(DATI + "/all_ECB_speeches.csv", sep="|", usecols=["speakers"])
print("\ncolonne del CSV: speakers (singolare 'speaker' assente)")
print("valori distinti in 'speakers':", sp.speakers.nunique())
print(sp.speakers.value_counts().head(15).to_string())
m = re.search(r"SPEAKER_ROLES\s*=\s*\{(.*?)\n\}", code, re.S)
names = re.findall(r"['\"]([A-Za-z\-\. ]+)['\"]\s*:", m.group(1)) if m else []
print("\nnomi nel dizionario SPEAKER_ROLES:", names[:40])
lc = [n.lower() for n in names]
unmatched = [v for v in sp.speakers.dropna().unique() if not any(n in v.lower() for n in lc)]
print("valori distinti del CSV non riconosciuti dalla regola:", len(unmatched), "su", sp.speakers.nunique())
print("esempi:", unmatched[:15])
print("N.B. il dizionario puo' essere definito due volte (errore di sintassi in Z1 cella 5): l'estrazione prende la prima occorrenza.")
