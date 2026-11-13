"""Percorsi comuni degli script di verifica. Modificare solo questo file per cambiare ambiente.

Variabili d'ambiente (opzionali):
  ECB_REPO  cartella del repository Valleri ESTRATTO dallo zip Z1
            (ECB-monetary-surprises-extraction-llm--main/, contiene data/ e i 2 notebook)
  ECB_DATI  cartella 'Tesi Valleri/dati' (accounts, economic_bulletins, monthly_bulletins,
            press_conferences, all_ECB_speeches.csv)
  ECB_OUT   cartella di output degli script (default: ./out)
"""
import os
REPO = os.environ.get("ECB_REPO", "./ECB-monetary-surprises-extraction-llm--main")
DATI = os.environ.get("ECB_DATI", "./dati")
OUT = os.environ.get("ECB_OUT", "./out")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(REPO, "data") + os.sep
