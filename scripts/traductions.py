"""
Liste les textes affiches qui n'ont pas encore de traduction anglaise.

    python scripts/traductions.py            -> resume
    python scripts/traductions.py --export f -> f : JSON {numero: texte francais} a traduire

Une traduction s'ajoute dans traductions/en.json : {"texte francais": "english text"}.
Dans les f-strings, garder les {0}, {1}... (ce sont les valeurs calculees).
"""
import glob
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
import traduction as T  # noqa: E402

FICHIERS = ["principal.py", "barre.py", "graphiques.py", "sales.py", "backtest.py"] + \
    sorted(os.path.relpath(f, RACINE) for f in glob.glob(os.path.join(RACINE, "vues", "*.py")))


def textes():
    vus = {}
    for f in FICHIERS:
        for c in T.cibles(open(os.path.join(RACINE, f), encoding="utf-8").read()):
            vus.setdefault(c[1], f)
    return vus


if __name__ == "__main__":
    tous = textes()
    tab = T.table()
    manque = [k for k in tous if k not in tab]
    print(f"{len(tous)} textes, {len(tous) - len(manque)} traduits, {len(manque)} a traduire")
    if "--export" in sys.argv:
        dest = sys.argv[sys.argv.index("--export") + 1]
        json.dump({i: k for i, k in enumerate(manque)}, open(dest, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=0)
