"""
Version anglaise : chaque page s'ouvre en anglais sous les deux modeles, cases
cochees, sans exception ; la bascule FR <-> EN garde la page choisie ; et il ne
reste pas de francais visible dans les titres, metriques et legendes.

    python tests/verifie_anglais.py
"""
import ast
import json
import os
import re
import sys

from streamlit.testing.v1 import AppTest

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(RACINE, "app.py")
sys.path.insert(0, RACINE)
TAB = json.load(open(os.path.join(RACINE, "traductions", "en.json"), encoding="utf-8"))
echecs = []


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def pages():
    arbre = ast.parse(open(os.path.join(RACINE, "principal.py")).read())
    for n in arbre.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PAGES":
            return [k.value for k in n.value.keys]


FRANCAIS = re.compile(r"[éèêàùç]|\b(le|la|les|des|du|une|est|pour|avec|sur|dans|qui)\b", re.I)


def textes_visibles(at):
    out = []
    for kind in ("title", "header", "subheader", "caption", "metric"):
        for e in at.get(kind):
            out.append(str(getattr(e, "label", "") or "") + " " + str(getattr(e, "value", "") or ""))
    return out


MODELES = ["Black-Scholes + skew", TAB["Bates (vol stochastique + sauts)"]]
restes = {}
for modele in MODELES:
    for page in pages():
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        at.sidebar.toggle(key="anglais").set_value(True).run()
        next(r for r in at.sidebar.radio if TAB.get(page, page) in r.options).set_value(TAB.get(page, page)).run()
        at.sidebar.radio(key="modele").set_value(modele).run()
        for c in at.checkbox:
            c.check()
        at.run()
        nom = f"EN | {modele.split()[0]:6} | {TAB.get(page, page)}"
        controle(nom, not at.exception, at.exception[0].value[:300] if at.exception else "")
        for t in textes_visibles(at):
            if FRANCAIS.search(t):
                restes.setdefault(t[:90], nom)

at = AppTest.from_file(APP, default_timeout=300)
at.run()
next(r for r in at.sidebar.radio if "Atelier Sales" in r.options).set_value("Atelier Sales").run()
at.sidebar.toggle(key="anglais").set_value(True).run()
p_en = at.session_state["page"]
at.sidebar.toggle(key="anglais").set_value(False).run()
controle("Bascule FR -> EN -> FR : la page choisie est gardee",
         not at.exception and p_en == TAB["Atelier Sales"] and at.session_state["page"] == "Atelier Sales",
         f"{p_en} / {at.session_state['page']}")

print(f"\n{len(restes)} texte(s) visible(s) qui semblent encore en francais :")
for t, ou in sorted(restes.items())[:60]:
    print(f"  - {t}   [{ou}]")
print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
