"""
Verification complete de l'appli, lancee par GitHub a chaque push.

    python tests/verifie.py

1. Chaque page s'ouvre sans exception, sous les deux modeles, avec toutes
   les cases cochees (les calculs caches derriere une case sont testes aussi).
2. Les chiffres de reference du CLAUDE.md n'ont pas bouge : s'ils changent,
   c'est que le pricing a change.

Sort avec le code 1 au premier probleme, ce qui fait echouer le test GitHub.
"""

import ast
import os
import sys
import time

from streamlit.testing.v1 import AppTest

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(RACINE, "app.py")
sys.path.insert(0, RACINE)

MODELES = ["Black-Scholes + skew", "Bates (vol stochastique + sauts)"]
echecs = []


def pages():
    """Les pages du menu, lues dans app.py pour ne jamais en oublier une."""
    arbre = ast.parse(open(APP).read())
    for n in arbre.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PAGES":
            return [k.value for k in n.value.keys]
    raise SystemExit("dictionnaire PAGES introuvable dans app.py")


def ouvre(page, modele, reglages=None):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].set_value(page)
    at.sidebar.radio[0].set_value(modele)
    at.run()
    for regle in reglages or []:
        regle(at)
        at.run()
    return at


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def metriques(at):
    return {m.label: m.value for m in at.metric}


def nombre(texte):
    return float(texte.replace("%", "").replace("p.a.", "").replace(",", "").strip())


def widget(liste, debut):
    return next(w for w in liste if w.label.startswith(debut))


# 1. Toutes les pages, deux modeles, cases decochees puis cochees ---------------
def coche_tout(at):
    for c in at.checkbox:
        c.check()


for modele in MODELES:
    for page in pages():
        for cocher in (False, True):
            t = time.time()
            at = ouvre(page, modele, [coche_tout] if cocher else None)
            nom = f"{modele.split()[0]:6} | {page}" + (" | cases cochees" if cocher else "")
            controle(nom, not at.exception,
                     at.exception[0].value[:300] if at.exception else f"{time.time() - t:.1f}s")

# 2. Chiffres de reference ------------------------------------------------------
# Parametres par defaut de la barre laterale (marge banque 0,8 % par an).
at = ouvre("2 - Capital Protection", MODELES[0])
p = nombre(metriques(at)["Participation offerte"])
controle("Capital garanti 5 ans : participation ~55 %", 53 <= p <= 57, f"{p:.0f}%")

at = ouvre("3 - Barrier Reverse Convertible", MODELES[0],
           [lambda a: widget(a.sidebar.slider, "Skew").set_value(0.0)])
c = nombre(metriques(at)["Coupon equitable"])
controle("BRC 1 an barriere 70 %, BS plat : coupon ~5,9 %", 5.6 <= c <= 6.2, f"{c:.2f}%")

at = ouvre("3 - Barrier Reverse Convertible", MODELES[1])
c = nombre(metriques(at)["Coupon equitable"])
controle("BRC 1 an barriere 70 %, Bates : coupon ~8,7 %", 8.3 <= c <= 9.1, f"{c:.2f}%")

res = []
for corr in (0.9, 0.3):
    at = ouvre("4 - Autocall Phoenix", MODELES[0], [
        lambda a: widget(a.select_slider, "Nombre de sous-jacents").set_value(3),
        lambda a, c=corr: widget(a.slider, "Correlation entre").set_value(c)])
    m = metriques(at)
    res.append((nombre(m["Equivalent annuel"]), nombre(m["Proba de perte en capital"])))
controle("Autocall worst-of 3 : coupon et proba de perte montent quand la correlation baisse",
         res[1][0] > res[0][0] and res[1][1] > res[0][1],
         f"coupon {res[0][0]:.1f} -> {res[1][0]:.1f} %, perte {res[0][1]:.0f} -> {res[1][1]:.0f} %")

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
