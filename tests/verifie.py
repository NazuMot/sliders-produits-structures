"""
Verification complete de l'appli, lancee par GitHub a chaque push.

    python tests/verifie.py

1. Chaque page s'ouvre sans exception, sous les deux modeles, avec toutes
   les cases cochees (les calculs caches derriere une case sont testes aussi).
2. Les chiffres de reference du CLAUDE.md n'ont pas bouge : s'ils changent,
   c'est que le pricing a change.
3. La recherche du glossaire et la page Entrainement (chaque reponse et chaque
   demonstration chiffree, mode hasard, compteur) fonctionnent.

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

# 3. Recherche du glossaire -----------------------------------------------------
def cherche(mot):
    return lambda a: widget(a.text_input, "Rechercher").set_value(mot)


def nb_resultats(at):
    for c in at.caption:
        if "trouvee(s) pour" in c.value:
            return int(c.value.split()[0])
    return 0


at = ouvre("Glossaire", MODELES[0], [cherche("barriere")])
n = nb_resultats(at)
controle("Glossaire : 'barriere' trouve des definitions, les onglets s'effacent",
         not at.exception and n > 0 and not at.tabs, f"{n} resultats")
at = ouvre("Glossaire", MODELES[0], [cherche("  BARRIÈRE ")])
controle("Glossaire : recherche insensible aux accents et a la casse",
         not at.exception and nb_resultats(at) == n, f"{nb_resultats(at)} resultats")
at = ouvre("Glossaire", MODELES[0], [cherche("xqzwvk")])
controle("Glossaire : mot absurde -> message propre",
         not at.exception and nb_resultats(at) == 0
         and any("Aucune definition" in i.value for i in at.info))
at = ouvre("Glossaire", MODELES[0], [cherche("barriere"), cherche("")])
controle("Glossaire : champ vide -> les six onglets reviennent",
         not at.exception and len(at.tabs) == 6)

# 4. Page Entrainement ----------------------------------------------------------
from vues.entrainement import QUESTIONS, CLIENT

N = len(QUESTIONS)
for modele in MODELES:
    at = ouvre("Entrainement", modele)
    rates = []
    for i, x in enumerate(QUESTIONS):
        at.selectbox(key="ent_q").set_value(i).run()
        at.button(key="ent_voir").click().run()
        # 2 metriques pour le compteur, davantage si la demonstration s'est affichee
        if at.exception or (x["demo"] and len(at.metric) <= 2):
            rates.append(f"{i + 1}: " + (at.exception[0].value[:200] if at.exception
                                         else "demonstration vide"))
    vues = metriques(at)["Questions vues"]
    controle(f"{modele.split()[0]:6} | Entrainement : les {N} reponses et demonstrations",
             not rates and vues == f"{N} / {N}", "; ".join(rates) or f"compteur {vues}")

at = ouvre("Entrainement", MODELES[0],
           [lambda a: a.selectbox(key="ent_theme").set_value(CLIENT)])
theme = [i for i, x in enumerate(QUESTIONS) if x["theme"] == CLIENT]
tirees, ok = [], not at.exception
for _ in theme:
    avant = at.selectbox(key="ent_q").value
    at.button(key="ent_hasard").click().run()
    q = at.selectbox(key="ent_q").value
    ok = ok and not at.exception and q != avant and q in theme
    at.button(key="ent_voir").click().run()
    tirees.append(q)
controle("Entrainement : hasard dans le theme, sans repetition avant d'avoir tout vu",
         ok and len(set(tirees)) == len(theme)
         and metriques(at)["Vues dans ce theme"] == f"{len(theme)} / {len(theme)}",
         f"{len(set(tirees))} questions differentes sur {len(theme)}")

avant = at.selectbox(key="ent_q").value
at.button(key="ent_suivante").click().run()
suivante = theme[(theme.index(avant) + 1) % len(theme)]
at.button(key="ent_zero").click().run()
controle("Entrainement : question suivante et remise a zero du compteur",
         not at.exception and at.selectbox(key="ent_q").value == suivante
         and metriques(at)["Questions vues"] == f"0 / {N}")

# 5. Scenarios de marche et wrong-way risk ---------------------------------------
def scenario(nom):
    return lambda a: a.sidebar.selectbox(key="scenario").set_value(nom)




parts = {}
for nom in ("Taux zero (2020-2021)", "Remontee des taux (2022-2023)"):
    at = ouvre("2 - Capital Protection", MODELES[0], [scenario(nom)])
    parts[nom] = nombre(metriques(at)["Participation offerte"]) if not at.exception else -1
    controle(f"Scenario '{nom}' : la barre laterale prend les valeurs du scenario",
             not at.exception and widget(at.sidebar.slider, "Taux sans risque").value
             == __import__("barre").SCENARIOS[nom]["valeurs"]["taux"])
controle("Scenarios : la participation du capital garanti remonte avec les taux",
         parts["Remontee des taux (2022-2023)"] > parts["Taux zero (2020-2021)"] >= 0,
         " -> ".join(f"{v:.0f} %" for v in parts.values()))

at = ouvre("3 - Barrier Reverse Convertible", MODELES[0],
           [lambda a: a.checkbox(key="brc_voir_wrong_way").check()])
t = at.table[-1].value if at.table else None
ok = not at.exception and t is not None and len(t) == 2
if ok:
    double = [float(v.rstrip("%")) for v in t["Krach ET defaut"]]
    queue = [float(v.split("%")[0]) for v in t["Perte moyenne dans le pire 1 %"]]
    ok = double[1] > double[0] and queue[1] > queue[0]
controle("BRC wrong-way : krach et defaut ensemble plus frequents, queue plus lourde",
         ok, f"{double} / {queue}" if t is not None and len(t) == 2 else "")

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
