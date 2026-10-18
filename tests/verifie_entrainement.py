"""
Verification du jeu « Devine le coupon » de la page Entrainement.

    python tests/verifie_entrainement.py

1. Tirages : deterministes pour une graine, dans les bornes et sur les crans des curseurs
2. Vitesse : moins de 2 s par tirage (valeur + effet de chaque parametre), deux modeles
3. Sous les deux modeles, pour chaque produit : tirage, saisie, revelation sans
   exception ; la valeur revelee = le calcul direct, et = la page produit ouverte
   par le bouton « Refaire ce produit dans sa page »
4. Verdicts (dans le mille / proche / loin), score, nouveau tirage, remise a zero
5. La graine passe par le lien
6. Version anglaise : le jeu s'affiche et se revele sans francais visible

Sort avec le code 1 au premier probleme.
"""

import json
import os
import random
import re
import sys
import time

from streamlit.testing.v1 import AppTest

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(RACINE, "app.py")
sys.path.insert(0, RACINE)

import vues.entrainement as E           # noqa: E402

TAB = json.load(open(os.path.join(RACINE, "traductions", "en.json"), encoding="utf-8"))
BS, BT = "Black-Scholes + skew", "Bates (vol stochastique + sauts)"
PAGE = "Entraînement"
# Reglages par defaut de la barre laterale
MARGE, RECOUVREMENT, NOMINAL = 0.008, 0.4, 1000.0
BATES = (0.24, 2.0, 0.5, -0.7, 0.3, -0.10, 0.15)
# Le chiffre de la page produit, pour chaque produit
METRIQUE = {E.DV_BRC: "Coupon équitable", E.DV_CP: "Participation offerte",
            E.DV_AC: "Équivalent annuel", E.DV_DC: "Décote"}
# Bornes et crans des curseurs (barre laterale et pages produits)
CURSEURS = {"taux": (0, 7, 0.25), "dividende": (0, 7, 0.25), "vol": (8, 70, 1),
            "skew": (0, 6, 0.5), "spread": (0, 500, 25),
            "brc_maturite": (0.25, 3, 0.25), "brc_barriere": (50, 95, 5),
            "cp_maturite": (1, 10, 0.5), "cp_protection": (80, 100, 5), "cp_cap": (110, 300, 10),
            "ac_maturite": (1, 6, 0.5), "ac_trigger": (80, 110, 5),
            "ac_bar_coupon": (50, 100, 5), "ac_bar_capital": (40, 90, 5),
            "dc_maturite": (0.25, 3, 0.25), "dc_cap": (70, 130, 5)}

echecs = []


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def nombre(texte):
    return float(texte.replace("%", "").replace("p.a.", "").replace("pts", "")
                 .replace(",", "").strip())


def metriques(at):
    return {m.label: m.value for m in at.metric}


def erreur(at):
    return at.exception[0].value[:300] if at.exception else ""


def choisit_page(at, page):
    next(r for r in at.sidebar.radio if page in r.options).set_value(page).run()


def ouvre_jeu(modele, anglais=False):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    if anglais:
        at.sidebar.toggle(key="anglais").set_value(True).run()
    choisit_page(at, TAB[PAGE] if anglais else PAGE)
    at.sidebar.radio(key="modele").set_value(modele).run()
    at.segmented_control(key="ent_exercice").set_value(
        TAB[E.JEU] if anglais else E.JEU).run()
    return at


def vraie_valeur(at, modele):
    """Le calcul direct du moteur, pour le tirage affiche."""
    produit, reg = E.dv_tirage(at.session_state["ent_dv_graine"],
                               at.selectbox(key="ent_dv_produit").value)
    v = E.dv_valeur(produit, reg, BATES if modele == BT else None, MARGE, RECOUVREMENT,
                    NOMINAL)
    return produit, reg, v * 100


# 1. Tirages ----------------------------------------------------------------------
hors, produits = [], set()
for g in range(600):
    for choix in [E.DV_HASARD] + E.DV_PRODUITS:
        produit, reg = E.dv_tirage(g, choix)
        produits.add(produit)
        if (produit, reg) != E.dv_tirage(g, choix) or (choix != E.DV_HASARD and produit != choix):
            hors.append(f"graine {g} non deterministe")
        for c, v in reg.items():
            if c in CURSEURS:
                a, b, pas = CURSEURS[c]
                if not (a <= v <= b and abs((v - a) / pas - round((v - a) / pas)) < 1e-9):
                    hors.append(f"{c}={v}")
        if produit == E.DV_AC and reg["ac_bar_capital"] > reg["ac_bar_coupon"]:
            hors.append("barriere de capital au-dessus de la barriere de coupon")
controle("Tirages deterministes, dans les bornes et sur les crans des curseurs",
         not hors and produits == set(E.DV_PRODUITS), "; ".join(sorted(set(hors))[:5]))

# 2. Vitesse ----------------------------------------------------------------------
rng = random.Random(7)
for nom, bates in (("BS", None), ("Bates", BATES)):
    pire = 0.0
    for produit in E.DV_PRODUITS:
        for _ in range(6):
            _, reg = E.dv_tirage(rng.randrange(10 ** 6), produit)
            t = time.perf_counter()
            res = E.dv_resultat(produit, reg, bates, MARGE, RECOUVREMENT, NOMINAL)
            pire = max(pire, time.perf_counter() - t)
    controle(f"{nom:6} | moins de 2 s par tirage, valeur + effet de chaque parametre",
             pire < 2.0, f"pire {pire:.2f} s")

# 3. Revelation et page produit, deux modeles -------------------------------------
for modele in (BS, BT):
    for produit in E.DV_PRODUITS:
        at = ouvre_jeu(modele)
        at.selectbox(key="ent_dv_produit").set_value(produit).run()
        _, reg, v = vraie_valeur(at, modele)
        at.number_input(key="ent_dv_estimation").set_value(round(v, 1) + 0.5).run()
        at.button(key="ent_dv_reveler").click().run()
        mt = metriques(at)
        pourquoi = [x.value for x in at.markdown if x.value.startswith("**Pourquoi.**")]
        nom = f"{modele.split()[0]:6} | {produit}"
        controle(f"{nom} : tirage + saisie + révélation",
                 not at.exception and "Vraie valeur" in mt and mt["Tirages joués"] == "1"
                 and len(pourquoi) == 1 and "Ce qui a le plus pesé" in pourquoi[0],
                 erreur(at) or str(mt))
        revele = mt.get("Vraie valeur", "nan")
        controle(f"{nom} : valeur révélée = calcul direct du moteur",
                 abs(nombre(revele) - v) < (0.051 if produit == E.DV_CP else 0.0051),
                 f"{revele} / {v:.4f}")
        at.button(key="ent_dv_refaire").click().run()
        page = at.session_state["page"]
        mp = metriques(at)
        affiche = mp.get(METRIQUE[produit], "nan")
        egal = (abs(nombre(affiche) - v) <= 0.5 if produit == E.DV_CP
                else affiche.replace(" p.a.", "") == revele.replace(" p.a.", ""))
        controle(f"{nom} : la page produit affiche la même valeur",
                 not at.exception and page == E.DV_PAGES[produit] and egal,
                 erreur(at) or f"page {affiche}, jeu {revele}")
        controle(f"{nom} : la barre latérale a pris le marché tiré",
                 all(at.session_state[c] == reg[c] for c in ("taux", "dividende", "vol", "spread"))
                 and at.session_state["trajectoires"] == E.DV_N)

# 4. Verdicts, score, nouveau tirage, remise a zero ------------------------------
at = ouvre_jeu(BS)
textes, ok_nouveau = [], True
for decalage in (0.5, 2.0, 5.0, -5.0):
    _, _, v = vraie_valeur(at, BS)
    while v + decalage < 0:                 # une estimation negative n'existe pas
        at.button(key="ent_dv_nouveau").click().run()
        _, _, v = vraie_valeur(at, BS)
    at.number_input(key="ent_dv_estimation").set_value(round(v + decalage, 2)).run()
    at.button(key="ent_dv_reveler").click().run()
    textes.append(" ".join(x.value for x in (*at.success, *at.info, *at.warning)))
    avant = at.session_state["ent_dv_graine"]
    at.button(key="ent_dv_nouveau").click().run()
    ok_nouveau = (ok_nouveau and at.session_state["ent_dv_graine"] != avant
                  and at.number_input(key="ent_dv_estimation").value is None
                  and not any(m.label == "Vraie valeur" for m in at.metric))
mt = metriques(at)
controle("Verdicts : dans le mille à 1 point, proche à 3, loin au-delà",
         not at.exception and textes[0].startswith("Dans le mille")
         and "Proche" in textes[1] and "Loin" in textes[2] and "trop haut" in textes[2]
         and "Loin" in textes[3] and "trop bas" in textes[3], " | ".join(t[:40] for t in textes))
controle("Score : 4 tirages, 1 dans le mille, 1 proche ; nouveau tirage vide la saisie",
         ok_nouveau and mt["Tirages joués"] == "4" and mt["Dans le mille"] == "1"
         and mt["Proches"] == "1", str(mt))

# revelation sans estimation : affichee, mais hors score ; seconde revelation ignoree
at.button(key="ent_dv_reveler").click().run()
at.number_input(key="ent_dv_estimation").set_value(1.0).run()
at.button(key="ent_dv_reveler").click().run()
mt = metriques(at)
controle("Révélation sans estimation : hors score, l'estimation tardive ne compte pas",
         not at.exception and mt["Tirages joués"] == "4" and mt["Ton estimation"] == "—")
at.button(key="ent_dv_zero").click().run()
controle("Remise à zéro du score",
         not at.exception and metriques(at)["Tirages joués"] == "0"
         and metriques(at)["Écart moyen"] == "—")

# point de repere : la valeur des reglages par defaut
at.checkbox(key="ent_dv_repere").check().run()
repere = [c.value for c in at.caption if c.value.startswith("Point de repère")]
controle("Point de repère affiché", not at.exception and len(repere) == 1, erreur(at))

# 5. La graine passe par le lien --------------------------------------------------
at = AppTest.from_file(APP, default_timeout=300)
at.query_params.update({"page": PAGE, "ent_exercice": E.JEU, "ent_dv_graine": "123456"})
at.run()
controle("La graine du lien redonne le même tirage",
         not at.exception and at.session_state["ent_dv_graine"] == 123456
         and at.query_params.get("ent_dv_graine") in ("123456", ["123456"]), erreur(at))

# 6. Version anglaise -------------------------------------------------------------
FRANCAIS = re.compile(r"[éèêàùç]|\b(le|la|les|des|du|une|est|pour|avec|sur|dans|qui)\b", re.I)
for modele in (BS, TAB[BT]):
    at = ouvre_jeu(modele, anglais=True)
    for produit in E.DV_PRODUITS:
        at.selectbox(key="ent_dv_produit").set_value(TAB.get(produit, produit)).run()
        at.number_input(key="ent_dv_estimation").set_value(5.0).run()
        at.button(key="ent_dv_reveler").click().run()
        at.checkbox(key="ent_dv_repere").check().run()
        textes = [str(getattr(e, "label", "") or "") + " " + str(getattr(e, "value", "") or "")
                  for kind in ("title", "subheader", "caption", "metric", "markdown", "info",
                               "success", "warning")
                  for e in at.get(kind)]
        textes += [str(x) for t in at.table for x in t.value.to_numpy().ravel()]
        restes = [t[:80] for t in textes if FRANCAIS.search(t)]
        controle(f"EN | {modele.split()[0]:6} | {produit} : révélé, sans français visible",
                 not at.exception and not restes, erreur(at) or "; ".join(restes[:3]))
        at.button(key="ent_dv_nouveau").click().run()

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
