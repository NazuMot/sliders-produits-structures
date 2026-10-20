"""
Page « Que vendre aujourd'hui ? » et scenario « Aujourd'hui » de la barre laterale.

    python tests/verifie_aujourdhui.py

1. La page s'ouvre sans exception sous les deux modeles, en francais et en anglais.
2. Le scenario « Aujourd'hui » ecrit dans la barre laterale les valeurs du fichier
   data/marche_du_jour.json (aux crans des curseurs), avec sa date.
3. Les chiffres de la page sont ceux des pages produits avec les memes reglages :
   bouton « Ouvrir un produit » (marche du jour + reglages par defaut de la page),
   et scenario choisi a la main dans la barre laterale.
4. L'appli marche avec un fichier absent, illisible, incomplet, faux ou ancien
   (variable SLIDERS_MARCHE_DU_JOUR, lue par donnees_du_jour.charge).
5. Le script quotidien (scripts/recupere_marche_du_jour.py), sans reseau : refus des
   valeurs invraisemblables, reprise des valeurs de la veille, calculs de vol et de
   dividende ; et le fichier committe est vraisemblable.

Sort avec le code 1 au premier probleme, ce qui fait echouer le test GitHub.
"""

import datetime as dt
import importlib.util
import json
import math
import os
import sys
import tempfile

from streamlit.testing.v1 import AppTest

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(RACINE, "app.py")
sys.path.insert(0, RACINE)

import donnees_du_jour as DJ  # noqa: E402

TAB = json.load(open(os.path.join(RACINE, "traductions", "en.json"), encoding="utf-8"))
PAGE = "Que vendre aujourd'hui ?"
MODELES = ["Black-Scholes + skew", "Bates (vol stochastique + sauts)"]
PRODUITS = ["cp", "brc", "ac", "bonus", "tw", "dc"]
NOMS = ["Capital Protection", "Barrier Reverse Convertible", "Autocall Phoenix",
        "Bonus Certificate", "Twin-Win", "Discount Certificate"]
PAGES_PRODUITS = ["2 - Capital Protection", "3 - Barrier Reverse Convertible",
                  "4 - Autocall Phoenix", "5 - Bonus Certificate", "6 - Twin-Win",
                  "7 - Discount Certificate"]
# la metrique de la page produit qui porte le meme chiffre
METRIQUES = ["Participation offerte", "Coupon équitable", "Équivalent annuel",
             "Rendement si marché latéral", "Participation à la baisse", "Décote"]
CURSEURS = {"taux": "Taux sans risque", "dividende": "Taux de dividende",
            "vol": "Volatilité implicite ATM", "spread": "Spread de crédit"}
echecs = []
TMP = tempfile.mkdtemp()


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def choisit(at, page):
    next(r for r in at.sidebar.radio if page in r.options).set_value(page)
    at.run()


def ouvre(modele=MODELES[0], page=PAGE, anglais=False):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    if anglais:
        at.sidebar.toggle(key="anglais").set_value(True).run()
        page, modele = TAB.get(page, page), TAB.get(modele, modele)
    choisit(at, page)
    at.sidebar.radio(key="modele").set_value(modele).run()
    return at


def erreur(at):
    return at.exception[0].value[:300] if at.exception else ""


def curseur(at, debut):
    return next(s for s in at.sidebar.slider if s.label.startswith(debut)).value


def tableau_produits(at):
    t = next(t.value for t in at.table if "Produit" in t.value.columns)
    return dict(zip(t["Produit"], t["Aujourd'hui"]))


def fichier(contenu):
    """Ecrit un fichier de marche d'essai et le fait lire par l'appli."""
    chemin = os.path.join(TMP, f"marche_{len(os.listdir(TMP))}.json")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(contenu if isinstance(contenu, str) else json.dumps(contenu))
    os.environ["SLIDERS_MARCHE_DU_JOUR"] = chemin
    return chemin


def vrai_fichier():
    os.environ.pop("SLIDERS_MARCHE_DU_JOUR", None)


REEL = DJ.charge()
JOUR = DJ.reglages_du_jour(REEL)

# 1. La page s'ouvre, deux modeles, francais et anglais ----------------------------
for anglais in (False, True):
    for modele in MODELES:
        at = ouvre(modele, anglais=anglais)
        nom = f"{'EN' if anglais else 'FR'} | {modele.split()[0]:6} | page Que vendre aujourd'hui"
        ok = not at.exception and len(at.table) >= 2 and len(at.info) >= 1
        controle(nom, ok, erreur(at) or f"{len(at.table)} tableaux")

# 2. Scenario « Aujourd'hui » -------------------------------------------------------
controle("Fichier réel : date, taux, vol et dividende viennent du fichier",
         JOUR["date"] is not None and all(JOUR["origine"].values()), str(JOUR["origine"]))
attendu = {"taux": round(DJ.valeur(REEL, "taux_3_ans") / 0.25) * 0.25,
           "dividende": round(DJ.valeur(REEL, "dividende") / 0.25) * 0.25,
           "vol": float(round(DJ.valeur(REEL, "vstoxx")))}
controle("Scénario du jour = valeurs du fichier aux crans des curseurs",
         all(abs(JOUR["valeurs"][k] - v) < 1e-9 for k, v in attendu.items()),
         f"{JOUR['valeurs']} / {attendu}")
for modele in MODELES:
    at = ouvre(modele, page="Accueil")
    at.sidebar.selectbox(key="scenario").set_value("Aujourd'hui").run()
    valeurs = {k: curseur(at, v) for k, v in CURSEURS.items()}
    legende = " ".join(c.value for c in at.sidebar.caption)
    ok = (not at.exception and all(abs(valeurs[k] - JOUR["valeurs"][k]) < 1e-9 for k in valeurs)
          and f"Données du {DJ.date_fr(JOUR['date'])}" in legende)
    if modele == MODELES[1]:
        ok = ok and curseur(at, "Vol long terme") == JOUR["valeurs"]["vol_lt"]
    controle(f"{modele.split()[0]:6} | scénario « Aujourd'hui » : curseurs et date dans la barre",
             ok, erreur(at) or str(valeurs))

# 3. Memes chiffres que les pages produits ------------------------------------------
for modele in MODELES:
    at = ouvre(modele)
    page = tableau_produits(at)
    ecarts = []
    for p, nom, cle in zip(PRODUITS, NOMS, METRIQUES):
        at.button(key=f"aj_ouvre_{p}").click().run()
        m = {x.label: x.value for x in at.metric}
        if at.exception or m.get(cle) != page[nom]:
            ecarts.append(f"{nom} : page {page[nom]}, produit {m.get(cle)} {erreur(at)}")
        choisit(at, PAGE)
    controle(f"{modele.split()[0]:6} | bouton Ouvrir : les 6 pages produits affichent le même "
             "chiffre", not ecarts, "; ".join(ecarts) or ", ".join(page.values()))

    # le scenario seul (sans le bouton), sur une page produit aux reglages par defaut
    at = ouvre(modele, page=PAGES_PRODUITS[1])
    at.sidebar.selectbox(key="scenario").set_value("Aujourd'hui").run()
    brc = {x.label: x.value for x in at.metric}.get(METRIQUES[1])
    choisit(at, PAGE)
    controle(f"{modele.split()[0]:6} | scénario choisi à la main : même coupon de BRC",
             not at.exception and brc == tableau_produits(at)[NOMS[1]],
             f"{brc} / {tableau_produits(at)[NOMS[1]] if not at.exception else erreur(at)}")

# la reponse d'entretien cite les chiffres de la page
at = ouvre()
page = tableau_produits(at)
reponse = at.info[0].value
controle("Réponse d'entretien : chiffrée avec les valeurs de la page, 4 phrases",
         reponse.startswith("**À dire en entretien**") and any(v in reponse for v in page.values())
         and reponse.count(". ") >= 3 and ("scénarios simulés" in reponse or "défaut" in reponse),
         reponse[:160])

# 4. Fichier absent, illisible, incomplet, faux, ancien --------------------------------
complet = json.load(open(DJ.FICHIER, encoding="utf-8"))
cas = {
    "absent": os.path.join(TMP, "n_existe_pas.json"),
    "illisible": "{pas du json",
    "liste": "[1, 2, 3]",
    "vide": {},
    "incomplet (indice et VSTOXX seuls)": {
        "date_donnees": complet["date_donnees"],
        "champs": {k: complet["champs"][k] for k in ("eurostoxx50", "vstoxx")}},
    "sans VSTOXX": {**complet, "champs": {k: v for k, v in complet["champs"].items()
                                          if k != "vstoxx"}},
    "valeurs fausses": {"date_donnees": 12, "habituel": 5, "anciens": "x",
                        "champs": {"vstoxx": {"valeur": "abc", "date": 3},
                                   "taux_3_ans": "x", "dividende": {"valeur": None},
                                   "eurostoxx50": {"valeur": float("nan")}}},
    "ancien": {**complet, "date_donnees": "2026-01-02"},
}
for nom, contenu in cas.items():
    if nom == "absent":
        os.environ["SLIDERS_MARCHE_DU_JOUR"] = contenu
    else:
        fichier(contenu)
    d = DJ.charge()
    reg = DJ.reglages_du_jour(d)
    for modele in (MODELES if nom == "absent" else MODELES[:1]):
        at = ouvre(modele)
        alertes = " ".join(w.value for w in at.warning)
        ok = not at.exception and len(at.table) >= 2
        if nom in ("absent", "illisible", "liste", "vide", "valeurs fausses"):
            ok = ok and "introuvable" in alertes and reg["valeurs"]["taux"] == 3.0 \
                and reg["valeurs"]["vol"] == 22.0 and reg["valeurs"]["dividende"] == 2.5
        elif nom.startswith("incomplet"):
            ok = ok and "Donnée absente" in alertes and reg["valeurs"]["taux"] == 3.0 \
                and reg["valeurs"]["vol"] == round(complet["champs"]["vstoxx"]["valeur"])
        elif nom == "sans VSTOXX":
            ok = ok and "VSTOXX indisponible" in alertes and reg["origine"]["vol"] == \
                "vol_realisee_3m"
        elif nom == "ancien":
            ok = ok and "n'a pas tourné depuis" in alertes
        controle(f"Fichier {nom} | {modele.split()[0]:6} : la page s'ouvre et le dit", ok,
                 erreur(at) or alertes[:120])
    at = ouvre(page="Accueil")
    at.sidebar.selectbox(key="scenario").set_value("Aujourd'hui").run()
    controle(f"Fichier {nom} : le scénario « Aujourd'hui » s'applique sans erreur",
             not at.exception and curseur(at, "Taux sans risque") == reg["valeurs"]["taux"],
             erreur(at))
vrai_fichier()

controle("date_fr : formats du fichier",
         DJ.date_fr("2026-10-19") == "19/10/2026" and DJ.date_fr("2026-09") == "09/2026"
         and DJ.date_fr(None) is None and DJ.date_fr("n'importe quoi") is None)
controle("Ancienneté des données : en jours calendaires",
         DJ.anciennete({"date_donnees": "2026-10-19"}, dt.date(2026, 10, 22)) == 3
         and DJ.anciennete({}) is None)

# 5. Script quotidien, sans reseau -------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "recupere_marche_du_jour", os.path.join(RACINE, "scripts", "recupere_marche_du_jour.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def c(v, d="2026-10-19"):
    return {"valeur": v, "date": d, "source": "essai"}


champs = {"eurostoxx50": {**c(6000.0), "controle_yahoo": 6100.0}, "vstoxx": c(250.0),
          "estr": c(4.5), "taux_depot_bce": c(2.5), "taux_3_ans": c(3.0)}
refus = R.valide(champs)
controle("Script : valeurs invraisemblables refusées (VSTOXX 250, Yahoo à 1,7 %, €STR loin du "
         "taux de dépôt)", set(champs) == {"taux_depot_bce", "taux_3_ans"} and len(refus) == 3,
         "; ".join(refus))
sortie = R.assemble({"taux_3_ans": c(3.0)}, {}, {
    "champs": {"vstoxx": c(18.0, "2026-10-16"), "taux_3_ans": c(2.0, "2026-10-16")},
    "habituel": {"vstoxx": {"mediane": 19.6, "rang": 0.4, "n": 10}}})
controle("Script : une source en panne garde la valeur de la veille, avec sa date",
         sortie["champs"]["vstoxx"]["date"] == "2026-10-16" and sortie["anciens"] == ["vstoxx"]
         and sortie["champs"]["taux_3_ans"]["valeur"] == 3.0
         and sortie["habituel"]["vstoxx"]["mediane"] == 19.6
         and sortie["date_donnees"] == "2026-10-19", json.dumps(sortie)[:200])
serie = [(i, 100 * math.exp(0.01 * (i % 2))) for i in range(64)]
v = R.vol_realisee(serie, 63)
controle("Script : vol réalisée annualisée (rendements de +/-1 % par jour -> ~15,9 %)",
         abs(v - 0.01 * math.sqrt(252) * 100 * math.sqrt(64 / 63)) < 0.05, f"{v:.2f} %")
div = R.dividendes_12m(R.charge_historique())
med = R.mediane([x for k, x in div.items() if k >= R.DEBUT_HABITUEL])
controle("Script : dividende net médian depuis 2004 entre 2 et 3,5 %", 2.0 <= med <= 3.5,
         f"{med:.2f} %")

# le fichier committe
h = R.charge_historique()
dernier = h[max(h)][0]
manque = [k for k in R.BORNES if DJ.valeur(REEL, k) is None]
hors = [k for k, (a, b) in R.BORNES.items() if DJ.valeur(REEL, k) is not None
        and not a <= DJ.valeur(REEL, k) <= b]
ok = (not manque and not hors and DJ.date_donnees(REEL) is not None
      and abs(DJ.valeur(REEL, "eurostoxx50") / dernier - 1) < 0.3
      and abs(DJ.valeur(REEL, "taux_3_ans") - DJ.valeur(REEL, "estr")) < 2.5
      and DJ.valeur(REEL, "vstoxx") > 0.8 * DJ.valeur(REEL, "vol_realisee_3m")
      and all(DJ.habituel(REEL, k)[1] is not None for k in DJ.HABITUEL_DEFAUT))
controle("data/marche_du_jour.json : complet et vraisemblable", ok,
         f"données du {REEL.get('date_donnees')}, manquants {manque}, hors bornes {hors}")

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
