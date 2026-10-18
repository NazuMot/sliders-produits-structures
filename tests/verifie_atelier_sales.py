"""
Verification de la page Atelier Sales.

    python tests/verifie_atelier_sales.py

0. Conventions : chaque widget a une cle « as_... », le lien partageable
   (bind="query-params") et une aide ; le titre a son aide longue ; le lien
   rouvre bien la page comme on l'a laissee.
1. Chaque section s'ouvre pour chaque produit, sous les deux modeles.
2. Le client demande : la page affiche exactement la reponse de sales.py,
   pour chaque parametre cherche ; cas impossibles ; « Retenir » inscrit la
   structure arrondie et la demande reste satisfaite.
3. Ou va l'argent : coupon et cascade de sales.py, somme = nominal, budget
   qui retombe a zero, deux reseaux, droite de la retro exacte.
4. Revendre : valeur a l'emission = nominal - marge - retro (982 EUR), meme
   valeur que sales.py, sens des chocs, barriere touchee, autocall rappele.
5. Pitch : 3 a 5 phrases, toujours le scenario de perte et le risque
   emetteur, pour chaque produit, profil et vue de marche.
6. Objections : chaque objection s'affiche pour chaque produit et chaque
   profil, sous les deux modeles, reponse cachee puis revelee ; les
   chiffres cites sont ceux des autres sections (coupon, frais, valeur de
   revente, probabilite de perte, duree de vie, risque emetteur).

Sort avec le code 1 au premier probleme, comme verifie.py.
"""

import ast
import math
import os
import re
import sys
import time

import numpy as np

from streamlit.testing.v1 import AppTest

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(RACINE, "app.py")
PAGE = os.path.join(RACINE, "vues", "atelier_sales.py")
sys.path.insert(0, RACINE)

import backtest as B
import graphiques as G
import sales as S
from vues import atelier_sales as A

BS, BATES = "Black-Scholes + skew", "Bates (vol stochastique + sauts)"
MODELES = [BS, BATES]
RETRO = 0.01                          # retrocession par defaut de la page
MK = {BS: S.Marche(retro=RETRO), BATES: S.Marche(modele="bates", retro=RETRO)}
echecs = []


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def ouvre(produit, section, modele=BS, reglages=None):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    next(r for r in at.sidebar.radio if "Atelier Sales" in r.options).set_value("Atelier Sales")
    at.sidebar.radio(key="modele").set_value(modele)
    at.run()
    at.segmented_control(key="as_produit").set_value(produit).run()
    at.segmented_control(key="as_section").set_value(section).run()
    for regle in reglages or []:
        regle(at)
        at.run()
    return at


def metriques(at):
    return {m.label: m.value for m in at.metric}


def erreur(at):
    return at.exception[0].value[:300] if at.exception else ""


def pa(x):
    return f"{x * 100:,.2f}% p.a."


def eur(x):
    return f"{x:,.0f}".replace(",", " ") + " EUR"


# 0. Conventions de la page ---------------------------------------------------
print("0. Conventions")
WIDGETS = {"slider", "select_slider", "selectbox", "radio", "checkbox", "number_input",
           "segmented_control", "pills", "toggle", "text_input", "multiselect"}


def cle_ok(n):
    """La cle commence par as_ (texte, f-string ou choix entre deux textes)."""
    if isinstance(n, ast.Constant):
        return isinstance(n.value, str) and n.value.startswith("as_")
    if isinstance(n, ast.JoinedStr):
        return isinstance(n.values[0], ast.Constant) and n.values[0].value.startswith("as_")
    if isinstance(n, ast.IfExp):
        return cle_ok(n.body) and cle_ok(n.orelse)
    return False


arbre = ast.parse(open(PAGE).read())
fautes, n_widgets, titre_ok = [], 0, False
for n in ast.walk(arbre):
    if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
        continue
    kw = {k.arg: k.value for k in n.keywords}
    if n.func.attr == "title":
        titre_ok = "help" in kw and len(ast.unparse(kw["help"])) > 400
    if n.func.attr not in WIDGETS:
        continue
    n_widgets += 1
    if not ("key" in kw and cle_ok(kw["key"])):
        fautes.append(f"ligne {n.lineno} : clé")
    if not (isinstance(kw.get("bind"), ast.Constant) and kw["bind"].value == "query-params"):
        fautes.append(f"ligne {n.lineno} : bind")
    if "help" not in kw:
        fautes.append(f"ligne {n.lineno} : help")
controle(f"Les {n_widgets} widgets ont une clé as_..., bind='query-params' et une aide",
         n_widgets > 20 and not fautes, "; ".join(fautes))
controle("Le titre a une aide longue", titre_ok)

at = AppTest.from_file(APP, default_timeout=300)
for k, v in {"page": "Atelier Sales", "as_produit": A.CP, "as_section": A.PITCH,
             "as_profil": A.INSTIT, "as_cp_maturite": "8.0"}.items():
    at.query_params[k] = v
at.run()
controle("Le lien partageable rouvre produit, section, profil et curseurs",
         not at.exception and at.segmented_control(key="as_produit").value == A.CP
         and at.segmented_control(key="as_section").value == A.PITCH
         and at.slider(key="as_cp_maturite").value == 8.0
         and "8 ans" in at.code[0].value, erreur(at))


# 1. Toutes les sections, tous les produits, deux modeles -----------------------
print("\n1. Ouverture de chaque section")
for modele in MODELES:
    for produit in A.PRODUITS:
        for section in A.SECTIONS:
            t = time.time()
            at = ouvre(produit, section, modele)
            controle(f"{modele.split()[0]:6} | {produit[:18]:18} | {section}", not at.exception,
                     erreur(at) or f"{time.time() - t:.1f}s")
at = ouvre(A.AC, A.ARGENT, BATES,
           [lambda a: a.select_slider(key="as_ac_sous_jacents").set_value(3)])
controle("Bates | Autocall worst-of 3 : la page s'ouvre", not at.exception, erreur(at))


# 2. Le client demande ----------------------------------------------------------
print("\n2. Le client demande")


def demande(produit, option, cible, modele=BS, autres=()):
    """Ouvre la section, choisit la demande et la cible, renvoie (page, reponse de sales.py)."""
    cle_dem = {A.BRC: "as_dem_brc", A.AC: "as_dem_ac", A.CP: "as_dem_cp"}[produit]
    cle_cible = {A.BRC: "as_dem_cpn_brc", A.AC: "as_dem_cpn_ac", A.CP: "as_dem_part"}[produit]
    libelle = next(k for k, v in A.CHERCHE[produit].items() if v == option)
    at = ouvre(produit, A.DEMANDE, modele, [
        lambda a: a.selectbox(key=cle_dem).set_value(libelle),
        lambda a: a.number_input(key=cle_cible).set_value(cible * 100)] + list(autres))
    s = {A.BRC: A.Structure(A.BRC, 1.0), A.AC: A.Structure(A.AC, 3.0),
         A.CP: A.Structure(A.CP, 5.0)}[produit]
    return at, A._reponse.__wrapped__(MK[modele], s, option, cible), s


def affiche_comme_sales(at, rep, option):
    m = metriques(at)
    trouve = next(v for k, v in m.items() if k.startswith(A.NOM_PARAMETRE[option]))
    attendu = A._duree(rep.valeur) if option == "maturite" else f"{rep.valeur * 100:,.1f}%"
    msg = [x.value for x in (at.success if rep.possible else at.error)]
    return trouve == attendu and any(rep.message in x for x in msg), f"{trouve} / {attendu}"


CAS = [(A.BRC, "barriere", 0.08), (A.BRC, "strike", 0.07), (A.BRC, "maturite", 0.08),
       (A.AC, "barriere_capital", 0.05), (A.AC, "trigger", 0.05),
       (A.CP, "cap", 0.60), (A.CP, "protection", 0.60)]
for produit, option, cible in CAS:
    for modele in MODELES:
        at, rep, s = demande(produit, option, cible, modele)
        ok, detail = affiche_comme_sales(at, rep, option)
        controle(f"{modele.split()[0]:6} | {produit[:18]:18} | {option} : la page affiche "
                 f"la réponse de sales.py", not at.exception and ok, erreur(at) or detail)

at, rep, s = demande(A.BRC, "barriere", 0.08)
curseur, s_client = A.arrondi_client(s, "barriere", rep.valeur)
attendu = S.coupon_brc(MK[BS], 1.0, curseur / 100)
m = metriques(at)
controle("BRC 8 % : arrondi au point supérieur, coupon recalcule et demande satisfaite",
         m["Arrondi pour le client"] == f"{curseur}%" and curseur == math.ceil(rep.valeur * 100)
         and m["Obtenu à l'arrondi"] == pa(attendu) and attendu >= 0.08,
         f"{rep.valeur * 100:.2f} % -> {curseur} %, {attendu * 100:.2f} %")
phrase = next(x.value for x in at.markdown if x.value.startswith("**À dire au client.**"))
controle("BRC 8 % : la phrase client cite la structure arrondie",
         f"{100 - curseur:.0f} %" in phrase and f"{attendu * 100:,.2f} %" in phrase, phrase[:160])

at, rep, s = demande(A.BRC, "barriere", 0.30)
m = metriques(at)
controle("BRC 30 % : impossible, message clair et meilleure valeur atteignable",
         not rep.possible and any("Impossible" in x.value for x in at.error)
         and m["Obtenu"] == pa(rep.obtenu) and not at.success
         and any("n'est pas faisable" in x.value for x in at.markdown),
         f"au mieux {m['Obtenu']} avec {m['Barrière trouvée']}")
at, rep, s = demande(A.CP, "cap", 3.0)
controle("Capital Protection 300 % de participation : impossible, meilleur cap affiché",
         not rep.possible and any("Impossible" in x.value for x in at.error)
         and metriques(at)["Obtenu"] == f"{rep.obtenu * 100:,.0f}%", metriques(at)["Obtenu"])
at, rep, s = demande(A.AC, "barriere_capital", 0.11, BATES)
controle("Bates | Autocall 11 % : impossible avec 1 % de rétro, meilleure barrière affichée",
         not rep.possible and any("Impossible" in x.value for x in at.error),
         f"au mieux {rep.obtenu * 100:.2f} %")
at, rep, s = demande(A.BRC, "barriere", 0.01)
controle("BRC 1 % : cible dépassée dès la barrière la plus basse, dit au client",
         rep.possible and any("Bonne nouvelle" in x.value for x in at.markdown))

for produit, option, cible in [(A.BRC, "barriere", 0.07), (A.AC, "barriere_capital", 0.05),
                               (A.CP, "cap", 0.60), (A.CP, "protection", 0.60)]:
    at, rep, s = demande(produit, option, cible)
    curseur, s_client = A.arrondi_client(s, option, rep.valeur)
    at.button(key="as_retenir").click().run()
    cle = A.CURSEUR[option][0]
    lu = at.slider(key=cle).value
    at.segmented_control(key="as_section").set_value(A.ARGENT).run()
    m = metriques(at)
    if produit == A.CP:
        attendu = S.participation_cp(MK[BS], s_client.T, s_client.protection, s_client.cap)
        ok = m["Participation offerte"] == f"{attendu * 100:,.0f}%" and attendu >= cible - 1e-9
        lu_res = m["Participation offerte"]
    elif produit == A.AC:
        attendu = 4 * S.coupon_autocall(MK[BS], 3.0, 4, 1.0, 0.70, s_client.bar_capital)
        ok = m["Coupon offert"] == pa(attendu) and attendu >= cible
        lu_res = m["Coupon offert"]
    else:
        attendu = S.coupon_brc(MK[BS], 1.0, s_client.barriere)
        ok = m["Coupon offert"] == pa(attendu) and attendu >= cible
        lu_res = m["Coupon offert"]
    controle(f"Retenir ({produit[:18]}, {option}) : curseur à {curseur}, la section suivante "
             f"part de cette structure", not at.exception and lu == curseur and ok,
             erreur(at) or f"curseur {lu}, {lu_res}")


# 3. Ou va l'argent -------------------------------------------------------------
print("\n3. Où va l'argent")
s_brc, s_ac, s_cp = A.Structure(A.BRC, 1.0), A.Structure(A.AC, 3.0), A.Structure(A.CP, 5.0)
for modele in MODELES:
    mk = MK[modele]
    for s, libelle, valeur in [
            (s_brc, "Coupon offert", lambda c: pa(c.resultat)),
            (s_ac, "Coupon offert", lambda c: pa(c.infos["coupon_pa"])),
            (s_cp, "Participation offerte", lambda c: f"{c.resultat * 100:,.0f}%")]:
        at = ouvre(s.produit, A.ARGENT, modele)
        c = A._cascade.__wrapped__(mk, s)
        m = metriques(at)
        total = at.table[0].value.iloc[-1]
        controle(f"{modele.split()[0]:6} | {s.produit[:18]:18} | résultat et total = sales.py",
                 m[libelle] == valeur(c) and total["EUR"] == eur(mk.nominal)
                 and abs(c.somme - mk.nominal) < 1e-6
                 and m["Rétrocession"] == eur(next(l.montant for l in c.lignes if l.cle == "retro"))
                 and m["Marge banque"] == eur(next(l.montant for l in c.lignes if l.cle == "marge")),
                 f"{m[libelle]}, total {total['EUR']}")
        ca = A._cascade.__wrapped__(mk.avec(retro=0.005), s)
        cb = A._cascade.__wrapped__(mk.avec(retro=0.015), s)
        va, vb = A._resultat_pa(s, ca), A._resultat_pa(s, cb)
        fa = (lambda x: f"{x * 100:,.0f}%") if s.produit == A.CP else pa
        lus = [m[k] for k in m if k.startswith("Réseau")]
        c0 = A._resultat_pa(s, A._cascade.__wrapped__(mk.avec(retro=0.0), s))
        droite = c0 - 1.5 * c.infos["cout_retro_par_pt"]
        controle(f"{modele.split()[0]:6} | {s.produit[:18]:18} | deux réseaux = sales.py, "
                 "droite de la rétro exacte",
                 lus == [fa(va), fa(vb)] and vb < va and abs(max(droite, 0.0) - vb) < 1e-9,
                 f"A {lus[0]}, B {lus[1]}")

mk = MK[BS].avec(spread=150)
for s in (s_brc, s_ac, A.Structure(A.CP, 5.0, protection=0.95, cap=1.6)):
    c = A._cascade.__wrapped__(mk, s)
    etapes = A._budget(c)
    controle(f"Budget {s.produit[:18]:18} (spread 150 bp) : toutes les lignes, somme nulle",
             len(etapes) == len(c.lignes) and abs(sum(e["valeur"] for e in etapes)) < 1e-6,
             f"{len(etapes)} lignes")

at = ouvre(A.BRC, A.ARGENT, BS, [lambda a: a.slider(key="as_retro").set_value(0.0)])
c = A._cascade.__wrapped__(MK[BS].avec(retro=0.0), s_brc)
controle("Rétro nulle : le coupon est celui de la page BRC (7.84 % p.a., vol locale)",
         metriques(at)["Coupon offert"] == pa(S.coupon_brc(S.Marche(), 1.0, 0.70)) == "7.84% p.a.",
         metriques(at)["Coupon offert"])


# 4. Revendre en cours de vie ---------------------------------------------------
print("\n4. Revendre en cours de vie")


def revente(produit, reglages, modele=BS):
    at = ouvre(produit, A.REVENTE, modele, reglages)
    return at, metriques(at)


a_zero = [lambda a: a.slider(key="as_sec_mois").set_value(0),
          lambda a: a.slider(key="as_sec_niveau").set_value(100)]
# autocall : marge et retro sur la duree de vie attendue (~0,9 an), pas sur 3 ans
for produit, attendu in [(A.BRC, 982), (A.AC, 983), (A.CP, 910)]:
    at, m = revente(produit, a_zero)
    controle(f"{produit[:18]:18} à l'émission : valeur = nominal - marge - rétro = {attendu} EUR",
             not at.exception and m["Valeur juste"] == eur(attendu)
             and m["Prix de rachat"] == eur(attendu - 5), erreur(at) or m.get("Valeur juste"))

at, m = revente(A.BRC, [])
v = S.valeur_secondaire_brc(MK[BS], MK[BS], 1.0, 0.70, 1.0, True, 0.5, 0.90)
controle("BRC a 6 mois, sous-jacent a 90 % : valeur et rachat = sales.py",
         m["Valeur juste"] == eur(v.valeur) and m["Prix de rachat"] == eur(v.prix_rachat)
         and m["Bilan si revente"] == G.eur(v.bilan_client, signe=True),
         f"{m['Valeur juste']}, rachat {m['Prix de rachat']}")


def valeur(m):
    return float(m["Valeur juste"].replace(" EUR", "").replace(" ", ""))


base = valeur(m)
_, m2 = revente(A.BRC, [lambda a: a.checkbox(key="as_sec_touchee").check()])
controle("BRC : barrière déjà touchée -> valeur plus basse", valeur(m2) < base,
         f"{base:.0f} -> {valeur(m2):.0f} EUR")
_, m2 = revente(A.BRC, [lambda a: a.slider(key="as_sec_dvol").set_value(8)])
controle("BRC : vol +8 pts -> valeur plus basse (le client est vendeur d'options)",
         valeur(m2) < base, f"{base:.0f} -> {valeur(m2):.0f} EUR")
_, m2 = revente(A.BRC, [lambda a: a.slider(key="as_sec_dspread").set_value(200)])
controle("BRC : spread +200 bp -> valeur plus basse", valeur(m2) < base,
         f"{base:.0f} -> {valeur(m2):.0f} EUR")
_, m0 = revente(A.CP, [])
_, m2 = revente(A.CP, [lambda a: a.slider(key="as_sec_dtaux").set_value(100)])
controle("Capital Protection : taux +100 bp -> valeur plus basse", valeur(m2) < valeur(m0),
         f"{valeur(m0):.0f} -> {valeur(m2):.0f} EUR")

at, m = revente(A.AC, [lambda a: a.slider(key="as_sec_mois").set_value(14),
                       lambda a: a.number_input(key="as_sec_obs2").set_value(85),
                       lambda a: a.number_input(key="as_sec_obs3").set_value(60)])
v = S.valeur_secondaire_autocall(MK[BS], MK[BS], 3.0, 4, 1.0, 0.70, 0.60, True, 14 / 12, 0.90,
                                 [0.95, 0.85, 0.60, 0.95])
controle("Autocall a 14 mois, 4 observations passées : valeur et coupons = sales.py",
         not at.exception and m["Valeur juste"] == eur(v.valeur)
         and v.coupons_encaisses > 0 and v.infos["coupons_manques"] == 0,
         erreur(at) or f"{m.get('Valeur juste')}, coupons {v.coupons_encaisses:.0f} EUR")
at, m = revente(A.AC, [lambda a: a.slider(key="as_sec_mois").set_value(14),
                       lambda a: a.number_input(key="as_sec_obs2").set_value(104)])
controle("Autocall : observation au-dessus du trigger -> produit rappelé, rien à racheter",
         not at.exception and any("rappelé à l'observation 2" in x.value for x in at.success),
         erreur(at))
at, m = revente(A.BRC, [lambda a: a.slider(key="as_sec_mois").set_value(11),
                        lambda a: a.slider(key="as_brc_maturite").set_value(0.5)])
controle("BRC : maturité raccourcie après coup, la date reste dans la vie du produit",
         not at.exception, erreur(at))


# 5. Pitch client ---------------------------------------------------------------
print("\n5. Pitch client")
structures = [s_brc, A.Structure(A.BRC, 2.0, barriere=0.6, strike=0.9, continu=False),
              s_ac, A.Structure(A.AC, 4.0, par_an=2, trigger=0.95, bar_coupon=0.8,
                                bar_capital=0.5, memoire=False, n_actifs=3, correlation=0.4),
              s_cp, A.Structure(A.CP, 6.0, protection=0.9, cap=1.5)]
fautes, n = [], 0
for mk in (MK[BS], MK[BATES].avec(spread=120)):
    for s in structures:
        f = A.faits(mk, s)
        for profil in A.PROFILS:
            for vue in A.VUES:
                n += 1
                p = A.pitch(s, profil, vue, f)
                roles = dict(p["phrases"])
                nom = f"{s.produit[:12]} {profil[:12]} {vue}"
                if not 3 <= len(p["phrases"]) <= 5:
                    fautes.append(f"{nom} : {len(p['phrases'])} phrases")
                if "perte" not in roles or not any(ch.isdigit() for ch in roles["perte"]):
                    fautes.append(f"{nom} : pas de scénario de perte chiffré")
                if 'émetteur' not in roles.get("emetteur", "") or "%" not in roles["emetteur"]:
                    fautes.append(f"{nom} : risque émetteur absent")
                if mk.spread and profil != A.PARTICULIER and "bp" not in roles["emetteur"]:
                    fautes.append(f"{nom} : spread non cite")
                if p["adequation"] not in ("oui", "partiel", "non"):
                    fautes.append(f"{nom} : adéquation")
controle(f"{n} pitchs : 3 à 5 phrases, perte chiffrée et risque émetteur toujours nommes",
         not fautes, "; ".join(fautes[:5]))

f = A.faits(MK[BS], s_brc)
mal = [(s_brc, "Forte hausse"), (s_brc, "Forte baisse"), (s_ac, "Forte baisse"),
       (s_cp, "Stabilité")]
controle("Les vues contraires au produit sont dites au client (adéquation 'non')",
         all(A.pitch(s, A.PARTICULIER, v, A.faits(MK[BS], s))["adequation"] == "non"
             for s, v in mal))
p = A.pitch(s_brc, A.CGP, "Stabilité", f)
sans = (f["resultat"] + RETRO * 100 * f["cout_retro"]) * 100
controle("Pitch CGP : coupon sans rétro = coupon de la page BRC",
         f"{sans:,.2f} %" in dict(p["phrases"])["retro"]
         and abs(sans / 100 - S.coupon_brc(S.Marche(), 1.0, 0.70)) < 1e-9, f"{sans:.2f} %")

for produit in A.PRODUITS:
    for profil in A.PROFILS:
        at = ouvre(produit, A.PITCH, BS,
                   [lambda a, p=profil: a.segmented_control(key="as_profil").set_value(p)])
        texte = at.code[0].value if at.code else ""
        attendu = " ".join(t for _, t in A.pitch(
            {A.BRC: s_brc, A.AC: s_ac, A.CP: s_cp}[produit], profil, "Stabilité",
            A.faits(MK[BS], {A.BRC: s_brc, A.AC: s_ac, A.CP: s_cp}[produit]))["phrases"])
        controle(f"Page | {produit[:18]:18} | {profil[:12]:12} : pitch affiche = modèle",
                 not at.exception and texte == attendu and 'émetteur' in texte,
                 erreur(at) or texte[:80])



# 6. Objections clients ---------------------------------------------------------
print("\n6. Objections clients")
COUPE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý«])")       # fin de phrase


def phrases(texte):
    return len(COUPE.split(texte))


h = B.charge_historique()


def cours(jour):
    return h.prix[np.searchsorted(h.dates, np.datetime64(jour), side="right") - 1]


annee = (h.dates > np.datetime64("2007-12-31")) & (h.dates <= np.datetime64("2008-12-31"))
controle("Krach 2008 cité = données réelles (-44 % sur l'année, plus bas à 49 %)",
         abs(1 - cours("2008-12-31") / cours("2007-12-31") - A.BAISSE_2008) < 5e-4
         and abs(h.prix[annee].min() / cours("2007-12-31") - A.PLUS_BAS_2008) < 5e-4,
         f"{1 - cours('2008-12-31') / cours('2007-12-31'):.4f}, "
         f"{h.prix[annee].min() / cours('2007-12-31'):.4f}")

fautes, n = [], 0
for mk in (MK[BS], MK[BATES].avec(spread=120)):
    for s in structures + [A.Structure(A.BRC, 1.0, barriere=0.45),
                           A.Structure(A.AC, 3.0, bar_capital=0.5)]:
        f = A.faits(mk, s)
        for profil in A.PROFILS:
            for cle in A.LISTE_OBJECTIONS[s.produit]:
                n += 1
                nom = f"{s.produit[:12]} {profil[:12]} {cle}"
                try:
                    o = A.objection(s, profil, cle, f)
                except Exception as e:                       # noqa: BLE001
                    fautes.append(f"{nom} : {e!r}")
                    continue
                if not 2 <= phrases(o["reponse"]) <= 4:
                    fautes.append(f"{nom} : {phrases(o['reponse'])} phrases")
                if not any(ch.isdigit() for ch in o["chiffre"]):
                    fautes.append(f"{nom} : pas de chiffre à citer")
                if len(o["erreur"]) < 40:
                    fautes.append(f"{nom} : erreur à éviter absente")
controle(f"{n} objections (6 + 2 structures x 3 profils x 2 marchés) : 2 à 4 phrases, un "
         "chiffre, une erreur à éviter", not fautes, "; ".join(fautes[:5]))
controle("8 à 10 objections par produit, dont le rappel pour l'autocall seul",
         all(8 <= len(v) <= 10 for v in A.LISTE_OBJECTIONS.values())
         and "rappel" in A.LISTE_OBJECTIONS[A.AC]
         and "rappel" not in A.LISTE_OBJECTIONS[A.BRC] + A.LISTE_OBJECTIONS[A.CP])

# chaque objection, chaque produit, chaque profil, deux modeles : affichee = modele
CLE_OBJ = {A.BRC: "as_obj_brc", A.AC: "as_obj_ac", A.CP: "as_obj_cp"}
S_DEFAUT = {A.BRC: s_brc, A.AC: s_ac, A.CP: s_cp}


def reponse_affichee(at):
    return next((x.value for x in at.markdown if x.value.startswith("**À dire au client.**")),
                None)


for modele in MODELES:
    for produit in A.PRODUITS:
        t0, fautes, n = time.time(), [], 0
        at = ouvre(produit, A.OBJECTIONS, modele)
        f = A.faits(MK[modele], S_DEFAUT[produit])
        for profil in A.PROFILS:
            at.segmented_control(key="as_profil").set_value(profil).run()
            for cle in A.LISTE_OBJECTIONS[produit]:
                n += 1
                at.selectbox(key=CLE_OBJ[produit]).set_value(A.QUESTIONS[cle]).run()
                cachee = reponse_affichee(at) is None and not at.toggle(key="as_obj_voir").value
                at.toggle(key="as_obj_voir").set_value(True).run()
                o = A.objection(S_DEFAUT[produit], profil, cle, f)
                if at.exception:
                    fautes.append(f"{profil[:12]} {cle} : {erreur(at)}")
                elif not cachee:
                    fautes.append(f"{profil[:12]} {cle} : réponse visible avant d'être demandée")
                elif (reponse_affichee(at) != "**À dire au client.** " + o["reponse"]
                      or not any(o["chiffre"] in x.value for x in at.info)
                      or not any(o["erreur"] in x.value for x in at.warning)):
                    fautes.append(f"{profil[:12]} {cle} : affiché différent du modèle")
        controle(f"{modele.split()[0]:6} | {produit[:18]:18} | {n} objections affichées, "
                 "cachées puis révélées", not fautes,
                 "; ".join(fautes[:3]) or f"{time.time() - t0:.0f}s")

at = ouvre(A.AC, A.OBJECTIONS, BS)
avant = at.selectbox(key="as_obj_ac").value
at.toggle(key="as_obj_voir").set_value(True).run()
at.button(key="as_obj_hasard").click().run()
controle("Objection au hasard : une autre objection, réponse de nouveau cachée",
         not at.exception and at.selectbox(key="as_obj_ac").value != avant
         and at.selectbox(key="as_obj_ac").value in [A.QUESTIONS[k] for k in A.LISTE_OBJECTIONS[A.AC]]
         and not at.toggle(key="as_obj_voir").value and reponse_affichee(at) is None, erreur(at))


# les chiffres cites = ceux des autres sections de la page
def objection_page(produit, cle, profil=A.PARTICULIER, reglages=()):
    at = ouvre(produit, A.OBJECTIONS, BS, list(reglages) + [
        lambda a: a.segmented_control(key="as_profil").set_value(profil),
        lambda a: a.selectbox(key=CLE_OBJ[produit]).set_value(A.QUESTIONS[cle]),
        lambda a: a.toggle(key="as_obj_voir").set_value(True)])
    texte = (reponse_affichee(at) or "") + " ".join(x.value for x in at.info)
    return at, texte


def nombre(txt):
    return float(re.search(r"-?[0-9][0-9 ]*(\.[0-9]+)?", txt).group(0).replace(" ", ""))


for produit in A.PRODUITS:
    m = metriques(ouvre(produit, A.ARGENT, BS))
    resultat = m["Participation offerte" if produit == A.CP else "Coupon offert"]
    at, texte = objection_page(produit, "marge")
    coupon = f"{nombre(resultat):,.{0 if produit == A.CP else 2}f} %"
    controle(f"{produit[:18]:18} | « Combien vous gagnez » : "
             f"marge, rétro et {'participation' if produit == A.CP else 'coupon'} = Où va l'argent",
             not at.exception and f"marge {m['Marge banque']}" in texte
             and f"rétro {m['Rétrocession']}" in texte and coupon in texte,
             erreur(at) or f"{m['Marge banque']}, {m['Rétrocession']}, {resultat}")

    m = metriques(ouvre(produit, A.REVENTE, BS, a_zero))
    at, texte = objection_page(produit, "sortie")
    controle(f"{produit[:18]:18} | « Sortir avant l'échéance » : valeur et rachat = Revendre "
             "(mois 0, niveau 100)",
             not at.exception and f"valeur {m['Valeur juste']}" in texte
             and f"rachat {m['Prix de rachat']}" in texte,
             erreur(at) or f"{m['Valeur juste']}, {m['Prix de rachat']}")
    if produit != A.CP:
        at, texte = objection_page(produit, "piege")
        lu = m["Proba de perte à l'échéance"]
        proba = f"{nombre(lu):,.0f} %"
        controle(f"{produit[:18]:18} | « Où est le piège » : probabilité de perte = Revendre "
                 "(mois 0)", not at.exception and f"dans {proba} des cas" in texte,
                 erreur(at) or f"{lu} -> {proba}")

at = ouvre(A.AC, A.PITCH, BS, [lambda a: a.segmented_control(key="as_profil").set_value(A.CGP)])
duree = re.search(r"durée de vie moyenne ([0-9.]+) an", at.code[0].value).group(1)
_, texte = objection_page(A.AC, "rappel")
controle("Autocall | « Pourquoi il s'arrête » : durée de vie = celle du pitch",
         f"en moyenne {duree} an" in texte and f"moyenne {duree} an" in texte, duree)

spread = [lambda a: a.sidebar.slider(key="spread").set_value(150)]
at = ouvre(A.BRC, A.ARGENT, BS, spread)
credit = next(r for _, r in at.table[0].value.iterrows()
              if r["Composante"].startswith("Risque émetteur"))["EUR"].lstrip("-")
at, texte = objection_page(A.BRC, "faillite", A.CGP, spread)
controle("BRC spread 150 bp | « Et si la banque fait faillite » : apport du risque émetteur = "
         "Où va l'argent", not at.exception and f"spread 150 bp, soit {credit}" in texte
         and f"il apporte {credit}" in texte, erreur(at) or credit)

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
