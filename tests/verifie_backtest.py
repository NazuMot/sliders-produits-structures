"""
Verification de la page Backtest historique et du calibrage du Laboratoire.

    python tests/verifie_backtest.py

1. Backtest : les chiffres affiches sont ceux de backtest.py (et les chiffres
   de reference de tests/valide_donnees.py), pour chaque sous-jacent, onglet
   Autocall et onglet BRC, sous les deux modeles.
2. Le coupon de la vue de marche est celui des pages Autocall et BRC, et la
   case qui l'applique au backtest fonctionne.
3. Lien partageable : l'onglet BRC se rouvre depuis le lien ; curseur de la
   premiere emission ; cas extremes (6 ans depuis 2020, une seule observation).
4. Laboratoire : chaque echeance Eurex s'affiche, l'erreur de calibrage est
   celle du fichier, et le bouton passe la barre laterale sous Bates avec
   les valeurs calibrees (depuis Black-Scholes comme depuis un Bates regle
   a la main), et le lien partageable garde ces valeurs.

Sort avec le code 1 au premier probleme, ce qui fait echouer le test GitHub.
"""

import os
import sys
import time

from streamlit.testing.v1 import AppTest

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(RACINE, "app.py")
sys.path.insert(0, RACINE)

import backtest as B      # noqa: E402
import calibration as C   # noqa: E402

MODELES = ["Black-Scholes + skew", "Bates (vol stochastique + sauts)"]
SOUS_JACENTS = {"prix": "Indice des prix (SX5E)",
                "net_return": "Net return (dividendes réinvestis)",
                "decrement": "Indice à décrément (50 points)"}
BRC = "Barrier Reverse Convertible"
echecs = []


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def ouvre(page, modele=MODELES[0], reglages=None, query=None):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in (query or {}).items():
        at.query_params[k] = v
    at.run()
    at.sidebar.selectbox[0].set_value(page)
    at.sidebar.radio[0].set_value(modele)
    at.run()
    for regle in reglages or []:
        regle(at)
        at.run()
    return at


def metriques(at):
    return {m.label: m for m in at.metric}


def nombre(texte):
    return float(texte.split("%")[0].replace(",", "").strip())


def erreur(at):
    return at.exception[0].value[:300] if at.exception else ""


def sous_jacent(sj):
    return lambda a: a.radio(key="bt_sous_jacent").set_value(SOUS_JACENTS[sj])


def onglet_brc(a):
    a.session_state["bt_onglet"] = BRC


def dans_brc(action):
    """AppTest ne sait pas cliquer un onglet : on rouvre l'onglet BRC a chaque action."""
    return lambda a: (onglet_brc(a), action(a))


def proche(affiche, vrai, decimales=1):
    """Le chiffre affiche (en %) est la valeur exacte arrondie."""
    return abs(nombre(affiche) - round(vrai * 100, decimales)) < 1e-9


h = B.charge_historique()

# 1. Chiffres affiches = backtest.py ------------------------------------------
for modele in MODELES:
    for sj in SOUS_JACENTS:
        t = time.time()
        at = ouvre("Backtest historique", modele, [sous_jacent(sj)])
        nom = f"{modele.split()[0]:6} | {sj:10} | Autocall"
        if at.exception:
            controle(nom, False, erreur(at))
            continue
        m = metriques(at)
        r = B.backtest_autocall(h, sous_jacent=sj).resume
        ok = (proche(m["Rappelés à la 1re date"].value, r["rappel_premiere_date"])
              and proche(m["Pertes en capital"].value, r["perte"])
              and proche(m["Pire cas"].value, r["pire_capital"])
              and m["Pire cas"].delta == "émis le " + "/".join(r["pire_emission"].split("-")[::-1])
              and proche(m["Rendement par année investie"].value, r["tri_par_annee"])
              and f"({r['direct_par_annee'] * 100:.1f} %)" in m["Rendement par année investie"].delta)
        controle(nom + " : chiffres = backtest.py", ok,
                 f"rappel {m['Rappelés à la 1re date'].value}, perte {m['Pertes en capital'].value}"
                 f", pire {m['Pire cas'].value} {m['Pire cas'].delta}, {time.time() - t:.1f}s")

        onglet_brc(at)
        at.run()
        nom = f"{modele.split()[0]:6} | {sj:10} | BRC"
        if at.exception or "Barrière touchée" not in metriques(at):
            controle(nom, False, erreur(at) or "onglet BRC non affiché")
            continue
        m = metriques(at)
        r = B.backtest_brc(h, sous_jacent=sj).resume
        ok = (proche(m["Barrière touchée"].value, r["barriere_touchee"])
              and proche(m["Pertes en capital"].value, r["perte"])
              and proche(m["Pire cas"].value, r["pire_capital"])
              and m["Pire cas"].delta == "émis le " + "/".join(r["pire_emission"].split("-")[::-1])
              and proche(m["Rendement par année investie"].value, r["tri_par_annee"]))
        controle(nom + " : chiffres = backtest.py", ok,
                 f"touchée {m['Barrière touchée'].value}, perte {m['Pertes en capital'].value}, "
                 f"pire {m['Pire cas'].value} {m['Pire cas'].delta}")

# chiffres de reference de tests/valide_donnees.py, tels qu'affiches
at = ouvre("Backtest historique")
m = metriques(at)
controle("Autocall 3 ans depuis 2000 : 57,7 % rappelés à la 1re date, 3,5 % de pertes, "
         "pire cas 02/05/2000 (42,7 % rendu)",
         m["Rappelés à la 1re date"].value == "57.7 %" and m["Pertes en capital"].value == "3.5 %"
         and m["Pire cas"].value == "42.7 % rendu" and m["Pire cas"].delta == "émis le 02/05/2000"
         and any("286 émissions" in c.value for c in at.caption))
onglet_brc(at)
at.run()
m = metriques(at)
controle("BRC 1 an barrière 70 % depuis 2000 : 15,8 % de pertes, pire cas 03/03/2008",
         m["Pertes en capital"].value == "15.8 %" and m["Pire cas"].delta == "émis le 03/03/2008")

ac = B.backtest_autocall(h, sous_jacent="decrement").resume["perte"]
controle("indice à décrément : plus de pertes affichees que sur l'indice des prix",
         nombre(metriques(ouvre("Backtest historique", reglages=[sous_jacent("decrement")]))
                ["Pertes en capital"].value) > 3.5, f"{ac:.1%}")

# 2. Coupon de la vue de marche ------------------------------------------------
for modele in MODELES:
    at = ouvre("Backtest historique", modele)
    bt = nombre(metriques(at)["Coupon de la vue de marché"].value)
    ref = nombre(metriques(ouvre("4 - Autocall Phoenix", modele))["Équivalent annuel"].value)
    controle(f"{modele.split()[0]:6} | coupon de la vue de marché = page Autocall", bt == ref,
             f"{bt} vs {ref} %")
    at.checkbox(key="bt_ac_coupon_marche").check().run()
    r = B.backtest_autocall(h, coupon_pa=bt / 100).resume
    controle(f"{modele.split()[0]:6} | case coupon de marché : le backtest prend ce coupon",
             not at.exception and at.slider(key="bt_ac_coupon").disabled
             and abs(nombre(metriques(at)["Rendement par année investie"].value)
                     - r["tri_par_annee"] * 100) < 0.06, erreur(at))

    onglet_brc(at)
    at.run()
    bt = nombre(metriques(at)["Coupon de la vue de marché"].value)
    ref = nombre(metriques(ouvre("3 - Barrier Reverse Convertible", modele))
                 ["Coupon équitable"].value)
    controle(f"{modele.split()[0]:6} | coupon de la vue de marché = page BRC", bt == ref,
             f"{bt} vs {ref} %")
    dans_brc(lambda a: a.checkbox(key="bt_brc_coupon_marche").check())(at)
    at.run()
    controle(f"{modele.split()[0]:6} | case coupon de marché BRC : coupon backteste = coupon du desk",
             not at.exception and nombre(metriques(at)["Coupon backtesté"].value) == bt, erreur(at))

# 3. Lien, curseurs, cas limites ------------------------------------------------
at = ouvre("Backtest historique", query={"bt_onglet": "brc"})
controle("lien ?bt_onglet=brc : l'onglet BRC s'ouvre directement",
         not at.exception and "Barrière touchée" in metriques(at), erreur(at))

at = ouvre("Backtest historique", reglages=[lambda a: a.slider(key="bt_debut").set_value(2009)])
r = B.backtest_autocall(h, debut="2009-01-01").resume
controle("première émission en 2009 : moins d'émissions, chiffres = backtest.py",
         not at.exception and proche(metriques(at)["Pertes en capital"].value, r["perte"])
         and any(f"{r['n']} émissions" in c.value for c in at.caption),
         f"{r['n']} émissions, perte {r['perte']:.1%}")

at = ouvre("Backtest historique", reglages=[
    lambda a: a.slider(key="bt_debut").set_value(2020),
    lambda a: a.slider(key="bt_ac_maturite").set_value(6.0)])
r = B.backtest_autocall(h, T=6.0, debut="2020-01-01").resume
controle("cas extrême (2020, 6 ans) : peu d'émissions terminees, chiffres = backtest.py",
         not at.exception and any(f"{r['n']} émissions" in c.value for c in at.caption),
         erreur(at) or f"{r['n']} émissions")

at = ouvre("Backtest historique", reglages=[
    lambda a: a.selectbox(key="bt_ac_frequence").set_value("Annuelle"),
    lambda a: a.slider(key="bt_ac_maturite").set_value(1.0)])
controle("autocall 1 an annuel (une seule observation) : la page s'affiche",
         not at.exception, erreur(at))

at = ouvre("Backtest historique", reglages=[onglet_brc, dans_brc(
    lambda a: a.radio(key="bt_brc_observation").set_value("À l'échéance (européenne)"))])
r = B.backtest_brc(h, continu=False).resume
controle("BRC barrière à l'échéance : chiffres = backtest.py",
         not at.exception and proche(metriques(at)["Pertes en capital"].value, r["perte"]),
         erreur(at) or f"perte {r['perte']:.1%}")

at = ouvre("Backtest historique")
t = time.time()
at.slider(key="bt_ac_coupon").set_value(9.0).run()
duree = time.time() - t
controle("curseur déplacé : la page se recalcule vite (backtest et Monte Carlo en cache)",
         not at.exception and duree < 3.0, f"{duree:.2f}s")

# 4. Laboratoire : calibrage ----------------------------------------------------
cal = C.charge_parametres()
cu = cal["curseurs"]
BARRE = {"vol": cu["vol_atm"], "vol_lt": cu["theta_v"], "kappa": cu["kappa"],
         "vol_vol": cu["xi"], "rho": cu["rho"], "sauts": cu["lam"],
         "taille_saut": cu["mu_j"], "disp_saut": cu["sig_j"]}

at = ouvre("Laboratoire modèle")
m = metriques(at)
controle("Laboratoire : erreur de calibrage affichée = fichier (0,77 pt, curseurs 0,91 pt)",
         not at.exception
         and m["Erreur, toute la surface"].value == f"{cal['erreur_pts_vol']['moyenne_quadratique']:.2f} pt"
         and m["Aux crans des curseurs"].value
         == f"{cal['erreur_pts_vol']['moyenne_quadratique_curseurs']:.2f} pt", erreur(at))
rates = []
for e in cal["par_echeance"]:
    at.selectbox(key="labo_echeance").set_value(e["echeance"]).run()
    if at.exception or metriques(at)["Erreur, cette échéance"].value != f"{e['moyenne_quadratique']:.2f} pt":
        rates.append(e["echeance"] + " " + erreur(at))
controle(f"Laboratoire : les {len(cal['par_echeance'])} échéances Eurex s'affichent",
         not rates, "; ".join(rates))

for modele, avant in ((MODELES[0], []), (MODELES[1], [lambda a: a.slider(key="kappa").set_value(4.0),
                                                       lambda a: a.slider(key="rho").set_value(-0.2)])):
    at = ouvre("Laboratoire modèle", modele, avant)
    at.button(key="labo_appliquer").click().run()
    valeurs = {k: at.slider(key=k).value for k in BARRE} if not at.exception else {}
    controle(f"{modele.split()[0]:6} | bouton Appliquer : barre latérale sous Bates, valeurs calibrees",
             not at.exception and at.sidebar.radio[0].value == MODELES[1]
             and all(abs(valeurs[k] - v) < 1e-9 for k, v in BARRE.items())
             and len(at.success) == 1, erreur(at) or str(valeurs))
    lien = {k: at.query_params.get(k, [""])[-1] for k in ["modele", *BARRE]}
    controle(f"{modele.split()[0]:6} | bouton Appliquer : le lien partageable contient le calibrage",
             lien["modele"] == MODELES[1]
             and all(lien[k] and abs(float(lien[k]) - v) < 1e-9 for k, v in BARRE.items()),
             str(lien))
    at.run()
    controle(f"{modele.split()[0]:6} | après un nouveau rendu, les valeurs restent et le message part",
             not at.exception and all(abs(at.slider(key=k).value - v) < 1e-9 for k, v in BARRE.items())
             and not at.success, erreur(at))

at.sidebar.selectbox[0].set_value("3 - Barrier Reverse Convertible").run()
c = nombre(metriques(at)["Coupon équitable"].value) if not at.exception else 0
controle("après calibrage, la page BRC price sous Bates calibré", not at.exception and c > 0,
         erreur(at) or f"coupon {c:.2f} %")

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
