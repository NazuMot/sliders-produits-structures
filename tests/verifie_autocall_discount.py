"""
Verification des options de la page Autocall et de la page Discount Certificate.

    python tests/verifie_autocall_discount.py

Chaque chiffre affiche est recalcule directement avec moteur.py (memes
tirages) : la page doit afficher exactement la meme valeur.

1. Autocall, reglages par defaut : identique au calcul du moteur (rien de casse)
2. Step-down          5. Grecques du desk
3. Indice a decrement 6. Garde-fou memoire, temps de la page
4. Correlation de stress (worst-of)
7. Discount Certificate (formule fermee sous Black-Scholes, Monte Carlo sous Bates)

Sort avec le code 1 au premier probleme.
"""

import math
import os
import sys
import time
import tracemalloc

import numpy as np
from streamlit.testing.v1 import AppTest

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(RACINE, "app.py")
sys.path.insert(0, RACINE)

import moteur as M                      # noqa: E402
from moteur import S0                   # noqa: E402

BS, BT = "Black-Scholes + skew", "Bates (vol stochastique + sauts)"
AUTOCALL, DISCOUNT = "4 - Autocall Phoenix", "7 - Discount Certificate"

# Reglages par defaut de la barre laterale
r, q, vol, skew, marge, nominal, N = 0.03, 0.025, 0.22, 2.0, 0.008, 1000.0, 10000
BATES = dict(v0=0.22 ** 2, theta=0.24 ** 2, kappa=2.0, xi=0.5, rho=-0.7,
             lam=0.3, mu_j=-0.10, sig_j=0.15)

echecs = []


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def ouvre(page, modele, reglages=()):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    next(r for r in at.sidebar.radio if page in r.options).set_value(page)
    at.sidebar.radio(key="modele").set_value(modele)
    at.run()
    for regle in reglages:
        regle(at)
        at.run()
    return at


def metriques(at):
    return {m.label: m.value for m in at.metric}


def nombre(texte):
    return float(texte.replace("%", "").replace("p.a.", "").replace("ans", "")
                 .replace(",", "").strip())


def sans_erreur(at):
    return not at.exception, at.exception[0].value[:300] if at.exception else ""


def tableau(at, colonne):
    """Le tableau de la page qui a cette colonne (DataFrame), ou None."""
    for t in at.table:
        if colonne in t.value.columns:
            return t.value
    return None


def textes(liste):
    return " ".join(x.value for x in liste)


def regle(cle, valeur, sorte="slider"):
    return lambda at: getattr(at, sorte)(key=cle).set_value(valeur)


def coche(cle):
    return lambda at: at.checkbox(key=cle).check()


def chemins(T, modele, niveau, n=N, q_=q, decrement=0.0):
    if modele == BS:            # page : volatilite locale (barre.trajectoires, sim_decrement)
        return M.simule_lv(S0, T, r, q_, vol, skew, n, decrement=decrement)[0]
    return M.simule_bates(S0, T, r, q_, n, decrement=decrement, **BATES)[0]


def autocall(ch, T=3.0, n_obs=12, trig=1.0, bcp=0.70, bcap=0.60):
    c = M.coupon_autocall(ch, T, r, nominal, n_obs, trig, bcp, bcap, True, marge)
    duree, p_rappel, p_perte = M.stats_autocall(ch, T, n_obs, trig, bcap)
    return c, duree, p_perte


def attendu(c, duree, p_perte, par_an=4):
    return {"Coupon par observation": f"{c*100:,.2f}%",
            "Équivalent annuel": f"{c*par_an*100:,.2f}% p.a.",
            "Durée de vie moyenne": f"{duree:,.2f} ans",
            "Proba de perte en capital": f"{p_perte*100:,.1f}%"}


def meme_affichage(nom, at, att):
    ok, err = sans_erreur(at)
    vu = metriques(at)
    ecarts = [f"{k}: page {vu.get(k)} / moteur {v}" for k, v in att.items() if vu.get(k) != v]
    controle(nom, ok and not ecarts, err or "; ".join(ecarts) or vu.get("Équivalent annuel", ""))


def graphes(at):
    """Texte de chaque graphe : la spec Vega-Lite et ses donnees (tableaux Arrow)."""
    return [g.proto.spec + " ".join(d.data.data.decode("utf-8", "ignore") for d in g.proto.datasets)
            for g in at.get("vega_lite_chart")]


# =====================================================================
print("\n1. Autocall par défaut : rien de changé\n" + "-" * 40)
# =====================================================================
DEFAUT = {}
for modele in (BS, BT):
    res = autocall(chemins(3.0, modele, 0.60))
    DEFAUT[modele] = res
    at = ouvre(AUTOCALL, modele)
    meme_affichage(f"{modele[:6]} | autocall par défaut = moteur", at, attendu(*res))
    controle(f"{modele[:6]} | aucune table d'option sans option", not at.table,
             f"{len(at.table)} table(s)")


# =====================================================================
print("\n2. Step-down\n" + "-" * 12)
# =====================================================================
for modele in (BS, BT):
    sd = M.trigger_degressif(12, 1.0, 0.05, 0.80, obs_par_palier=4)
    res = autocall(chemins(3.0, modele, 0.60), trig=sd)
    at = ouvre(AUTOCALL, modele, [coche("ac_step_down")])
    meme_affichage(f"{modele[:6]} | step-down 100 %, -5 pts par an, plancher 80 = moteur",
                   at, attendu(*res))
    d = DEFAUT[modele]
    # marge sur la vie attendue : le coupon par periode bouge peu (moins de 0,5 pt p.a.)
    controle(f"{modele[:6]} | step-down : coupon proche, durée plus courte, moins de pertes",
             abs(res[0] - d[0]) * 400 < 0.5 and res[1] < d[1] and res[2] < d[2],
             f"coupon {d[0]*400:.2f} -> {res[0]*400:.2f} %, durée {d[1]:.2f} -> {res[1]:.2f} ans, "
             f"perte {d[2]*100:.1f} -> {res[2]*100:.1f} %")
    t = tableau(at, "Seuil de rappel")
    controle(f"{modele[:6]} | tableau fixe / step-down, première ligne = trigger fixe",
             t is not None and len(t) == 2
             and t.iloc[0]["Coupon annuel"] == f"{d[0]*400:,.2f}%")
    controle(f"{modele[:6]} | le faisceau trace le trigger dégressif 100 -> 90",
             any("trigger dégressif  100 -> 90" in g for g in graphes(at)))

at = ouvre(AUTOCALL, BS, [regle("ac_maturite", 6.0), regle("ac_frequence", "Annuelle", "selectbox"),
                          coche("ac_step_down"), regle("ac_sd_pas", 10), regle("ac_sd_plancher", 70)])
sd = M.trigger_degressif(6, 1.0, 0.10, 0.70)
controle("Step-down 6 ans annuel, -10 pts par an, plancher 70 : paliers 100, 90, 80, 70, 70",
         sd[:5] == [1.0, 0.9, 0.8, 0.7, 0.7])
meme_affichage("Step-down 6 ans annuel = moteur", at,
               attendu(*autocall(chemins(6.0, BS, 0.60), 6.0, 6, sd), par_an=1))
at = ouvre(AUTOCALL, BS, [regle("ac_maturite", 1.0), regle("ac_frequence", "Annuelle", "selectbox"),
                          coche("ac_step_down")])
meme_affichage("Step-down sur une seule observation : rien ne casse, = autocall fixe", at,
               attendu(*autocall(chemins(1.0, BS, 0.60), 1.0, 1), par_an=1))
at = ouvre(AUTOCALL, BS, [regle("ac_trigger", 90), coche("ac_step_down"), regle("ac_sd_plancher", 100)])
meme_affichage("Plancher au-dessus du trigger : ramene au trigger (= trigger fixe 90 %)", at,
               attendu(*autocall(chemins(3.0, BS, 0.60), trig=0.9)))


# =====================================================================
print("\n3. Indice à décrément\n" + "-" * 21)
# =====================================================================
DEC = regle("ac_indice", "Indice à décrément", "radio")
for modele in (BS, BT):
    res = autocall(chemins(3.0, modele, 0.60, q_=0.0, decrement=5.0))
    at = ouvre(AUTOCALL, modele, [DEC])
    meme_affichage(f"{modele[:6]} | décrément 50 points sur 1 000 (5 par an base 100) = moteur",
                   at, attendu(*res))
    d = DEFAUT[modele]
    controle(f"{modele[:6]} | décrément 5 % > dividende 2,5 % : coupon ET perte plus élevés",
             res[0] > d[0] and res[2] > d[2],
             f"coupon {d[0]*400:.2f} -> {res[0]*400:.2f} %, perte {d[2]*100:.1f} -> {res[2]*100:.1f} %")
    t = tableau(at, "Niveau moyen à l'échéance")
    pct = autocall(chemins(3.0, modele, 0.60, q_=0.05))
    controle(f"{modele[:6]} | tableau : classique = page par défaut, décrément en % = moteur q = 5 %, "
             "décrément en points = metrique affichée",
             t is not None and len(t) == 3
             and t.iloc[0]["Coupon annuel"] == f"{d[0]*400:,.2f}%"
             and t.iloc[1]["Coupon annuel"] == f"{pct[0]*400:,.2f}%"
             and t.iloc[2]["Coupon annuel"] == metriques(at)["Équivalent annuel"].replace(" p.a.", ""))
    controle(f"{modele[:6]} | effet points : perte plus forte qu'avec le décrément en %",
             res[2] > pct[2], f"{pct[2]*100:.1f} -> {res[2]*100:.1f} %")
    controle(f"{modele[:6]} | la phrase de Sales explique coupon et risque en hausse",
             "Le coupon est plus élevé" in textes(at.info))

at = ouvre(AUTOCALL, BS, [DEC, regle("ac_dec_niveau", 5000, "select_slider")])
res = autocall(chemins(3.0, BS, 0.60, q_=0.0, decrement=1.0))
meme_affichage("Décrément 50 points sur 5 000 (1 % par an) = moteur", at, attendu(*res))
controle("Décrément 1 % < dividende 2,5 % : coupon plus bas, la page le dit",
         res[0] < DEFAUT[BS][0] and "Ici le décrément retire moins" in textes(at.info))
at = ouvre(AUTOCALL, BS, [DEC, regle("ac_sous_jacents", 3, "select_slider")])
meme_affichage("Décrément choisi puis worst-of : le panier reste sur indices classiques", at,
               attendu(*autocall(M.simule_panier(3, 0.6, S0, 3.0, r, q,
                                                 M.vol_au_strike(0.6, vol, skew), N)[0])))
controle("... et la page le signale", "mono sous-jacent seulement" in textes(at.caption))


# =====================================================================
print("\n4. Corrélation de stress (worst-of)\n" + "-" * 35)
# =====================================================================
WO3 = regle("ac_sous_jacents", 3, "select_slider")
for modele in (BS, BT):
    v = vol if modele == BT else M.vol_au_strike(0.6, vol, skew)
    const = autocall(M.simule_panier(3, 0.6, S0, 3.0, r, q, v, N)[0])
    res = autocall(M.simule_panier(3, 0.6, S0, 3.0, r, q, v, N, correlation_stress=0.9,
                                   seuil_stress=0.10)[0])
    at = ouvre(AUTOCALL, modele, [WO3, coche("ac_stress")])
    meme_affichage(f"{modele[:6]} | worst-of 3, 60 %, 90 % en krach sous -10 % = moteur", at,
                   attendu(*res))
    controle(f"{modele[:6]} | la corrélation de stress fait BAISSER coupon et perte",
             res[0] < const[0] and res[2] < const[2],
             f"coupon {const[0]*400:.2f} -> {res[0]*400:.2f} %, perte {const[2]*100:.1f} -> {res[2]*100:.1f} %")
    t = tableau(at, "Corrélation")
    # le regime de stress ramene le coupon vers celui de la correlation haute ; avec la
    # marge sur la vie attendue, les deux peuvent se croiser de quelques centiemes
    ok = t is not None and len(t) == 3 and t.iloc[0]["Coupon annuel"] == f"{const[0]*400:,.2f}%"
    if ok:
        c0, c1, c2 = (nombre(t.iloc[i]["Coupon annuel"]) for i in range(3))
        ok = c0 > c1 and c0 > c2 and abs(c1 - c2) < 0.5 * (c0 - c2) + 0.3
    controle(f"{modele[:6]} | tableau : constante = moteur sans stress, le stress fait baisser le coupon",
             ok, "" if t is None else " / ".join(t["Coupon annuel"]))
    controle(f"{modele[:6]} | message contre-intuitif et surpaiement du desk",
             "contre-intuitif" in textes(at.info) and "surpayer le client" in textes(at.info))

at = ouvre(AUTOCALL, BS, [WO3, coche("ac_stress"), regle("ac_correl_stress", 0.3),
                          regle("ac_seuil_stress", 30)])
v = M.vol_au_strike(0.6, vol, skew)
meme_affichage("Krach à 30 %, corrélation de krach 30 % (sous la normale) = moteur", at,
               attendu(*autocall(M.simule_panier(3, 0.6, S0, 3.0, r, q, v, N, correlation_stress=0.3,
                                                 seuil_stress=0.30)[0])))
controle("... et la page explique le sens inverse", "plus basse que la corrélation normale"
         in textes(at.info))
at = ouvre(AUTOCALL, BS)
controle("Mono sous-jacent : pas de case corrélation de stress (aucune case grisee)",
         all(c.key != "ac_stress" for c in at.checkbox))
at = ouvre(AUTOCALL, BS, [WO3, coche("ac_stress"), regle("ac_sous_jacents", 1, "select_slider")])
meme_affichage("Stress coche puis retour au mono sous-jacent : sans effet", at, attendu(*DEFAUT[BS]))


# =====================================================================
print("\n5. Grecques du desk\n" + "-" * 19)
# =====================================================================
GRECQUES = coche("ac_voir_grecques")


def compare_grecques(nom, at, g, avec_correl):
    t = tableau(at, "Ce que le desk fait")
    ok, err = sans_erreur(at)
    if t is None:
        controle(nom, False, err or "pas de tableau")
        return
    lignes = dict(zip(t["Grecque"], t["Desk"]))
    att = {"Delta": f"{g['delta']:+,.2f}", "Gamma": f"{g['gamma']:+,.3f}", "Vega": f"{g['vega']:+,.2f}",
           "Dividende": f"{g['dividende'] + 0.0:+,.2f}", "Rho": f"{g['rho']:+,.2f}",
           "Theta": f"{g['theta']:+,.2f}"}
    if avec_correl:
        att["Corrélation"] = f"{g['correlation']:+,.2f}"
    ecarts = [f"{k}: {lignes.get(k)} / {v}" for k, v in att.items()
              if not str(lignes.get(k, "")).startswith(v + " ")]
    controle(nom, ok and not ecarts and ("Corrélation" in lignes) == avec_correl,
             "; ".join(ecarts) or ", ".join(f"{k} {v}" for k, v in att.items()))


V60 = M.vol_au_strike(0.6, vol, skew)
cas = [
    ("BS mono", BS, [], dict(sigma=V60), DEFAUT[BS][0]),
    ("Bates mono", BT, [], dict(sigma=V60, bates=BATES), DEFAUT[BT][0]),
    ("BS worst-of 3, stress", BS, [WO3, coche("ac_stress")],
     dict(sigma=V60, n_actifs=3, correlation=0.6, correlation_stress=0.9, seuil_stress=0.10),
     autocall(M.simule_panier(3, 0.6, S0, 3.0, r, q, V60, N, correlation_stress=0.9)[0])[0]),
    ("BS décrément 50 pts", BS, [DEC], dict(sigma=V60, q=0.0, decrement=5.0),
     autocall(chemins(3.0, BS, 0.60, q_=0.0, decrement=5.0))[0]),
    ("BS step-down", BS, [coche("ac_step_down")],
     dict(sigma=V60, trigger=tuple(M.trigger_degressif(12, 1.0, 0.05, 0.80, obs_par_palier=4))),
     None),
]
for nom, modele, regles, kw, cpn in cas:
    kw = dict(kw)
    sigma, trig, qq = kw.pop("sigma"), kw.pop("trigger", 1.0), kw.pop("q", q)
    if cpn is None:
        cpn = autocall(chemins(3.0, BS, 0.60), trig=list(trig))[0]
    t0 = time.time()
    at = ouvre(AUTOCALL, modele, regles + [GRECQUES])
    el = time.time() - t0
    g = M.grecques_autocall(3.0, r, qq, sigma, 12, trig, 0.7, 0.6, True, float(cpn), nominal,
                            n_paths=10000, **kw)
    compare_grecques(f"{nom} : grecques affichees = moteur ({el:.1f} s)", at, g,
                     kw.get("n_actifs", 1) > 1)
    signes = g["delta"] < 0 and g["vega"] > 0 and g["theta"] < 0
    signes &= (g["dividende"] == 0) if "decrement" in kw else g["dividende"] > 0
    if kw.get("n_actifs", 1) > 1:
        signes &= g["correlation"] < 0
    controle(f"{nom} : desk long vega, long dividendes (nul sur décrément), court corrélation",
             signes)
at = ouvre(AUTOCALL, BS, [DEC, GRECQUES])
controle("Décrément : la ligne dividende dit pourquoi le risque disparaît",
         "Aucun risque" in " ".join(tableau(at, "Ce que le desk fait")["Ce que le desk fait"]))


# =====================================================================
print("\n6. Garde-fou mémoire et temps\n" + "-" * 29)
# =====================================================================
from vues.autocall import trajectoires_panier    # noqa: E402

controle("Plafond des paniers : défaut inchangé (3 actifs, 3 ans, 10 000)",
         trajectoires_panier(10000, 3, 3.0) == 10000)
controle("Plafond des paniers : 4 actifs, 6 ans, 40 000 demandes -> 10 000",
         trajectoires_panier(40000, 4, 6.0) == 10000, str(trajectoires_panier(40000, 4, 6.0)))
controle("Plafond des paniers : jamais plus de 20 000, mono sous-jacent jamais plafonné",
         trajectoires_panier(40000, 2, 1.0) == 20000 and trajectoires_panier(40000, 1, 6.0) == 40000)

LOURD = [regle("trajectoires", 40000, "select_slider"), regle("ac_maturite", 6.0),
         regle("ac_sous_jacents", 4, "select_slider")]
tracemalloc.start()
t0 = time.time()
at = ouvre(AUTOCALL, BS, LOURD + [coche("ac_voir_worst_of")])
el = time.time() - t0
pic = tracemalloc.get_traced_memory()[1] / 1e6
tracemalloc.stop()
meme_affichage(f"Worst-of 4, 6 ans, 40 000 demandees : calculé sur 10 000 ({el:.1f} s, pointe {pic:.0f} Mo)",
               at, attendu(*autocall(M.simule_panier(4, 0.6, S0, 6.0, r, q, V60, 10000)[0], 6.0, 24)))
controle("... legende discrete affichée", "au lieu de 40 000" in textes(at.caption))
controle("... pointe de mémoire du calcul sous 450 Mo (site gratuit : ~1 Go ; ~376 Mo sur Mac, ~400 sous Linux)", pic < 450, f"{pic:.0f} Mo")

# toutes les options a la fois : premier calcul puis reaction a un curseur deja vu
TOUT = [regle("ac_sous_jacents", 3, "select_slider"), coche("ac_step_down"), coche("ac_stress"),
        GRECQUES, coche("ac_voir_worst_of")]
for modele in (BS, BT):
    t0 = time.time()
    at = ouvre(AUTOCALL, modele, TOUT)
    froid = time.time() - t0
    ok, err = sans_erreur(at)
    t0 = time.time()
    at.slider(key="ac_bar_coupon").set_value(65).run()
    at.slider(key="ac_bar_coupon").set_value(70).run()
    chaud = (time.time() - t0) / 2
    controle(f"{modele[:6]} | toutes les options : premier affichage {froid:.1f} s (6 executions), "
             f"puis {chaud:.2f} s par mouvement de curseur", ok and chaud < 1.5, err)
    t0 = time.time()
    at = ouvre(AUTOCALL, modele, [DEC, coche("ac_step_down"), GRECQUES])
    controle(f"{modele[:6]} | décrément + step-down + grecques ({time.time() - t0:.1f} s)",
             *sans_erreur(at))


# =====================================================================
print("\n7. Discount Certificate\n" + "-" * 23)
# =====================================================================
def discount_attendu(o, ST):
    return {"Décote": f"{o['decote']*100:,.2f}%",
            "Rendement maximal": f"{o['rendement_max']*100:,.2f}%",
            "Point mort": f"{o['prix']:,.2f}",
            "Proba de perte en capital": f"{(ST < o['prix']).mean()*100:,.1f}%"}


for T, cap in ((1.0, 100), (2.0, 90), (0.5, 110), (3.0, 70), (0.25, 130)):
    regles = [regle("dc_maturite", T), regle("dc_cap", cap)]
    for modele in (BS, BT):
        ch = chemins(T, modele, cap / 100)
        if modele == BS:
            o = M.discount_certificate(S0, cap, T, r, M.vol_au_strike(cap / 100, vol, skew), q, marge)
        else:
            o = M.discount_mc(ch, T, r, cap / 100, marge)
        at = ouvre(DISCOUNT, modele, regles + [coche("dc_voir_decomposition")])
        ok, err = sans_erreur(at)
        vu, att = metriques(at), discount_attendu(o, ch[:, -1].astype(np.float64))
        ecarts = [f"{k}: {vu.get(k)} / {v}" for k, v in att.items() if vu.get(k) != v]
        t = tableau(at, "Points de % du spot")
        somme = None if t is None else sum(nombre(x) for x in t["Points de % du spot"][:3])
        ok_decomp = t is not None and abs(somme - (S0 - o["prix"])) < 0.02 \
            and abs(nombre(t["Points de % du spot"].iloc[3]) - (S0 - o["prix"])) < 0.006
        controle(f"{modele[:6]} | Discount T = {T:g}, cap {cap} : metriques = moteur, décomposition "
                 "= décote", ok and not ecarts and ok_decomp,
                 err or "; ".join(ecarts) or f"décote {vu['Décote']}, max {vu['Rendement maximal']}, "
                                             f"point mort {vu['Point mort']}")

# la decote a deux sources : sous BS, dividendes + call = S0 - valeur exactement
o = M.discount_certificate(S0, 100.0, 1.0, r, vol, q)
call = M.bs_call(S0, 100.0, 1.0, r, vol, q)
controle("Décomposition : prime du call affichée = call Black-Scholes",
         abs((S0 * math.exp(-q) - o["valeur"]) - call) < 1e-9)
at = ouvre(DISCOUNT, BS, [regle("vol", 35.0)])
haute = nombre(metriques(at)["Décote"])
at = ouvre(DISCOUNT, BS, [regle("vol", 15.0)])
controle("Vol plus haute -> décote plus forte", haute > nombre(metriques(at)["Décote"]),
         f"{metriques(at)['Décote']} à 15 % de vol, {haute:.2f}% à 35 %")
at = ouvre(DISCOUNT, BS)
controle("Les deux graphes du Discount s'affichent (profil + choix du cap)",
         len(graphes(at)) == 2 and "point mort" in graphes(at)[0]
         and "cap choisi" in graphes(at)[1])

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
