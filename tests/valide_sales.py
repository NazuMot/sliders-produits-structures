"""
Validation chiffree de sales.py (outils du Sales).

    python tests/valide_sales.py

1. Coherence avec les pages : memes coupons et participations, au bit pres
   contre le code des pages, et contre l'affichage reel des pages.
2. Aller-retour du reverse enquiry : je fixe une structure -> coupon ; je
   redonne ce coupon au solveur -> je retrouve la structure a 0,5 pt pres.
3. Ou va l'argent : la somme des lignes retombe sur le nominal, la marge
   n'est pas un residu deguise, la retro coute ce qui est annonce.
4. Marche secondaire : a t = 0, valeur = nominal - marge - retro (sur la vie
   attendue pour l'autocall) ; la
   decomposition retombe exactement sur la valeur du jour ; les sens de
   variation sont les bons.
5. Cas limites, puis temps de calcul.

Sort avec le code 1 au premier probleme.
"""

import math
import os
import sys
import time
from types import SimpleNamespace

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import moteur as M
import sales as S
from moteur import S0

echecs = []


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def proche(a, b, tol):
    return abs(a - b) <= tol


BS = S.Marche()
BATES = S.Marche(modele="bates")
MODELES = [("BS", BS), ("Bates", BATES)]
N = BS.nominal


# 1. Coherence avec les pages ---------------------------------------------------
print("1. Coherence avec les pages")


def page_brc(mk, T, bar, strike, continu):
    """Le calcul de vues/brc.py, recopie a l'identique."""
    if mk.bates:
        ch, sg = M.simule_bates(S0, T, mk.r, mk.q, mk.n_paths, mk.vol_atm ** 2, mk.theta_v ** 2,
                                mk.kappa, mk.xi, mk.rho, mk.lam, mk.mu_j, mk.sig_j, seed=42)
    else:
        ch, sg = M.simule_lv(S0, T, mk.r, mk.q, mk.vol_atm, mk.skew, mk.n_paths, seed=42)
    tau = M.tire_defaut(len(ch), T, mk.spread, mk.recovery) if mk.spread > 0 else None
    return M.coupon_brc(ch, T, mk.r, mk.nominal, bar, strike, continu, mk.marge, sg, tau,
                        mk.recovery)


def page_autocall(mk, T, par_an, trig, bcp, bcap, memoire, n_actifs, correl):
    """Le calcul de vues/autocall.py, recopie a l'identique."""
    n_obs = max(int(T * par_an), 1)
    if n_actifs > 1:
        v = mk.vol_atm if mk.bates else M.vol_au_strike(bcap, mk.vol_atm, mk.skew)
        ch, sg = M.simule_panier(n_actifs, correl, S0, T, mk.r, mk.q, v, mk.n_paths, seed=42)
    elif mk.bates:
        ch, sg = M.simule_bates(S0, T, mk.r, mk.q, mk.n_paths, mk.vol_atm ** 2, mk.theta_v ** 2,
                                mk.kappa, mk.xi, mk.rho, mk.lam, mk.mu_j, mk.sig_j, seed=42)
    else:
        ch, sg = M.simule_lv(S0, T, mk.r, mk.q, mk.vol_atm, mk.skew, mk.n_paths, seed=42)
    tau = M.tire_defaut(len(ch), T, mk.spread, mk.recovery) if mk.spread > 0 else None
    return M.coupon_autocall(ch, T, mk.r, mk.nominal, n_obs, trig, bcp, bcap, memoire,
                             mk.marge, tau, mk.recovery)


def page_cp(mk, T, prot, cap):
    """Le calcul de vues/capital_protection.py, recopie a l'identique."""
    r_fin = mk.r + mk.spread / 10000.0
    zc = prot * math.exp(-r_fin * T)
    budget = 1.0 - zc - mk.marge * T
    v_call = mk.vol_atm if mk.bates else M.vol_au_strike(1.0, mk.vol_atm, mk.skew)
    v_cap = mk.vol_atm if mk.bates else M.vol_au_strike(cap, mk.vol_atm, mk.skew)
    cout = max((M.bs_call(S0, S0, T, mk.r, v_call, mk.q)
                - M.bs_call(S0, cap * S0, T, mk.r, v_cap, mk.q)) / S0, 1e-9)
    return max(budget / cout, 0.0)


for nom, base in MODELES:
    for spread in (0, 150):
        mk = base.avec(spread=spread)
        etiq = f"{nom:5} spread {spread:3} bp"
        a, b = S.coupon_brc(mk, 1.0, 0.70, 1.0, True), page_brc(mk, 1.0, 0.70, 1.0, True)
        controle(f"{etiq} | BRC 1 an barriere 70 % identique a la page", a == b, f"{a*100:.4f}%")
        a = S.coupon_brc(mk, 2.0, 0.60, 0.95, False)
        b = page_brc(mk, 2.0, 0.60, 0.95, False)
        controle(f"{etiq} | BRC 2 ans europeen strike 95 % identique", a == b, f"{a*100:.4f}%")
        a = S.coupon_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60, True)
        b = page_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60, True, 1, 0.6)
        controle(f"{etiq} | Autocall 3 ans identique a la page", a == b, f"{a*400:.4f}% p.a.")
        a = S.coupon_autocall(mk, 4.0, 2, 0.95, 0.65, 0.55, False, 3, 0.4)
        b = page_autocall(mk, 4.0, 2, 0.95, 0.65, 0.55, False, 3, 0.4)
        controle(f"{etiq} | Autocall worst-of 3 sans memoire identique", a == b,
                 f"{a*200:.4f}% p.a.")
        a, b = S.participation_cp(mk, 5.0, 1.0, 3.0), page_cp(mk, 5.0, 1.0, 3.0)
        controle(f"{etiq} | Capital Protection 5 ans identique a la page", a == b,
                 f"{a*100:.2f}%")
        a, b = S.participation_cp(mk, 4.0, 0.9, 1.5), page_cp(mk, 4.0, 0.9, 1.5)
        controle(f"{etiq} | Capital Protection 90 % cap 150 % identique", a == b,
                 f"{a*100:.2f}%")

# Marche.depuis_barre : memes noms que l'objet renvoye par barre_laterale()
reglages_bs = dict(BATES=False, r=0.03, q=0.025, vol_atm=0.22, skew=2.0, theta_v=None,
                   kappa=None, xi=None, rho=None, lam=None, mu_j=None, sig_j=None, spread=0,
                   recovery=0.4, marge=0.008, nominal=1000.0, n_paths=10000)
controle("depuis_barre reproduit la vue Black-Scholes par defaut",
         S.Marche.depuis_barre(SimpleNamespace(**reglages_bs)) == BS)
reglages_bt = dict(reglages_bs, BATES=True, skew=0.0, theta_v=0.30, kappa=1.0, xi=0.4,
                   rho=-0.5, lam=0.0, mu_j=0.0, sig_j=0.10)
mb = S.Marche.depuis_barre(SimpleNamespace(**reglages_bt), retro=0.01)
controle("depuis_barre garde un parametre Bates nul (sauts coupes)",
         mb.bates and mb.lam == 0.0 and mb.mu_j == 0.0 and mb.retro == 0.01)

# Contre l'affichage reel des pages (AppTest, parametres par defaut)
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None
if AppTest is None:
    controle("Affichage des pages (streamlit absent)", False)
else:
    APP = os.path.join(RACINE, "app.py")
    for nom, mk in MODELES:
        modele = "Bates (vol stochastique + sauts)" if mk.bates else "Black-Scholes + skew"
        lu = {}
        for page in ("3 - Barrier Reverse Convertible", "4 - Autocall Phoenix",
                     "2 - Capital Protection"):
            at = AppTest.from_file(APP, default_timeout=300)
            at.run()
            at.sidebar.selectbox[0].set_value(page)
            at.sidebar.radio[0].set_value(modele)
            at.run()
            lu.update({m.label: m.value for m in at.metric})
        attendu = f"{S.coupon_brc(mk, 1.0, 0.70)*100:,.2f}% p.a."
        controle(f"{nom:5} | page BRC affiche le coupon de sales.py",
                 lu.get("Coupon equitable") == attendu, f"{lu.get('Coupon equitable')}")
        attendu = f"{S.coupon_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60)*100:,.2f}%"
        controle(f"{nom:5} | page Autocall affiche le coupon de sales.py",
                 lu.get("Coupon par observation") == attendu,
                 f"{lu.get('Coupon par observation')}")
        attendu = f"{S.participation_cp(mk, 5.0, 1.0, 3.0)*100:,.0f}%"
        controle(f"{nom:5} | page Capital Protection affiche la participation de sales.py",
                 lu.get("Participation offerte") == attendu, f"{lu.get('Participation offerte')}")


# 2. Aller-retour du reverse enquiry --------------------------------------------
print("\n2. Aller-retour du reverse enquiry (tolerance 0,5 pt)")

for nom, mk in MODELES:
    for bar in (0.55, 0.65, 0.75, 0.85):
        c = S.coupon_brc(mk, 1.0, bar)
        rep = S.brc_barriere(mk, c, 1.0)
        controle(f"{nom:5} | BRC barriere {bar:.0%} -> coupon {c*100:.2f}% -> barriere",
                 rep.possible and proche(rep.valeur, bar, 0.005) and rep.obtenu >= c,
                 f"{rep.valeur*100:.2f}%")
    c = S.coupon_brc(mk, 1.5, 0.65, 1.0, False)
    rep = S.brc_barriere(mk, c, 1.5, 1.0, False)
    controle(f"{nom:5} | BRC europeen barriere 65 % aller-retour",
             rep.possible and proche(rep.valeur, 0.65, 0.005), f"{rep.valeur*100:.2f}%")
    for k in (0.85, 0.95, 1.05):
        c = S.coupon_brc(mk, 1.0, 0.70, k)
        rep = S.brc_strike(mk, c, 1.0, 0.70)
        controle(f"{nom:5} | BRC strike {k:.0%} aller-retour",
                 rep.possible and proche(rep.valeur, k, 0.005), f"{rep.valeur*100:.2f}%")
    # la reponse peut etre plus courte que T si une maturite plus courte paie autant
    # (courbe plate sous Bates) : on verifie que c'est bien la plus courte qui paie
    for T in (1.0, 1.5):
        c = S.coupon_brc(mk, T, 0.60)
        rep = S.brc_maturite(mk, c, 0.60)
        plus_courtes = [x for t_, x in rep.courbe if t_ < rep.valeur]
        controle(f"{nom:5} | BRC maturite {T:g} an(s) aller-retour", rep.possible
                 and rep.valeur <= T and rep.obtenu >= c and all(x < c for x in plus_courtes),
                 f"{rep.valeur:g} an(s), {rep.obtenu*100:.2f}% pour {c*100:.2f}%")
    # sous BS + skew, le coupon de l'autocall culmine vers 70 % de barriere de capital
    for bcap in ((0.45, 0.55, 0.65) if not mk.bates else (0.50, 0.70, 0.85)):
        c = S.coupon_autocall(mk, 3.0, 4, 1.0, 0.70, bcap) * 4
        rep = S.autocall_barriere_capital(mk, c, 3.0, 4)
        controle(f"{nom:5} | Autocall barriere de capital {bcap:.0%} aller-retour",
                 rep.possible and proche(rep.valeur, bcap, 0.005), f"{rep.valeur*100:.2f}%")
    for tg in (0.90, 1.00):
        c = S.coupon_autocall(mk, 3.0, 4, tg, 0.70, 0.60) * 4
        rep = S.autocall_trigger(mk, c, 3.0, 4)
        # marge sur la vie attendue : pres de 100 %, le coupon varie peu avec le trigger ;
        # le solveur rend le trigger le plus bas qui tient la demande
        controle(f"{nom:5} | Autocall trigger {tg:.0%} aller-retour",
                 rep.possible and rep.valeur <= tg + 0.005 and rep.obtenu >= c - 1e-9,
                 f"{rep.valeur*100:.2f}% pour {rep.obtenu*100:.2f}% p.a.")
    for prot in (0.85, 0.95):
        p = S.participation_cp(mk, 5.0, prot)
        rep = S.cp_protection(mk, p, 5.0)
        controle(f"{nom:5} | Capital Protection protection {prot:.0%} aller-retour",
                 rep.possible and proche(rep.valeur, prot, 1e-6), f"{rep.valeur*100:.4f}%")
    for cap in (1.30, 1.60):
        p = S.participation_cp(mk, 5.0, 1.0, cap)
        rep = S.cp_cap(mk, p, 5.0)
        controle(f"{nom:5} | Capital Protection cap {cap:.0%} aller-retour",
                 rep.possible and proche(rep.valeur, cap, 1e-6), f"{rep.valeur*100:.4f}%")

mk = BATES
c = S.coupon_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60, True, 3, 0.5) * 4
rep = S.autocall_barriere_capital(mk, c, 3.0, 4, n_actifs=3, correlation=0.5)
controle("Bates | Autocall worst-of 3 barriere de capital 60 % aller-retour",
         rep.possible and proche(rep.valeur, 0.60, 0.005), f"{rep.valeur*100:.2f}%")


# 3. Ou va l'argent -------------------------------------------------------------
print("\n3. Ou va l'argent")

for nom, base in MODELES:
    for spread, retro in ((0, 0.0), (150, 0.0), (150, 0.01)):
        mk = base.avec(spread=spread, retro=retro)
        etiq = f"{nom:5} spread {spread:3} retro {retro:.0%}"
        cascades = [("BRC", S.cascade_brc(mk, 1.0, 0.70), 1.0),
                    ("Autocall", S.cascade_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60), 3.0),
                    ("Worst-of", S.cascade_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60, True, 3, 0.6),
                     3.0),
                    ("Capital Protection", S.cascade_cp(mk, 5.0, 1.0, 3.0), 5.0)]
        for produit, cas, T in cascades:
            lignes = {l.cle: l.montant for l in cas.lignes}
            somme = sum(lignes.values())
            # autocall : frais sur la duree de vie attendue (moteur.coupon_autocall)
            d = cas.infos.get("duree_vie", T)
            controle(f"{etiq} | {produit} : somme = nominal, marge = {mk.marge:.1%} x {d:.2f} an(s)",
                     proche(somme, N, 1e-6) and proche(lignes["marge"], mk.marge * d * N, 1e-6)
                     and proche(lignes["retro"], retro * d * N, 1e-9),
                     f"somme {somme:,.6f}, marge {lignes['marge']:.4f}")
            controle(f"{etiq} | {produit} : risque emetteur nul sans spread, negatif sinon",
                     (lignes["credit"] == 0) if spread == 0 else (lignes["credit"] < 0),
                     f"{lignes['credit']:.2f} EUR")
        cp = cascades[3][1]
        controle(f"{etiq} | Capital Protection : obligation + risque = ZC au taux de financement",
                 proche(cp.lignes[0].montant + cp.lignes[1].montant,
                        cp.infos["zc_taux_financement"], 1e-9))
        if retro == 0.0:
            controle(f"{etiq} | cascade BRC donne le coupon de la page",
                     cascades[0][1].resultat == page_brc(mk, 1.0, 0.70, 1.0, True))
            controle(f"{etiq} | cascade Autocall donne le coupon de la page",
                     cascades[1][1].resultat == page_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60,
                                                              True, 1, 0.6))

    # la retro coute exactement ce qui est annonce (payoff affine en coupon)
    mk0, mk1 = base.avec(spread=100), base.avec(spread=100, retro=0.01)
    for produit, f in (("BRC", lambda m: S.cascade_brc(m, 1.0, 0.70)),
                       ("Autocall", lambda m: S.cascade_autocall(m, 3.0, 4, 1.0, 0.70, 0.60)),
                       ("Capital Protection", lambda m: S.cascade_cp(m, 5.0))):
        c0, c1 = f(mk0), f(mk1)
        res0 = c0.infos.get("coupon_pa", c0.resultat)
        res1 = c1.infos.get("coupon_pa", c1.resultat)
        controle(f"{nom:5} | {produit} : 1 % de retro par an coute ce que la cascade annonce",
                 proche(res0 - res1, c0.infos["cout_retro_par_pt"], 1e-9),
                 f"-{(res0 - res1)*100:.2f} pt")

c0 = S.coupon_brc(BS, 1.0, 0.70)
c1 = S.coupon_brc(BS.avec(spread=200), 1.0, 0.70)
controle("Le spread emetteur augmente le coupon (comme dans les pages)", c1 > c0,
         f"{c0*100:.2f}% -> {c1*100:.2f}%")


# 4. Marche secondaire ----------------------------------------------------------
print("\n4. Marche secondaire")

for nom, base in MODELES:
    mk = base.avec(spread=120, retro=0.01)
    produits = [
        ("BRC", 1.0, lambda: S.valeur_secondaire_brc(mk, mk, 1.0, 0.70, 1.0, True, 0.0, 1.0)),
        ("Autocall", 3.0, lambda: S.valeur_secondaire_autocall(mk, mk, 3.0, 4, 1.0, 0.70, 0.60,
                                                               True, 0.0, 1.0)),
        ("Worst-of", 3.0, lambda: S.valeur_secondaire_autocall(
            mk, mk, 3.0, 4, 1.0, 0.70, 0.60, True, 0.0, 1.0, n_actifs=3, correlation=0.6)),
        ("Capital Protection", 5.0, lambda: S.valeur_secondaire_cp(mk, mk, 5.0, 1.0, 3.0,
                                                                   0.0, 1.0)),
    ]
    vies = {"Autocall": S.cascade_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60).infos["duree_vie"],
            "Worst-of": S.cascade_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60, True, 3,
                                           0.6).infos["duree_vie"]}
    for produit, T, f in produits:
        v = f()
        attendu = N * (1 - (mk.marge + mk.retro) * vies.get(produit, T))
        controle(f"{nom:5} | {produit} : a t = 0, valeur = nominal - marge - retro",
                 proche(v.valeur, attendu, 1e-6) and proche(v.valeur_emission, attendu, 1e-6)
                 and all(abs(e) < 1e-6 for _, e in v.decomposition),
                 f"{v.valeur:,.4f} pour {attendu:,.4f}")
        controle(f"{nom:5} | {produit} : fourchette de 1 % autour de la valeur",
                 proche(v.prix_rachat, v.valeur - 5.0, 1e-9)
                 and proche(v.prix_vente, v.valeur + 5.0, 1e-9))

    # meme pricer que le moteur a t = 0, au bit pres ou presque
    for n_act in (1, 3):
        for memoire in (True, False):
            ch, _ = mk.panier(n_act, 0.6, 3.0, 0.60)
            tau = mk.defauts(len(ch), 3.0)
            obs = S._observations(3.0, 4, 0.0, ch.shape[1] - 1)
            a = S._composantes_autocall(ch, 3.0, obs, mk, 1.0, 0.70, 0.60, memoire, 0.02, 0,
                                        tau)["pv"]
            b = M.pv_autocall(ch, 3.0, mk.r, N, 12, 1.0, 0.70, 0.60, memoire, 0.02, tau,
                              mk.recovery)
            controle(f"{nom:5} | pricer en cours de vie = moteur.pv_autocall a t = 0 "
                     f"({n_act} ss-jacent(s), memoire {memoire})", proche(a, b, 1e-9 * N))
    ch, sg = mk.trajectoires(1.0, 0.70)
    tau = mk.defauts(len(ch), 1.0)
    a = S._composantes_brc(ch, sg, 1.0, mk, 0.70, 1.0, True, 0.08, False, tau)["pv"]
    b = M.pv_brc(ch, 1.0, mk.r, N, 0.70, 1.0, True, 0.08, sg, tau, mk.recovery)
    controle(f"{nom:5} | pricer BRC en cours de vie = moteur.pv_brc a t = 0", proche(a, b, 1e-9 * N))

    # la decomposition retombe exactement sur la valeur du jour
    jour = mk.avec(vol_atm=0.28, r=0.035, q=0.02, spread=200)
    v = S.valeur_secondaire_brc(mk, jour, 1.5, 0.65, 1.0, True, 0.6, 0.82)
    controle(f"{nom:5} | BRC : emission + effets = valeur du jour",
             proche(v.valeur_emission + sum(e for _, e in v.decomposition), v.valeur, 1e-9),
             ", ".join(f"{n_} {e:+.1f}" for n_, e in v.decomposition))
    eff = dict(v.decomposition)
    # (pas de signe impose aux taux : taux en hausse et dividende en baisse montent le
    # forward, ce qui allege le put vendu et peut faire monter la valeur)
    controle(f"{nom:5} | BRC : sous-jacent en baisse, vol et spread en hausse = pertes",
             eff["Sous-jacent"] < 0 and eff["Volatilite"] < 0 and eff["Spread emetteur"] < 0
             and eff["Temps ecoule"] > 0)
    v = S.valeur_secondaire_autocall(mk, jour, 3.0, 4, 1.0, 0.70, 0.60, True, 1.1, 0.9,
                                     niveaux_observes=[0.95, 0.68, 0.97, 0.91])
    controle(f"{nom:5} | Autocall : emission + effets = valeur du jour",
             proche(v.valeur_emission + sum(e for _, e in v.decomposition), v.valeur, 1e-9)
             and v.infos["observations_passees"] == 4,
             f"{v.valeur:,.1f} EUR, {v.coupons_encaisses:,.1f} EUR de coupons encaisses")
    v = S.valeur_secondaire_cp(mk, mk.avec(r=0.045), 5.0, 1.0, 3.0, 1.0, 1.0)
    controle(f"{nom:5} | Capital Protection : la hausse des taux fait baisser la valeur",
             dict(v.decomposition)["Taux et dividende"] < 0,
             f"{dict(v.decomposition)['Taux et dividende']:+.1f} EUR")

    # sens de variation
    def vbrc(**kw):
        p = dict(T=1.0, barriere=0.70, strike=1.0, continu=True, t=0.5, niveau=0.85)
        p.update(kw)
        return S.valeur_secondaire_brc(mk, mk, **p).valeur
    controle(f"{nom:5} | BRC : barriere deja touchee = valeur plus basse",
             vbrc(touchee=True) < vbrc(touchee=False),
             f"{vbrc(touchee=True):,.0f} contre {vbrc(touchee=False):,.0f}")
    controle(f"{nom:5} | BRC : sous-jacent plus haut = valeur plus haute",
             vbrc(niveau=0.95) > vbrc(niveau=0.85) > vbrc(niveau=0.75))
    controle(f"{nom:5} | BRC : sous la barriere continue, touchee automatiquement",
             vbrc(niveau=0.65) == vbrc(niveau=0.65, touchee=True))
    cpn = S.coupon_brc(mk, 1.0, 0.70)
    dt = 1.0 / 52
    v = vbrc(t=1.0 - dt, niveau=1.20)
    attendu = N * (1 + cpn) * math.exp(-mk.r * dt)
    controle(f"{nom:5} | BRC a une semaine de l'echeance, loin au-dessus : nominal + coupon",
             proche(v, attendu, 0.005 * N), f"{v:,.1f} pour {attendu:,.1f}")
    v = vbrc(t=1.0 - dt, niveau=0.60, touchee=True)
    attendu = N * (0.60 + cpn) * math.exp(-mk.r * dt)
    controle(f"{nom:5} | BRC a une semaine de l'echeance, touche a 60 % : 60 % + coupon",
             proche(v, attendu, 0.02 * N), f"{v:,.1f} pour {attendu:,.1f}")

e = S.etat_autocall([0.95, 0.65, 0.66, 0.90], 1.0, 0.70, True)
controle("Etat autocall : memoire rattrapee (1 + 3 coupons), pas de rappel",
         e["vivant"] and e["coupons_verses"] == 4 and e["manques"] == 0)
e = S.etat_autocall([0.95, 0.65, 0.66], 1.0, 0.70, False)
controle("Etat autocall sans memoire : coupons manques perdus",
         e["vivant"] and e["coupons_verses"] == 1 and e["manques"] == 2)
e = S.etat_autocall([0.95, 1.02, 0.80], 1.0, 0.70)
controle("Etat autocall : rappele a la 2e observation", not e["vivant"] and e["rappel_obs"] == 2)
v = S.valeur_secondaire_autocall(BS, BS, 3.0, 4, 1.0, 0.70, 0.60, True, 0.6, 0.9,
                                 niveaux_observes=[0.95, 1.02])
controle("Autocall deja rappele : plus rien a racheter",
         v.valeur == 0 and v.coupons_encaisses > 0, v.message[:60])
v1 = S.valeur_secondaire_autocall(BS, BS, 3.0, 4, 1.0, 0.70, 0.60, True, 0.6, 0.9, manques=2)
v0 = S.valeur_secondaire_autocall(BS, BS, 3.0, 4, 1.0, 0.70, 0.60, True, 0.6, 0.9, manques=0)
controle("Autocall : des coupons manques en memoire augmentent la valeur", v1.valeur > v0.valeur,
         f"{v0.valeur:,.1f} -> {v1.valeur:,.1f}")

for nom_, f in (("date apres l'echeance",
                 lambda: S.valeur_secondaire_brc(BS, BS, 1.0, 0.7, 1.0, True, 1.0, 1.0)),
                ("observations incoherentes avec la date",
                 lambda: S.valeur_secondaire_autocall(BS, BS, 3.0, 4, 1.0, 0.7, 0.6, True, 0.6,
                                                      1.0, niveaux_observes=[0.9]))):
    try:
        f()
        controle(f"Refus propre : {nom_}", False)
    except ValueError:
        controle(f"Refus propre : {nom_}", True)


# 5. Cas limites ------------------------------------------------------------------
print("\n5. Cas limites")

rep = S.brc_barriere(BS, 0.50, 1.0)
controle("BRC : coupon de 50 % impossible, meilleure valeur donnee",
         not rep.possible and rep.obtenu == max(c for _, c in rep.courbe)
         and rep.message.startswith("Impossible"), rep.message)
rep = S.brc_barriere(BS, 0.005, 1.0)
controle("BRC : coupon de 0,5 % deja paye a la barriere la plus basse",
         rep.possible and rep.valeur == 0.40 and rep.obtenu >= 0.005, rep.message)
rep = S.brc_maturite(BATES, 0.30, 0.60)
controle("BRC : maturite impossible pour 30 %", not rep.possible and len(rep.courbe) == 12,
         rep.message)
rep = S.autocall_trigger(BS, 0.30, 3.0, 4)
controle("Autocall : trigger impossible pour 30 %", not rep.possible, rep.message)
rep = S.cp_cap(BS.avec(r=0.0), 1.0, 5.0, 1.0)
controle("Capital Protection a taux nul : budget negatif, aucun cap n'aide",
         not rep.possible and rep.obtenu == 0.0, rep.message)
rep = S.cp_protection(BS.avec(r=0.005), 2.5, 5.0)
controle("Capital Protection a 0,5 % : 250 % de participation hors d'atteinte",
         not rep.possible and rep.valeur == 0.70, rep.message)
rep = S.cp_cap(BS.avec(r=0.06), 0.5, 5.0)
controle("Capital Protection a 6 % : pas besoin de cap", rep.possible and rep.valeur == 3.0,
         rep.message)
rep = S.cp_protection(BS.avec(r=0.06), 0.5, 5.0)
controle("Capital Protection a 6 % : protection 100 % suffit", rep.possible
         and rep.valeur == 1.0 and rep.obtenu >= 0.5, rep.message)
mk = BS.avec(n_paths=9999, spread=100)
controle("Nombre de trajectoires impair accepte",
         proche(S.cascade_brc(mk, 1.0, 0.7).somme, N, 1e-6)
         and proche(S.valeur_secondaire_brc(mk, mk, 1.0, 0.7, 1.0, True, 0.0, 1.0).valeur,
                    N * (1 - mk.marge), 1e-6))


# 6. Temps de calcul (10 000 trajectoires, configuration par defaut des pages) ------
print("\n6. Temps de calcul")
for nom, mk in MODELES:
    j = mk.avec(vol_atm=0.27, r=0.035)
    appels = [
        ("brc_barriere", lambda: S.brc_barriere(mk, 0.08, 1.0)),
        ("brc_strike", lambda: S.brc_strike(mk, 0.07, 1.0, 0.70)),
        ("brc_maturite", lambda: S.brc_maturite(mk, 0.08, 0.60)),
        ("autocall_barriere_capital", lambda: S.autocall_barriere_capital(mk, 0.07, 3.0, 4)),
        ("autocall_barriere_capital worst-of 3",
         lambda: S.autocall_barriere_capital(mk, 0.09, 3.0, 4, n_actifs=3)),
        ("autocall_trigger", lambda: S.autocall_trigger(mk, 0.06, 3.0, 4)),
        ("cp_protection + cp_cap", lambda: (S.cp_protection(mk, 0.8, 5.0), S.cp_cap(mk, 1.0, 5.0))),
        ("cascade_brc", lambda: S.cascade_brc(mk, 1.0, 0.70)),
        ("cascade_autocall", lambda: S.cascade_autocall(mk, 3.0, 4, 1.0, 0.70, 0.60)),
        ("valeur_secondaire_brc", lambda: S.valeur_secondaire_brc(mk, j, 1.0, 0.7, 1.0, True,
                                                                  0.5, 0.85)),
        ("valeur_secondaire_autocall", lambda: S.valeur_secondaire_autocall(
            mk, j, 3.0, 4, 1.0, 0.7, 0.6, True, 1.1, 0.9)),
    ]
    for nom_, f in appels:
        t0 = time.time()
        f()
        dt = time.time() - t0
        controle(f"{nom:5} | {nom_} en moins de 3 s", dt < 3.0, f"{dt:.2f} s")

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
