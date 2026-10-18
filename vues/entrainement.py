"""Entrainement : questions d'entretien de Sales en produits structures."""

import math
import random

import numpy as np
import streamlit as st

import moteur as M
from moteur import S0
from communs import sim_bs, sim_bates, sim_panier, sim_defaut


CLIENT = "Discours client"
MECANIQUE = "Mecanique des produits"
MARCHE = "Parametres de marche"
EMETTEUR = "Risque emetteur"
GRECQUES = "Grecques et couverture"
THEMES = [CLIENT, MECANIQUE, MARCHE, EMETTEUR, GRECQUES]
TOUS = "Tous les themes"


# =====================================================================
# OUTILS DE CALCUL
# Les demonstrations reprennent les calculs des pages produits, avec la
# vue de marche de la barre laterale. Sans choc, les trajectoires sont
# exactement celles des pages (meme cache, meme graine).
# =====================================================================

def _pct(x, d=1):
    return f"{x*100:,.{d}f}%"


def _pts(x):
    return f"{x*100:+,.2f} pts"


def _ligne(*metriques):
    """Une rangee de st.metric : (libelle, valeur) ou (libelle, valeur, ecart)."""
    for col, x in zip(st.columns(len(metriques)), metriques):
        col.metric(x[0], x[1], x[2] if len(x) > 2 else None, delta_color="off")


def _chemins(m, T, niveau_ref, dvol=0.0, vol=None, skew=None, rho=None, r=None, spot=S0):
    """
    Trajectoires du modele choisi, avec un choc eventuel.
    dvol deplace toute la vol (sous Bates : vol de depart ET vol long terme) ;
    vol remplace seulement la vol de depart (un choc de court terme).
    """
    r = m.r if r is None else r
    v = (m.vol_atm if vol is None else vol) + dvol
    if m.BATES:
        return sim_bates(spot, T, r, m.q, m.n_paths, v ** 2, (m.theta_v + dvol) ** 2,
                         m.kappa, m.xi, m.rho if rho is None else rho,
                         m.lam, m.mu_j, m.sig_j, 42)
    sk = m.skew if skew is None else skew
    return sim_bs(spot, T, r, m.q, M.vol_au_strike(niveau_ref, v, sk), m.n_paths, 42)


def _brc(m, chemins, T=1.0, bar=0.70, continu=True, marge=None, tau="vue", r=None):
    """Coupon d'un BRC strike 100 %, comme la page BRC. tau='vue' : spread de la barre."""
    ch, sg = chemins
    if isinstance(tau, str):
        tau = m.defauts(len(ch), T)
    return M.coupon_brc(ch, T, m.r if r is None else r, m.nominal, bar, 1.0, continu,
                        m.marge if marge is None else marge, sg, tau, m.recovery)


def _risques_brc(chemins, T=1.0, bar=0.70, continu=True):
    """(proba de toucher la barriere, proba de perte en capital), strike 100."""
    ch, sg = chemins
    ST = ch[:, -1].astype(np.float64)
    p_no = (M.proba_non_touche(ch, bar * S0, sg, T) if continu
            else (ST > bar * S0).astype(np.float64))
    return 1.0 - p_no.mean(), ((1.0 - p_no) * (ST < S0)).mean()


def _participation(m, T=5.0, protection=1.0, cap=3.0, r=None, q=None, marge=None):
    """Capital garanti, meme calcul que la page : (participation, zero-coupon, budget, call)."""
    r = m.r if r is None else r
    q = m.q if q is None else q
    zc = protection * math.exp(-(r + m.spread / 10000.0) * T)
    budget = 1.0 - zc - (m.marge if marge is None else marge) * T
    v_call = m.vol_atm if m.BATES else M.vol_au_strike(1.0, m.vol_atm, m.skew)
    v_cap = m.vol_atm if m.BATES else M.vol_au_strike(cap, m.vol_atm, m.skew)
    cout = max((M.bs_call(S0, S0, T, r, v_call, q)
                - M.bs_call(S0, cap * S0, T, r, v_cap, q)) / S0, 1e-9)
    return max(budget / cout, 0.0), zc, budget, cout


# Autocall de reference : 3 ans, trimestriel, trigger 100, coupon 70, capital 60
AC_T, AC_OBS = 3.0, 12


def _autocall(m, chemins, memoire=True, tau="vue"):
    """(coupon annuel, duree de vie moyenne, proba de rappel, proba de perte)."""
    ch = chemins[0]
    if isinstance(tau, str):
        tau = m.defauts(len(ch), AC_T)
    cpn = M.coupon_autocall(ch, AC_T, m.r, m.nominal, AC_OBS, 1.0, 0.70, 0.60, memoire,
                            m.marge, tau, m.recovery)
    duree, p_rappel, p_perte = M.stats_autocall(ch, AC_T, AC_OBS, 1.0, 0.60)
    return cpn * AC_OBS / AC_T, duree, p_rappel, p_perte


def _vol_put(m):
    """
    Vol de la formule fermee, prise au niveau de la barriere comme dans les pages
    produits : celle du skew en Black-Scholes ; sous Bates, la vol implicite a 70 %
    extraite des trajectoires simulees, pour rester coherent avec le coupon.
    """
    if not m.BATES:
        return M.vol_au_strike(0.70, m.vol_atm, m.skew)
    v = M.vol_implicite_bates(_chemins(m, 1.0, 0.70)[0], 1.0, m.r, m.q, [0.70])[0]
    return float(v) if np.isfinite(v) else m.vol_atm


def _put_di(m, S, T, sigma):
    """Put down-and-in strike 100, barriere 70, pour 100 de sous-jacent."""
    return M.put_down_and_in(S, 100.0, 70.0, T, m.r, sigma, m.q)


def _couverture(g, S):
    """Actions a detenir contre un put achete, en fraction du nominal (strike 100)."""
    return -g["delta"] * S / 100


def _grecques_di(m, S, T, sigma):
    prix = lambda S, T, sigma, r: M.put_down_and_in(S, 100.0, 70.0, T, r, sigma, m.q)
    return M.grecques(prix, S, T, sigma, m.r)


def _note_formule(m, sigma):
    st.caption(f"Formule fermee de Reiner-Rubinstein, vol {sigma*100:.0f}% : "
               + ("la vol implicite a 70 % du smile genere par Bates, la formule restant "
                  "en Black-Scholes." if m.BATES else "la vol au niveau de la barriere."))


# =====================================================================
# DEMONSTRATIONS CHIFFREES
# =====================================================================

def demo_bonne_affaire(m):
    st.caption("BRC 1 an, barriere 70 % observee en continu, strike 100 %.")
    ch = _chemins(m, 1.0, 0.70)
    cpn = _brc(m, ch)
    toucher, perte = _risques_brc(ch)
    _ligne(("Coupon du BRC", f"{cpn*100:,.2f}% p.a."),
           ("Taux sans risque", f"{m.r*100:,.2f}%"),
           ("Proba de toucher la barriere", _pct(toucher)),
           ("Proba de perte en capital", _pct(perte)))
    st.caption(f"Les {(cpn - m.r)*100:,.1f} points au-dessus du taux sans risque "
               "remunerent la protection que le client vend.")


def demo_securite(m):
    st.caption("BRC 1 an, barriere 70 % continue, strike 100 % : trois facons de "
               "rendre le produit plus sur.")
    variantes = [("BRC de depart", _chemins(m, 1.0, 0.70), 1.0, 0.70, True),
                 ("Barriere a 60 %", _chemins(m, 1.0, 0.60), 1.0, 0.60, True),
                 ("Barriere observee a l'echeance", _chemins(m, 1.0, 0.70), 1.0, 0.70, False),
                 ("Maturite 6 mois", _chemins(m, 0.5, 0.70), 0.5, 0.70, True)]
    lignes = []
    for nom, ch, T, bar, continu in variantes:
        cpn = _brc(m, ch, T=T, bar=bar, continu=continu)
        toucher, perte = _risques_brc(ch, T, bar, continu)
        lignes.append({"Variante": nom, "Coupon annuel": f"{cpn*100:,.2f}%",
                       "Coupon total": f"{cpn*T*100:,.2f}%",
                       "Proba de perte": _pct(perte), "_c": cpn, "_p": perte})
    a, b = lignes[0], lignes[1]
    _ligne(("Coupon, barriere 70 %", a["Coupon annuel"]),
           ("Coupon, barriere 60 %", b["Coupon annuel"], _pts(b["_c"] - a["_c"])),
           ("Proba de perte, 70 %", a["Proba de perte"]),
           ("Proba de perte, 60 %", b["Proba de perte"], _pts(b["_p"] - a["_p"])))
    st.table([{k: v for k, v in x.items() if not k.startswith("_")} for x in lignes],
             hide_index=True)


def demo_autocall(m):
    st.caption("Autocall Phoenix 3 ans, observations trimestrielles, trigger 100 %, "
               "barriere de coupon 70 %, barriere de capital 60 %, effet memoire.")
    ch = _chemins(m, AC_T, 0.60)
    cpn, duree, _, perte = _autocall(m, ch)
    i1 = M.idx_obs(ch[0].shape[1] - 1, AC_OBS)[0]
    rappel_1 = (ch[0][:, i1] >= S0).mean()
    _ligne(("Coupon affiche", f"{cpn*100:,.2f}% p.a."),
           ("Duree de vie moyenne", f"{duree:,.2f} ans sur {AC_T:.0f}"),
           ("Rappele des le 1er trimestre", _pct(rappel_1)),
           ("Proba de perte en capital", _pct(perte)))


def demo_leviers_garanti(m):
    st.caption("Capital garanti de depart : 5 ans, protection 100 %, sans plafond.")
    p0 = _participation(m)[0]
    leviers = [("8 ans au lieu de 5", _participation(m, T=8.0)[0]),
               ("Protection 90 %", _participation(m, protection=0.90)[0]),
               ("Hausse plafonnee a +50 %", _participation(m, cap=1.5)[0])]
    _ligne(("Participation de depart", _pct(p0, 0)),
           *[(nom, _pct(p, 0), _pts(p - p0)) for nom, p in leviers])
    if p0 <= 0:
        st.caption("Avec ces taux et cette marge, le produit de depart ne peut pas etre "
                   "construit : la garantie et la marge consomment tout le nominal.")


def demo_stress(m):
    choc = min(2 * m.vol_atm, 1.0)
    st.caption(f"BRC 1 an, strike 100 % du nouveau niveau. La vol passe de "
               f"{m.vol_atm*100:.0f}% a {choc*100:.0f}%"
               + (" (sous Bates : vol de depart seulement, la vol long terme ne bouge "
                  "pas, comme dans un vrai pic de stress)." if m.BATES else "."))
    avant = _chemins(m, 1.0, 0.70)
    apres70, apres60 = _chemins(m, 1.0, 0.70, vol=choc), _chemins(m, 1.0, 0.60, vol=choc)
    c0, c1, c2 = _brc(m, avant), _brc(m, apres70), _brc(m, apres60, bar=0.60)
    p0, p1, p2 = (_risques_brc(avant)[1], _risques_brc(apres70)[1],
                  _risques_brc(apres60, bar=0.60)[1])
    _ligne(("Avant, barriere 70 %", f"{c0*100:,.2f}% p.a."),
           ("Apres, barriere 70 %", f"{c1*100:,.2f}% p.a.", _pts(c1 - c0)),
           ("Apres, barriere 60 %", f"{c2*100:,.2f}% p.a.", _pts(c2 - c0)))
    _ligne(("Proba de perte avant", _pct(p0)),
           ("Apres, barriere 70 %", _pct(p1)),
           ("Apres, barriere 60 %", _pct(p2)))
    st.caption("Apres une baisse de 20 %, une barriere a 60 % du nouveau niveau se situe "
               "a 48 % des plus hauts. Les probabilites sont calculees avec la vol de "
               "stress : si la vol realisee retombe ensuite, comme souvent apres un pic, le "
               "risque reel est plus faible que celui que le prix paie.")


def demo_bonus(m):
    T, bar = 2.0, 0.70
    st.caption("Deux ans, barriere 70 % observee en continu. Sans risque emetteur, pour "
               "comparer les deux profils a armes egales.")
    ch, sg = _chemins(m, T, bar)
    ST = ch[:, -1].astype(np.float64)
    p_no = M.proba_non_touche(ch, bar * S0, sg, T)

    def pv_bonus(niveau):
        haut = np.maximum(ST / S0, niveau)
        return math.exp(-m.r * T) * (p_no * haut + (1 - p_no) * ST / S0).mean()

    lo, hi, cible = 1.0, 2.5, 1.0 - m.marge * T
    for _ in range(40):
        mid = (lo + hi) / 2
        hi, lo = (mid, lo) if pv_bonus(mid) > cible else (hi, mid)
    cpn = _brc(m, (ch, sg), T=T, bar=bar, tau=None)
    _ligne(("Bonus : remboursement minimum", _pct(lo)),
           ("Bonus : si le marche fait +30 %", _pct(max(1.3, lo))),
           ("BRC : coupon", f"{cpn*100:,.2f}% p.a."),
           ("BRC : si le marche fait +30 %", _pct(1 + cpn * T)))
    st.caption(f"Proba de toucher la barriere sur deux ans : {(1-p_no.mean())*100:,.1f}%. "
               "Dans ce cas, le bonus disparait et le BRC expose a la baisse.")


def demo_pire(m):
    st.caption("BRC 1 an, barriere 70 % continue, strike 100 %. Remboursement total, "
               "coupon compris, en % du nominal.")
    ch, sg = _chemins(m, 1.0, 0.70)
    cpn = _brc(m, (ch, sg))
    ST = ch[:, -1].astype(np.float64)
    p_no = M.proba_non_touche(ch, 0.70 * S0, sg, 1.0)
    bas = np.minimum(ST / S0, 1.0) + cpn          # si la barriere a ete touchee
    w_perte = (1.0 - p_no) * (ST < S0)
    p_perte = w_perte.mean()
    moyen = (w_perte * bas).sum() / max(w_perte.sum(), 1e-12)
    # queue de distribution : chaque trajectoire pese p_no (intacte) et 1 - p_no (touchee)
    vals = np.concatenate([bas, np.full_like(bas, 1.0 + cpn)])
    poids = np.concatenate([1.0 - p_no, p_no])
    ordre = np.argsort(vals)
    cumul = np.cumsum(poids[ordre]) / poids.sum()
    sel = ordre[:max(int(np.searchsorted(cumul, 0.05)), 1)]
    queue = (vals[sel] * poids[sel]).sum() / max(poids[sel].sum(), 1e-12)
    _ligne(("Proba de perte en capital", _pct(p_perte)),
           ("Remboursement moyen si perte", _pct(moyen) if p_perte > 0 else "-"),
           ("Moyenne des 5 % pires cas", _pct(queue)),
           ("Si l'emetteur fait defaut", _pct(m.recovery, 0)))


def demo_marge(m):
    st.caption("BRC 1 an barriere 70 %, et capital garanti 5 ans a 100 %.")
    ch = _chemins(m, 1.0, 0.70)
    c, c0 = _brc(m, ch), _brc(m, ch, marge=0.0)
    p, p0 = _participation(m)[0], _participation(m, marge=0.0)[0]
    _ligne(("Coupon BRC offert", f"{c*100:,.2f}% p.a."),
           ("BRC sans marge", f"{c0*100:,.2f}% p.a.", _pts(c0 - c)),
           ("Participation capital garanti", _pct(p, 0)),
           ("Capital garanti sans marge", _pct(p0, 0), _pts(p0 - p)))
    st.caption(f"Marge de {m.marge*100:,.1f}% par an, soit {m.nominal*m.marge:,.0f} EUR "
               f"par an sur {m.nominal:,.0f} EUR. Sur un produit de 5 ans, elle est "
               "prelevee cinq fois sur le meme budget option.")


def demo_mtm(m):
    st.caption("BRC 1 an, barriere 70 % continue, strike 100 %. Six mois plus tard, le "
               "sous-jacent est a 75. Valeurs en % du nominal, coupon compris.")
    ch, sg = _chemins(m, 1.0, 0.70)
    cpn = _brc(m, (ch, sg))
    v0 = M.pv_brc(ch, 1.0, m.r, 1.0, 0.70, 1.0, True, cpn, sg, m.defauts(len(ch), 1.0),
                  m.recovery)
    # meme produit vu a mi-vie : trajectoires de six mois qui partent de 75, et le
    # coupon total restant du a l'echeance (cpn sur un an = 2 * cpn par an sur six mois)
    ch1, sg1 = apres = _chemins(m, 0.5, 0.70, spot=75.0)
    v1 = M.pv_brc(ch1, 0.5, m.r, 1.0, 0.70, 1.0, True, 2 * cpn, sg1,
                  m.defauts(len(ch1), 0.5), m.recovery)
    toucher, perte = _risques_brc(apres, T=0.5)
    _ligne(("Valeur a l'emission", _pct(v0)),
           ("Valeur aujourd'hui", _pct(v1), _pts(v1 - v0)),
           ("Proba de toucher d'ici l'echeance", _pct(toucher)),
           ("Proba de perte en capital", _pct(perte)))
    st.caption("A l'emission, le produit vaut le nominal moins la marge : c'est ainsi que "
               "le coupon est calcule. Six mois plus tard, meme modele, meme vue de marche.")


def demo_decomposition(m):
    st.caption("BRC 1 an, barriere 70 % continue, strike 100 %. D'ou vient chaque "
               "point de coupon.")
    ch, T = _chemins(m, 1.0, 0.70), 1.0
    total_sans = _brc(m, ch, tau=None)
    total = _brc(m, ch)
    taux = (math.exp(m.r * T) - 1) / T
    marge = -m.marge * math.exp(m.r * T)
    put = total_sans - taux - marge
    parts = [("Taux", _pts(taux)), ("Prime du put vendu", _pts(put))]
    if m.spread > 0:
        parts.append((f"Spread emetteur ({m.spread} bp)", _pts(total - total_sans)))
    parts += [("Marge banque", _pts(marge)), ("Coupon offert", f"{total*100:,.2f}% p.a.")]
    _ligne(*parts)


def demo_capital(m):
    st.caption("Capital garanti 5 ans, protection 100 %, sans plafond. Montants pour "
               f"{m.nominal:,.0f} EUR.")
    p, zc, budget, cout = _participation(m)
    _ligne(("Mis de cote pour la garantie", f"{zc*m.nominal:,.0f} EUR"),
           ("Budget option, apres marge", f"{budget*m.nominal:,.0f} EUR"),
           ("Prix du call a la monnaie", f"{cout*m.nominal:,.0f} EUR"),
           ("Participation", _pct(p, 0)))


def demo_memoire(m):
    st.caption("Autocall Phoenix 3 ans, trimestriel, trigger 100 %, barriere de coupon "
               "70 %, barriere de capital 60 %.")
    ch = _chemins(m, AC_T, 0.60)
    avec, sans = _autocall(m, ch, True)[0], _autocall(m, ch, False)[0]
    _ligne(("Coupon sans memoire", f"{sans*100:,.2f}% p.a."),
           ("Coupon avec memoire", f"{avec*100:,.2f}% p.a.", _pts(avec - sans)))


def demo_observation(m):
    st.caption("BRC 1 an, barriere 70 %, strike 100 %.")
    ch = _chemins(m, 1.0, 0.70)
    cc, ce = _brc(m, ch), _brc(m, ch, continu=False)
    pc, pe = _risques_brc(ch)[1], _risques_brc(ch, continu=False)[1]
    _ligne(("Coupon, observation continue", f"{cc*100:,.2f}% p.a."),
           ("Coupon, a l'echeance", f"{ce*100:,.2f}% p.a.", _pts(ce - cc)),
           ("Proba de perte, continue", _pct(pc)),
           ("Proba de perte, a l'echeance", _pct(pe), _pts(pe - pc)))


def demo_vol_brc(m):
    st.caption("BRC 1 an, barriere 70 % continue, strike 100 %. Vol choquee de +5 points"
               + (" (sous Bates : vol de depart et vol long terme)." if m.BATES else "."))
    base, choc = _chemins(m, 1.0, 0.70), _chemins(m, 1.0, 0.70, dvol=0.05)
    c0, c1 = _brc(m, base), _brc(m, choc)
    t0, t1 = _risques_brc(base)[0], _risques_brc(choc)[0]
    v = m.vol_atm * 100
    _ligne((f"Coupon, vol a {v:.0f}%", f"{c0*100:,.2f}% p.a."),
           (f"Coupon, vol a {v+5:.0f}%", f"{c1*100:,.2f}% p.a.", _pts(c1 - c0)),
           (f"Proba de toucher, vol a {v:.0f}%", _pct(t0)),
           (f"Proba de toucher, vol a {v+5:.0f}%", _pct(t1), _pts(t1 - t0)))


def demo_taux_garanti(m):
    r_alt = 0.005 if m.r > 0.015 else 0.03
    st.caption("Capital garanti 5 ans, protection 100 %, sans plafond. Montants pour "
               f"{m.nominal:,.0f} EUR.")
    p0, _, b0, _ = _participation(m)
    p1, _, b1, _ = _participation(m, r=r_alt)
    _ligne((f"Budget option a {m.r*100:.2f}%", f"{b0*m.nominal:,.0f} EUR"),
           (f"Participation a {m.r*100:.2f}%", _pct(p0, 0)),
           (f"Budget option a {r_alt*100:.2f}%", f"{b1*m.nominal:,.0f} EUR"),
           (f"Participation a {r_alt*100:.2f}%", _pct(p1, 0), _pts(p1 - p0)))
    if min(b0, b1) <= 0:
        st.caption("Un budget negatif veut dire que la garantie et la marge consomment "
                   "deja tout le nominal : le produit ne peut pas etre construit.")


def demo_worst_of(m):
    st.caption("Autocall Phoenix 3 ans, trimestriel, trigger 100 %, coupon 70 %, capital "
               "60 %. Les trois cas en Black-Scholes multi-actifs, comme dans la page "
               "Autocall, sans risque emetteur.")
    v_p = m.vol_atm if m.BATES else M.vol_au_strike(0.60, m.vol_atm, m.skew)
    n = min(m.n_paths, 20000)
    cas = [("Une seule action", sim_bs(S0, AC_T, m.r, m.q, v_p, n)),
           ("Worst-of 3, correlation 0,8", sim_panier(3, 0.8, S0, AC_T, m.r, m.q, v_p, n)),
           ("Worst-of 3, correlation 0,3", sim_panier(3, 0.3, S0, AC_T, m.r, m.q, v_p, n))]
    res = [(nom, _autocall(m, ch, tau=None)) for nom, ch in cas]
    c0, p0 = res[0][1][0], res[0][1][3]
    _ligne(*[(f"Coupon : {nom.lower()}", f"{x[0]*100:,.2f}% p.a.",
              None if i == 0 else _pts(x[0] - c0)) for i, (nom, x) in enumerate(res)])
    _ligne(*[(f"Proba de perte : {nom.lower()}", _pct(x[3]),
              None if i == 0 else _pts(x[3] - p0)) for i, (nom, x) in enumerate(res)])


def demo_skew(m):
    st.caption("BRC 1 an, barriere 70 % continue, strike 100 %.")
    if m.BATES:
        rho_alt = 0.0 if m.rho < 0 else -0.70
        cas = [(f"correlation spot/vol {rho_alt:+.2f}", rho_alt),
               (f"correlation spot/vol {m.rho:+.2f}", m.rho)]
        res = []
        for nom, rho in cas:
            ch = _chemins(m, 1.0, 0.70, rho=rho)
            vol70 = M.vol_implicite_bates(ch[0], 1.0, m.r, m.q, [0.70])[0]
            res.append((nom, vol70, _brc(m, ch)))
        st.caption("Sous Bates, le skew n'est pas un reglage : il vient de la correlation "
                   "spot/vol. La vol au strike 70 % est extraite des prix simules.")
    else:
        sk_alt = 0.0 if m.skew > 0 else 2.0
        res = [(f"skew {sk:.1f} pt", M.vol_au_strike(0.70, m.vol_atm, sk),
                _brc(m, _chemins(m, 1.0, 0.70, skew=sk))) for sk in (sk_alt, m.skew)]
    (n0, v0, c0), (n1, v1, c1) = res
    _ligne((f"Vol au strike 70 %, {n0}", _pct(v0)),
           (f"Coupon, {n0}", f"{c0*100:,.2f}% p.a."),
           (f"Vol au strike 70 %, {n1}", _pct(v1)),
           (f"Coupon, {n1}", f"{c1*100:,.2f}% p.a.", _pts(c1 - c0)))


def demo_dividende(m):
    q_alt = m.q + 0.02
    st.caption("Capital garanti 5 ans, protection 100 %, sans plafond.")
    p0, _, _, c0 = _participation(m)
    p1, _, _, c1 = _participation(m, q=q_alt)
    _ligne((f"Prix du call, dividende {m.q*100:.1f}%", _pct(c0)),
           (f"Participation, dividende {m.q*100:.1f}%", _pct(p0, 0)),
           (f"Prix du call, dividende {q_alt*100:.1f}%", _pct(c1), _pts(c1 - c0)),
           (f"Participation, dividende {q_alt*100:.1f}%", _pct(p1, 0), _pts(p1 - p0)))


def demo_taux_brc(m):
    st.caption("BRC 1 an, barriere 70 % continue, strike 100 %. Taux choque de +2 points.")
    r1 = m.r + 0.02
    c0 = _brc(m, _chemins(m, 1.0, 0.70))
    c1 = _brc(m, _chemins(m, 1.0, 0.70, r=r1), r=r1)
    _ligne((f"Coupon, taux a {m.r*100:.2f}%", f"{c0*100:,.2f}% p.a."),
           (f"Coupon, taux a {r1*100:.2f}%", f"{c1*100:,.2f}% p.a.", _pts(c1 - c0)),
           ("Part de la hausse transmise", _pct((c1 - c0) / 0.02, 0)))


def demo_spread(m):
    s = m.spread if m.spread > 0 else 150
    st.caption("BRC 1 an, barriere 70 % continue, strike 100 %."
               + ("" if m.spread > 0 else " Le spread de la barre laterale est a 0 : la "
                  "demonstration prend 150 bp."))
    ch = _chemins(m, 1.0, 0.70)
    tau = sim_defaut(len(ch[0]), 1.0, s, m.recovery)
    c0, c1 = _brc(m, ch, tau=None), _brc(m, ch, tau=tau)
    _ligne(("Coupon, emetteur sans risque", f"{c0*100:,.2f}% p.a."),
           (f"Coupon, spread {s} bp", f"{c1*100:,.2f}% p.a.", _pts(c1 - c0)),
           ("Proba de defaut sur 1 an", _pct((tau <= 1.0).mean(), 2)))


def demo_defaut(m):
    s = m.spread if m.spread > 0 else 150
    lam = s / 10000.0 / max(1.0 - m.recovery, 0.01)
    st.caption(f"Spread {s} bp, recouvrement {m.recovery*100:.0f} %."
               + ("" if m.spread > 0 else " Le spread de la barre laterale est a 0 : la "
                  "demonstration prend 150 bp."))
    p3, p5 = 1 - math.exp(-lam * 3), 1 - math.exp(-lam * 5)
    _ligne(("Intensite de defaut", f"{lam*100:,.2f}% par an"),
           ("Proba de defaut sur 3 ans", _pct(p3)),
           ("Proba de defaut sur 5 ans", _pct(p5)),
           ("Perte attendue sur 5 ans", f"{m.nominal*p5*(1-m.recovery):,.0f} EUR"))


def demo_vega(m):
    st.caption(f"Le put down-and-in d'un BRC 1 an, barriere 70 %, strike 100 %, pour "
               f"{m.nominal:,.0f} EUR de nominal.")
    n, sig = m.nominal / 100, _vol_put(m)
    g = _grecques_di(m, 100.0, 1.0, sig)
    choc = (_put_di(m, 100.0, 1.0, sig + 0.05) - g["prix"]) * n
    _ligne(("Valeur du put vendu par le client", f"{g['prix']*n:,.2f} EUR"),
           ("Vega du client", f"{-g['vega']*n:,.2f} EUR par pt"),
           ("Si la vol prend 5 points", f"{-choc:,.2f} EUR"))
    _note_formule(m, sig)


def demo_gamma(m):
    st.caption("Couverture du put down-and-in d'un BRC (barriere 70, strike 100), en "
               "actions detenues par le trader, en % du nominal.")
    sig = _vol_put(m)
    g0, g1 = _grecques_di(m, 100.0, 1.0, sig), _grecques_di(m, 72.0, 1 / 12, sig)
    vanille = lambda S, T, sigma, r: M.bs_put(S, 100.0, T, r, sigma, m.q)
    g2 = M.grecques(vanille, 69.0, 1 / 12, sig, m.r)
    c0, c1, c2 = _couverture(g0, 100), _couverture(g1, 72), _couverture(g2, 69)
    _ligne(("A l'emission, spot 100", _pct(c0, 0)),
           ("A 1 mois de l'echeance, spot 72", _pct(c1, 0)),
           ("Barriere touchee, spot 69", _pct(c2, 0)),
           ("A revendre d'un coup", _pct(c1 - c2, 0)))
    st.caption(f"Gamma a 72 a un mois : {g1['gamma']/max(g0['gamma'], 1e-12):,.0f} fois "
               "celui de l'emission.")
    _note_formule(m, sig)


def demo_theta(m):
    st.caption(f"Le put down-and-in d'un BRC 1 an, barriere 70 %, strike 100 %, pour "
               f"{m.nominal:,.0f} EUR, sous-jacent inchange a 100.")
    n, sig = m.nominal / 100, _vol_put(m)
    p0, p6 = _put_di(m, 100.0, 1.0, sig) * n, _put_di(m, 100.0, 0.5, sig) * n
    theta = _grecques_di(m, 100.0, 1.0, sig)["theta"] * n
    _ligne(("Put vendu, a l'emission", f"{p0:,.2f} EUR"),
           ("Meme spot, 6 mois plus tard", f"{p6:,.2f} EUR"),
           ("Encaisse par le client", f"{p0-p6:,.2f} EUR"),
           ("Par jour, au depart", f"{-theta:,.3f} EUR"))
    _note_formule(m, sig)


def demo_couverture(m):
    st.caption("La banque achete le put down-and-in (barriere 70, strike 100). Actions "
               "detenues par le trader pour etre neutre au delta, en % du nominal.")
    sig = _vol_put(m)
    g0, g1 = _grecques_di(m, 100.0, 1.0, sig), _grecques_di(m, 85.0, 0.75, sig)
    c0, c1 = _couverture(g0, 100), _couverture(g1, 85)
    _ligne(("A l'emission, spot 100", _pct(c0, 0)),
           ("3 mois plus tard, spot 85", _pct(c1, 0)),
           ("Rachete dans la baisse", _pct(c1 - c0, 0)))
    _note_formule(m, sig)


# =====================================================================
# LA BANQUE DE QUESTIONS
# =====================================================================

def Q(theme, question, reponse, cle, demo=None):
    return {"theme": theme, "q": question, "r": reponse, "cle": cle, "demo": demo}


QUESTIONS = [
    # ---------------- Discours client -------------------------------
    Q(CLIENT,
      "Un client vous dit : « Ce BRC paie 9 %, c'est une bonne affaire. » Que lui "
      "repondez-vous ?",
      "Je ne le contredis pas, je lui explique d'ou viennent les 9 %. Le coupon n'est pas "
      "un interet : c'est la prime que la banque lui verse pour l'assurer contre une forte "
      "baisse du sous-jacent. 9 % quand le taux sans risque est a 3 %, cela veut dire que "
      "la protection qu'il vend a de la valeur : vol elevee, barriere proche, ou emetteur "
      "qui paie un spread. Ensuite je verifie que sa vue colle au produit : il doit etre "
      "pret a detenir l'action s'il est livre sous le strike. Si oui, le produit lui "
      "convient ; sinon on baisse la barriere et on accepte moins de coupon.",
      "Le coupon est le prix d'un risque, jamais un cadeau. Relier le chiffre a la vue du "
      "client.",
      demo_bonne_affaire),
    Q(CLIENT,
      "Un client veut plus de securite sur son BRC. Que lui proposez-vous, et combien ca "
      "coute ?",
      "Trois leviers, tous payes en coupon. Baisser la barriere, de 70 % a 60 % par "
      "exemple. Observer la barriere seulement a l'echeance au lieu d'en continu. "
      "Raccourcir la maturite, ce qui laisse moins de temps au sous-jacent pour chuter. "
      "Chaque levier reduit la valeur du put que le client vend, donc la prime qu'on lui "
      "reverse. Mon role est de chiffrer l'arbitrage, « vous perdez tant de coupon et la "
      "probabilite de perte passe de A a B », puis de le laisser choisir.",
      "Plus de securite, c'est toujours moins de coupon. Savoir chiffrer le prix de la "
      "securite.",
      demo_securite),
    Q(CLIENT,
      "Expliquez un autocall a un client particulier en trente secondes.",
      "« Vous investissez pour trois ans au plus sur l'Euro Stoxx 50. Chaque trimestre, si "
      "l'indice est au-dessus de son niveau de depart, vous etes rembourse avec un coupon "
      "et le produit s'arrete. S'il est entre 70 % et 100 %, vous touchez le coupon et on "
      "continue. A l'echeance, vous recuperez votre capital tant que l'indice n'a pas perdu "
      "plus de 40 %. En dessous, vous subissez toute la baisse, comme si vous aviez achete "
      "l'indice. » Pas de jargon, trois scenarios, et le risque dit clairement.",
      "Clarte et honnetete sur le scenario de perte. Les mots trigger, Phoenix ou memoire "
      "ne servent a rien devant un particulier."),
    Q(CLIENT,
      "Le client compare le coupon de 8 % d'un autocall a une obligation a 4 %. "
      "Qu'est-ce qui cloche dans la comparaison ?",
      "Deux choses. Le risque d'abord : l'obligation rembourse le nominal sauf defaut, "
      "l'autocall expose le capital sous la barriere. La duree ensuite : le rappel anticipe "
      "coupe les coupons. Si le produit est rappele des la premiere ou la deuxieme "
      "observation, le scenario le plus frequent, le client touche 8 % par an pendant "
      "quelques mois, puis doit replacer son argent, peut-etre dans un marche moins "
      "favorable. Il faut raisonner en duree de vie attendue et en probabilite de perte, "
      "pas en coupon affiche.",
      "Le coupon d'un autocall n'est pas un rendement annuel garanti : duree incertaine et "
      "capital a risque.",
      demo_autocall),
    Q(CLIENT,
      "Un client veut profiter de la hausse du marche sans risquer son capital. Que "
      "proposez-vous ?",
      "Un capital garanti : une obligation zero-coupon qui assure 100 % a l'echeance, et un "
      "call qui donne la participation a la hausse. Je cadre tout de suite ce qui fait la "
      "participation : la maturite, le niveau de protection et un eventuel plafond. Et je "
      "precise que la garantie est celle de l'emetteur : si la banque fait defaut, la "
      "garantie tombe avec elle. Avec des taux nettement positifs, ce produit redevient "
      "attractif, ce qui n'etait pas le cas quand les taux etaient nuls.",
      "Proposer le produit qui colle a la vue, puis nommer le risque emetteur sans qu'on "
      "le demande."),
    Q(CLIENT,
      "Le client trouve trop faible la participation de son capital garanti. Quels leviers "
      "activez-vous ?",
      "Je rappelle que la participation n'est pas fixee par la banque : c'est le budget "
      "restant divise par le prix du call. Puis je propose les leviers un par un, avec leur "
      "prix. Allonger la maturite : on met moins de cote pour garantir le capital. "
      "Garantir 90 % au lieu de 100 %. Plafonner la hausse, ce qui revient a revendre un "
      "call plus haut. Choisir un sous-jacent moins volatil ou a plus fort dividende, dont "
      "le call coute moins cher. Chaque levier echange un peu de securite ou de potentiel "
      "contre de la participation, et c'est au client d'arbitrer.",
      "La participation est un resultat, pas un choix. Connaitre les leviers et leur sens.",
      demo_leviers_garanti),
    Q(CLIENT,
      "Un institutionnel vous dit que votre coupon est 50 bp sous celui d'un concurrent, "
      "sur le meme BRC. Que faites-vous ?",
      "Je verifie d'abord que les structures sont vraiment identiques : observation de la "
      "barriere, strike, dates, livraison physique ou cash, sous-jacent. Si oui, l'ecart "
      "vient en general de trois sources : le spread de credit de l'emetteur, la marge, et "
      "l'appetit du livre du trader pour ce risque a ce moment-la. Je le dis franchement : "
      "un coupon plus haut chez un emetteur moins bien note remunere un risque de credit. "
      "Puis je retourne voir le desk pour savoir si on peut s'aligner, sans rien promettre "
      "avant.",
      "Ne pas paniquer sur le prix : comparer a structure identique, expliquer l'ecart, "
      "puis negocier en interne."),
    Q(CLIENT,
      "Le marche vient de perdre 20 % et la volatilite a double. Votre client a peur. Que "
      "lui proposez-vous ?",
      "C'est souvent le meilleur moment pour vendre de la volatilite, a condition de le "
      "faire prudemment. Une vol elevee rend les puts chers : a barriere egale, le coupon "
      "d'un BRC est bien plus haut qu'avant la baisse. Et comme le niveau de depart est "
      "deja 20 % plus bas, une barriere a 60 % correspond a une baisse de 52 % depuis les "
      "plus hauts. Je propose donc une barriere plus profonde et une maturite courte, la "
      "vol courte etant la plus elevee en periode de stress. Mais je ne force pas un client "
      "qui a peur : je chiffre, il decide.",
      "Transformer la peur du marche en fenetre commerciale, sans pousser le client.",
      demo_stress),
    Q(CLIENT,
      "Un client est legerement haussier sur deux ans et ne veut pas renoncer a la hausse. "
      "BRC ou Bonus Certificate ?",
      "Le Bonus. Il garde toute la hausse et assure un remboursement minimum, le bonus, "
      "tant que la barriere tient. Le BRC plafonne le gain au coupon : si le marche monte "
      "de 30 %, le client du BRC touche son coupon et rien de plus, ce qui contredit sa "
      "vue. Le BRC convient mieux a un client qui pense que le marche va stagner, car il "
      "monetise directement l'absence de mouvement. Le travail du Sales, c'est d'abord de "
      "reformuler la vue du client, puis de choisir le payoff qui la paie.",
      "Partir de la vue du client, pas du produit qu'on a envie de vendre.",
      demo_bonus),
    Q(CLIENT,
      "Un client vous demande : « Quel est le pire scenario ? » Que repondez-vous ?",
      "Je reponds directement, avec un chiffre. Sur un BRC barriere 70 % : le sous-jacent "
      "touche la barriere et finit a -50 % ; le client recupere 50 % de son capital plus "
      "le coupon. Et independamment du marche, si l'emetteur fait defaut, il ne recupere "
      "que le taux de recouvrement. Je n'esquive jamais cette question : un client qui a "
      "entendu le pire scenario et achete quand meme est un client qui reste.",
      "Transparence sur la perte maximale, chiffree. C'est ce qui construit la confiance.",
      demo_pire),
    Q(CLIENT,
      "Le client vous demande combien la banque gagne sur ce produit.",
      "Je reponds franchement : les couts figurent de toute facon dans le document "
      "d'information (KID PRIIPs) et doivent etre detailles au titre de MiFID II. La marge "
      "est prise sur le budget option, souvent de l'ordre de 0,5 a 1,5 % par an selon le "
      "produit et le client. Elle paie la structuration, le capital immobilise et les "
      "risques que la banque ne couvre pas parfaitement : saut de delta a la barriere, "
      "skew, correlation. Un institutionnel la negocie ; l'important est qu'il sache ce "
      "qu'il paie, et pour quoi.",
      "Pas de tabou sur la marge : la justifier par les risques que la banque garde.",
      demo_marge),
    Q(CLIENT,
      "Comment presentez-vous le meme produit a un institutionnel et a un client prive ?",
      "A un institutionnel, je parle prix et risques de marche : quelle vol implicite et "
      "quel skew sont dans le prix, quel spread emetteur, comment le produit se comporte "
      "en mark-to-market, quelle liquidite sur le secondaire. Il compare avec ce qu'il "
      "pourrait repliquer lui-meme avec des options listees. A un client prive ou a son "
      "banquier, je parle scenarios : que se passe-t-il si le marche monte, stagne, baisse "
      "de 20 %, baisse de 50 %. Le produit est le meme, le langage change.",
      "Adapter le discours a l'interlocuteur : grecques pour l'un, scenarios pour l'autre."),
    Q(CLIENT,
      "Votre client a achete un BRC il y a six mois. Le sous-jacent est a 75 %, la "
      "barriere a 70 %. Que lui dites-vous ?",
      "Je l'appelle avant qu'il m'appelle. Je lui donne les faits : la barriere n'est pas "
      "touchee et le coupon reste du quoi qu'il arrive, mais le produit vaut nettement "
      "moins sur le marche secondaire, car le put qu'il a vendu est proche de s'activer. "
      "Puis je lui presente ses options : conserver le produit en acceptant le risque "
      "d'etre rembourse en actions, ou sortir maintenant au prix du secondaire. Je ne lui "
      "promets pas que ca va remonter. La relation se construit sur ces appels-la.",
      "Proactivite et honnetete dans la mauvaise passe : c'est la que se gagne la "
      "fidelite.",
      demo_mtm),
    Q(CLIENT,
      "Un client veut un placement sans risque qui rapporte 7 %. Que lui repondez-vous ?",
      "Que ca n'existe pas, et je l'explique en une phrase : tout ce qui depasse le taux "
      "sans risque remunere un risque. Puis je retourne la question : quel risque etes-vous "
      "pret a prendre ? Une baisse du marche au-dela d'un seuil, un gain plafonne, le "
      "credit d'un emetteur, une duree incertaine. A partir de sa reponse, je construis le "
      "produit. Refuser une promesse impossible, c'est ce qui rend credible tout le reste.",
      "Pas de rendement sans risque. Le Sales transforme une demande irrealiste en choix "
      "de risque."),
    Q(CLIENT,
      "Un client veut le coupon d'un worst-of sur trois actions mais ne mesure pas le "
      "risque. Comment le lui faire comprendre ?",
      "Une phrase : « Vous n'achetez pas trois actions, vous achetez la pire des trois. » "
      "Il suffit qu'une seule chute pour que le produit se comporte comme elle, meme si les "
      "deux autres montent. Puis je chiffre : coupon et probabilite de perte du worst-of "
      "face au meme produit sur une seule action. Le coupon plus eleve s'accompagne d'une "
      "probabilite de perte plus elevee, et c'est a lui de juger si l'echange lui convient.",
      "Faire comprendre le worst-of par une image simple, puis par les chiffres.",
      demo_worst_of),

    # ---------------- Mecanique des produits ------------------------
    Q(MECANIQUE,
      "Decomposez un Barrier Reverse Convertible.",
      "Une obligation emise par la banque, plus un put down-and-in que le client vend a la "
      "banque. Le client prete son argent et assure la banque contre une forte baisse du "
      "sous-jacent. Son coupon vient des interets de l'obligation et de la prime du put, "
      "plus le spread de credit si l'emetteur en paie un, moins la marge. Si la barriere "
      "n'est jamais touchee, il recupere 100 % plus le coupon. Si elle est touchee et que "
      "le sous-jacent finit sous le strike, il est rembourse en actions et encaisse toute "
      "la baisse depuis le strike.",
      "Savoir dire qui vend l'option a qui : le client est vendeur du put.",
      demo_decomposition),
    Q(MECANIQUE,
      "Sur un Autocall Phoenix, quelle difference entre le trigger de rappel, la barriere "
      "de coupon et la barriere de capital ?",
      "Trois niveaux, trois roles. Le trigger, souvent a 100 %, est observe a chaque date : "
      "au-dessus, le produit s'arrete et rembourse le nominal avec le coupon. La barriere "
      "de coupon, par exemple 70 %, decide si le coupon de la periode est paye. La barriere "
      "de capital, par exemple 60 %, n'est regardee qu'a l'echeance : en dessous, le client "
      "encaisse toute la baisse depuis 100 %, pas seulement la partie sous la barriere.",
      "Le piege de la barriere de capital : sous 60 %, la perte se compte depuis 100 %."),
    Q(MECANIQUE,
      "Comment est construit un capital garanti, et d'ou vient la participation ?",
      "Une obligation zero-coupon qui reconstitue 100 % a l'echeance, plus un call achete "
      "avec ce qui reste. A 3 % sur 5 ans, garantir 1 000 EUR coute environ 861 EUR "
      "aujourd'hui ; les 139 restants, moins la marge, achetent le call. La participation "
      "est donc un resultat : le budget option divise par le prix du call. La banque ne la "
      "choisit pas.",
      "La participation est un reste de budget.",
      demo_capital),
    Q(MECANIQUE,
      "Que se passe-t-il sur un Bonus Certificate si la barriere est touchee ?",
      "Le bonus disparait definitivement. Le produit, c'est le sous-jacent sans ses "
      "dividendes plus un put down-and-out ; quand la barriere casse, le put s'eteint et le "
      "client se retrouve avec un simple tracker. Il ne perd pas plus que s'il detenait le "
      "sous-jacent, mais il perd sa protection. C'est une protection conditionnelle, pas "
      "une garantie. Sa force face au BRC : la hausse n'est pas plafonnee.",
      "Protection conditionnelle n'est pas garantie : le mot a ne jamais employer."),
    Q(MECANIQUE,
      "Qu'est-ce que l'effet memoire sur un Phoenix, et combien coute-t-il ?",
      "Un coupon non verse parce que le sous-jacent etait sous la barriere de coupon est "
      "rattrape a la premiere date ou il repasse au-dessus. Le client ne perd donc ses "
      "coupons que si le sous-jacent reste durablement bas. C'est une option supplementaire "
      "offerte au client : elle a un prix, et la banque le paie en baissant le coupon par "
      "periode.",
      "Tout ce qui protege le client se paie en coupon.",
      demo_memoire),
    Q(MECANIQUE,
      "Sur un BRC, barriere observee en continu ou a l'echeance : qu'est-ce que ca change ?",
      "En continu, un seul passage sous la barriere a n'importe quel moment suffit a armer "
      "le risque, meme une meche en seance. A l'echeance, seul le cours final compte. "
      "L'observation a l'echeance protege mieux le client, donc le put qu'il vend vaut "
      "moins et le coupon baisse. Deux BRC « barriere 70 % » peuvent ainsi avoir des "
      "risques tres differents : c'est une ligne du termsheet a toujours lire.",
      "Le mode d'observation fait partie du prix du risque.",
      demo_observation),
    Q(MECANIQUE,
      "Comment un Twin-Win permet-il de gagner a la baisse ?",
      "Le produit, c'est un tracker plus l'achat d'un put down-and-out a la monnaie. Entre "
      "la barriere et le niveau initial, le put paie : la baisse se transforme en gain. Le "
      "financement vient des dividendes abandonnes. Le danger est la discontinuite : avec "
      "une barriere a 65 % et une participation de 100 % a la baisse, le client gagne pres "
      "de 35 % juste au-dessus de la barriere, et perd 35 % juste en dessous.",
      "Savoir expliquer la discontinuite a la barriere : le profil le plus dangereux a mal "
      "vendre."),

    # ---------------- Parametres de marche --------------------------
    Q(MARCHE,
      "Pourquoi le coupon d'un BRC monte-t-il quand la volatilite monte ?",
      "Parce que le client est vendeur d'un put. Plus la vol est haute, plus le sous-jacent "
      "a de chances de traverser la barriere, plus ce put vaut cher. La banque paie plus "
      "cher l'assurance que le client lui vend, et ca se voit dans le coupon. Corollaire "
      "commercial : les meilleurs moments pour placer des BRC sont les pics de vol, quand "
      "les clients ont le plus peur.",
      "Relier vol, prix du put et coupon, et en tirer la fenetre commerciale.",
      demo_vol_brc),
    Q(MARCHE,
      "Pourquoi les capitaux garantis avaient-ils presque disparu entre 2015 et 2021 ?",
      "Parce qu'ils se financent avec les taux. La banque place juste ce qu'il faut pour "
      "rembourser 100 % a l'echeance, et le reste achete le call. A 3 % sur 5 ans, il reste "
      "environ 14 % du nominal ; a 0,5 %, il en reste 2,5 %, avant marge. Avec des taux "
      "nuls ou negatifs, la participation devenait derisoire et le marche a bascule vers "
      "les BRC et les autocalls. La remontee des taux depuis 2022 les a fait revenir.",
      "Taux hauts, capital garanti vendable ; taux nuls, il disparait.",
      demo_taux_garanti),
    Q(MARCHE,
      "Pourquoi un autocall worst-of sur trois actions paie-t-il plus qu'un autocall sur "
      "une seule ?",
      "Parce que le client vend de la correlation. Il faut que les trois actions tiennent : "
      "il suffit qu'une seule chute pour que le produit se comporte comme elle. Moins les "
      "actions sont correlees, plus il y a de chances qu'il y ait un trainard, plus le "
      "risque que le client vend vaut cher, plus le coupon monte. Le coupon plus eleve "
      "n'est pas un cadeau : la probabilite de perte monte avec lui.",
      "Dire « le client est short correlation » et le traduire en probabilite de perte.",
      demo_worst_of),
    Q(MARCHE,
      "Quel est l'effet du skew sur le coupon d'un BRC ?",
      "Le skew rend les puts bas plus chers, en vol, que les options a la monnaie. Or le "
      "risque du put que vend le client d'un BRC se concentre autour de la barriere, loin "
      "sous le spot. Avec du skew, ce put est price a une vol plus haute que la vol ATM : "
      "le coupon monte. Le skew a deux causes : les marches baissent plus brutalement "
      "qu'ils ne montent, et les institutionnels achetent massivement des puts pour se "
      "proteger.",
      "Relier skew, vol au niveau de la barriere, prix du put et coupon.",
      demo_skew),
    Q(MARCHE,
      "Un dividende eleve sur le sous-jacent, c'est bon ou mauvais pour un capital "
      "garanti ?",
      "Bon pour la participation. Le client ne touche pas les dividendes : ils abaissent le "
      "prix a terme du sous-jacent, donc le prix du call que la banque doit acheter. Budget "
      "inchange, call moins cher : la participation monte. Le revers : sur un tracker, "
      "c'est exactement ce que le client abandonne, environ 7,8 % sur 3 ans avec 2,5 % de "
      "dividende.",
      "Les dividendes abandonnes financent la structure.",
      demo_dividende),
    Q(MARCHE,
      "Vol implicite et vol realisee : pourquoi l'ecart compte-t-il pour un produit "
      "structure ?",
      "Sur les indices, la vol implicite est en moyenne 2 a 4 points au-dessus de la vol "
      "qui se realise ensuite : c'est la prime de risque de variance. Le client qui vend de "
      "la vol a travers un BRC ou un autocall capte cette prime. C'est la justification "
      "economique de ces produits : il est paye pour porter le risque de krach, gagne la "
      "plupart des annees et perd beaucoup lors de la mauvaise. Le presenter ainsi est plus "
      "juste que de parler d'un coupon eleve.",
      "Savoir dire pourquoi ces produits existent : la variance risk premium."),
    Q(MARCHE,
      "Que fait une hausse des taux au coupon d'un BRC ?",
      "Elle l'augmente, mais moins que point pour point. Le coupon, c'est d'abord le taux "
      "de l'obligation, puis la prime du put. Quand les taux montent de 2 points, la "
      "composante obligataire monte d'environ 2 points ; mais le prix a terme du "
      "sous-jacent monte aussi, ce qui rend le put moins cher et freine la hausse. C'est "
      "pour cela qu'a vol egale, les BRC affichaient des coupons bien plus eleves en 2023 "
      "qu'en 2020.",
      "Separer la composante taux de la composante option.",
      demo_taux_brc),

    # ---------------- Risque emetteur -------------------------------
    Q(EMETTEUR,
      "Pourquoi un spread de credit plus large augmente-t-il le coupon ?",
      "Un produit structure est une dette de la banque. Si l'emetteur paie 150 bp "
      "au-dessus du taux sans risque pour se financer sur le marche obligataire, il peut "
      "reverser ce financement plus cher dans le produit. Le client est donc remunere pour "
      "un risque de credit en plus du risque de marche. A structure identique, un coupon "
      "superieur a la concurrence doit faire poser la question : l'emetteur est-il moins "
      "bien note ?",
      "Coupon plus haut peut vouloir dire signature plus faible.",
      demo_spread),
    Q(EMETTEUR,
      "Que recupere le client si l'emetteur fait defaut, meme si la barriere n'a jamais "
      "ete touchee ?",
      "Seulement le taux de recouvrement de la dette senior de la banque, 40 % par "
      "convention, parfois bien moins. Le client n'est pas proprietaire des actions, il est "
      "creancier de l'emetteur. En 2008, des porteurs de produits Lehman, y compris a "
      "capital garanti, n'ont recupere qu'une petite fraction de leur mise, quel que soit "
      "le comportement de leur sous-jacent. « Capital garanti » veut dire garanti par "
      "l'emetteur, pas sans risque.",
      "Un produit structure est une dette de la banque, pas une detention d'actions."),
    Q(EMETTEUR,
      "Comment estimer le risque de defaut d'un emetteur a partir de son spread ?",
      "Regle simple : intensite de defaut egale spread divise par un moins le taux de "
      "recouvrement. A 150 bp avec 40 % de recouvrement, cela donne 2,5 % par an, soit "
      "environ 7 % de probabilite de defaut sur 3 ans. La perte attendue, probabilite fois "
      "perte en cas de defaut, retombe a peu pres sur le spread : c'est ce que le marche "
      "fait payer. Ces probabilites sont risque-neutres, donc plus elevees que les defauts "
      "historiques, car elles incluent une prime de risque.",
      "Savoir passer d'un spread a une probabilite de defaut, et nuancer.",
      demo_defaut),
    Q(EMETTEUR,
      "Deux BRC identiques, deux emetteurs, 60 bp d'ecart de coupon. Comment conseillez-vous "
      "un institutionnel ?",
      "A structure identique, l'ecart vient surtout de la signature : l'emetteur qui paie "
      "plus se finance plus cher, le marche le juge plus risque. Je compare notations et "
      "CDS, et je regarde l'exposition que le client a deja sur chaque banque : un "
      "institutionnel ne veut pas concentrer son risque de contrepartie. On peut repartir "
      "les tickets entre plusieurs emetteurs, ou, si le risque de credit le gene vraiment, "
      "proposer une version collateralisee en echange d'un coupon plus faible.",
      "Le coupon superieur se paie en risque de credit. Penser diversification des "
      "contreparties."),
    Q(EMETTEUR,
      "Qu'est-ce que le wrong-way risk, et pourquoi compte-t-il sur un produit structure ?",
      "C'est quand le risque de defaut de la contrepartie augmente precisement quand "
      "l'exposition est la plus forte. Une banque a plus de chances de faire defaut en "
      "plein krach, au moment ou le sous-jacent d'un BRC s'effondre. Le client cumule "
      "alors la perte de marche et la perte de credit. Les modeles simples, dont celui de "
      "cette appli, supposent le defaut independant du marche : ils sous-estiment ce "
      "risque.",
      "Comprendre que credit et marche sont correles au pire moment."),

    # ---------------- Grecques et couverture ------------------------
    Q(GRECQUES,
      "Quelle est la position en vega d'un client qui achete un BRC ?",
      "Short vega : il a vendu un put. Si la vol implicite monte apres l'achat, le put vaut "
      "plus cher et la valeur de marche de son BRC baisse, meme si le sous-jacent n'a pas "
      "bouge. En face, la banque est long vega et revend de la vol sur le marche pour se "
      "couvrir. C'est un phenomene connu : les desks qui ont vendu beaucoup d'autocalls se "
      "retrouvent longs de vol longue et la revendent, ce qui pese sur ces vols.",
      "Le client d'un BRC ou d'un autocall est vendeur de volatilite.",
      demo_vega),
    Q(GRECQUES,
      "Pourquoi le gamma explose-t-il pres de la barriere ?",
      "Juste au-dessus de la barriere, a l'approche de l'echeance, la valeur du put "
      "down-and-in change enormement pour un petit mouvement : un point de plus ou de moins "
      "decide si le client perd 30 % ou rien. Le delta du trader devient tres grand, il "
      "peut detenir plusieurs fois le nominal en actions. Si la barriere est touchee, le "
      "put devient un put ordinaire et le delta retombe d'un coup : il doit revendre une "
      "grosse partie de sa couverture, dans un marche qui baisse. Ce risque se paie dans le "
      "prix : les barrieres proches du spot sont moins bien traitees.",
      "Lier gamma, barriere et cout de couverture, et comprendre pourquoi le desk degrade "
      "le prix.",
      demo_gamma),
    Q(GRECQUES,
      "Le client d'un BRC est-il long ou short theta ?",
      "Long theta, comme tout vendeur d'option. Chaque jour qui passe sans que la barriere "
      "soit touchee, le put qu'il a vendu perd de la valeur, et c'est lui qui l'encaisse. "
      "La contrepartie : il est short gamma, il souffre des grands mouvements. Theta et "
      "gamma sont les deux faces de la meme position.",
      "Vendeur d'option : long theta, short gamma.",
      demo_theta),
    Q(GRECQUES,
      "Comment la banque couvre-t-elle un BRC qu'elle vient de vendre ?",
      "La banque est acheteuse du put down-and-in. Un put perd de la valeur quand le "
      "sous-jacent monte : pour neutraliser ce risque, le trader achete des actions, en "
      "quantite egale au delta, et ajuste chaque jour. Quand le marche baisse, le delta du "
      "put s'alourdit et il en rachete ; quand il remonte, il revend. Le vega et le skew se "
      "couvrent en revendant de la vol sur le marche d'options. Ce qui reste, le saut de "
      "delta a la barriere ou la correlation d'un worst-of, est garde dans le livre et "
      "paye par la marge.",
      "Le Sales ne porte pas le risque, mais doit savoir comment le trader le couvre.",
      demo_couverture),
    Q(GRECQUES,
      "Pourquoi la couverture en delta coute-t-elle de l'argent, alors qu'en theorie elle "
      "est parfaite ?",
      "Black-Scholes suppose un rebalancement continu et gratuit : dans ce monde, la "
      "couverture replique exactement l'option. En pratique, on rebalance a intervalles "
      "discrets, ce qui laisse une erreur de couverture, et chaque transaction paie la "
      "fourchette achat-vente. Rebalancer plus souvent reduit l'erreur mais augmente les "
      "frais : c'est un arbitrage. Ce cout fait partie de ce que la marge doit couvrir, "
      "elle n'est pas un pur profit.",
      "La marge paie aussi un cout de couverture reel."),
    Q(GRECQUES,
      "Comment calcule-t-on une grecque sur un produit price par Monte Carlo ?",
      "Par bump and revalue : on decale un parametre de peu, on reprice, et on divise "
      "l'ecart par le choc. Le detail qui compte : reutiliser exactement les memes tirages "
      "aleatoires pour les deux prix. Sinon le bruit de simulation, souvent plus grand que "
      "l'effet cherche, domine la difference et la grecque est inutilisable. Pour un "
      "gamma, c'est encore pire, d'ou l'interet des formules fermees quand elles existent, "
      "comme pour le put down-and-in.",
      "Connaitre la technique standard et son piege numerique."),
]


# =====================================================================
# PAGE
# =====================================================================

def _vues():
    return st.session_state.setdefault("ent_vues", set())


def _revele():
    st.session_state["ent_revele"] = st.session_state["ent_q"]
    _vues().add(st.session_state["ent_q"])


def _suivante(liste):
    i = st.session_state.get("ent_q")
    st.session_state["ent_q"] = liste[(liste.index(i) + 1) % len(liste)] if i in liste else liste[0]


def _hasard(liste):
    """Une question du theme jamais vue si possible, toujours differente de l'actuelle."""
    i = st.session_state.get("ent_q")
    autres = [j for j in liste if j != i] or liste
    st.session_state["ent_q"] = random.choice([j for j in autres if j not in _vues()] or autres)


def _zero():
    st.session_state["ent_vues"] = set()
    st.session_state["ent_revele"] = None


def affiche(m):
    st.title("Entrainement", help="**Ce que c'est.** Une banque de questions d'entretien pour un poste de Sales en\nproduits structures, classees en cinq themes. Le plus fourni est le discours client :\nc'est ce qu'un desk teste en premier chez un Sales.\n\n**Comment s'en servir.** Lis la question, reponds a voix haute en trente secondes, puis\naffiche la reponse. Elle est redigee comme on la dirait en entretien : courte, juste, avec\nle pourquoi financier. La ligne « Ce que le recruteur verifie » donne l'idee a ne pas\nrater.\n\n**Les demonstrations chiffrees.** Quand la reponse repose sur un mecanisme de prix, le\nmoteur le calcule avec la vue de marche de la barre laterale. Change la vol, les taux, le\nspread ou le modele : les chiffres suivent, et tu peux citer des ordres de grandeur justes.\n\n**Le compteur.** Une question compte comme vue des que sa reponse est affichee.\nLe bouton « Question au hasard » privilegie les questions pas encore vues.")
    st.caption("Reponds a voix haute avant de regarder : c'est la formulation qui se "
               "travaille, pas seulement l'idee.")

    c1, c2 = st.columns([1, 3])
    theme = c1.selectbox(
        "Theme", [TOUS] + THEMES, key="ent_theme", bind="query-params",
        help="Filtre la banque de questions. Le discours client est le theme prioritaire : "
             "savoir adapter un produit au besoin du client et l'expliquer simplement.")
    liste = [i for i, x in enumerate(QUESTIONS) if theme == TOUS or x["theme"] == theme]
    if st.session_state.get("ent_q") not in liste:
        st.session_state["ent_q"] = liste[0]
    q = c2.selectbox(
        "Question", liste, key="ent_q", bind="query-params", format_func=lambda i: f"{i+1}. {QUESTIONS[i]['q']}",
        help="Choisis une question precise, ou laisse les boutons en dessous te guider. "
             "Le numero est celui de la question dans la banque complete.")

    b = st.columns(4)
    b[0].button("Voir la reponse", key="ent_voir", type="primary", on_click=_revele,
                help="Affiche la reponse redigee et, si la question s'y prete, une "
                     "demonstration chiffree calculee avec la vue de marche actuelle.")
    b[1].button("Question suivante", key="ent_suivante", on_click=_suivante, args=(liste,),
                help="Passe a la question suivante du theme choisi, dans l'ordre de la "
                     "banque.")
    b[2].button("Question au hasard", key="ent_hasard", on_click=_hasard, args=(liste,),
                help="Tire une question du theme choisi, en priorite parmi celles dont tu "
                     "n'as pas encore vu la reponse. Le plus proche d'un vrai entretien.")
    b[3].button("Remettre le compteur a zero", key="ent_zero", on_click=_zero,
                help="Oublie les questions vues pendant cette session.")

    x = QUESTIONS[q]
    st.markdown(f"### {x['q']}")
    st.caption(f"{x['theme']} · question {liste.index(q) + 1} sur {len(liste)}"
               + (" · avec demonstration chiffree" if x["demo"] else ""))

    if st.session_state.get("ent_revele") == q:
        with st.container(border=True):
            st.markdown(x["r"])
            st.caption("Ce que le recruteur verifie : " + x["cle"])
        if x["demo"]:
            st.subheader("Demonstration chiffree",
                         help="Calculee par le moteur de l'appli avec la vue de marche de "
                              "la barre laterale. Les parametres du produit sont fixes et "
                              "rappeles sous ce titre.")
            m.bandeau_modele()
            x["demo"](m)
    else:
        st.info("Prends trente secondes, reponds a voix haute, puis clique sur « Voir la "
                "reponse ».")

    vues = _vues()
    st.markdown("---")
    k = st.columns(2)
    k[0].metric("Questions vues", f"{len(vues)} / {len(QUESTIONS)}",
                help="Une question compte comme vue des que sa reponse a ete affichee. Le "
                     "compteur dure le temps de la session : il repart a zero si tu "
                     "recharges la page.")
    k[1].metric("Vues dans ce theme", f"{len(vues & set(liste))} / {len(liste)}",
                help="Progression dans le theme selectionne.")
