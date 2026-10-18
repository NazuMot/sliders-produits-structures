"""Entrainement : questions d'entretien de Sales en produits structures, et jeu « Devine le coupon »."""

import math
import random

import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0
from communs import MAX_CACHE, sim_bs, sim_bates, sim_panier, sim_defaut


CLIENT = "Discours client"
MECANIQUE = "Mécanique des produits"
MARCHE = "Paramètres de marché"
EMETTEUR = "Risque émetteur"
GRECQUES = "Grecques et couverture"
THEMES = [CLIENT, MECANIQUE, MARCHE, EMETTEUR, GRECQUES]
TOUS = "Tous les thèmes"


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
    st.caption(f"Formule fermée de Reiner-Rubinstein, vol {sigma*100:.0f}% : "
               + ("la vol implicite à 70 % du smile généré par Bates, la formule restant "
                  "en Black-Scholes." if m.BATES else "la vol au niveau de la barrière."))


# =====================================================================
# DEMONSTRATIONS CHIFFREES
# =====================================================================

def demo_bonne_affaire(m):
    st.caption("BRC 1 an, barrière 70 % observée en continu, strike 100 %.")
    ch = _chemins(m, 1.0, 0.70)
    cpn = _brc(m, ch)
    toucher, perte = _risques_brc(ch)
    _ligne(("Coupon du BRC", f"{cpn*100:,.2f}% p.a."),
           ("Taux sans risque", f"{m.r*100:,.2f}%"),
           ("Proba de toucher la barrière", _pct(toucher)),
           ("Proba de perte en capital", _pct(perte)))
    st.caption(f"Les {(cpn - m.r)*100:,.1f} points au-dessus du taux sans risque "
               "rémunèrent la protection que le client vend.")


def demo_securite(m):
    st.caption("BRC 1 an, barrière 70 % continue, strike 100 % : trois façons de "
               "rendre le produit plus sur.")
    variantes = [("BRC de départ", _chemins(m, 1.0, 0.70), 1.0, 0.70, True),
                 ("Barrière à 60 %", _chemins(m, 1.0, 0.60), 1.0, 0.60, True),
                 ("Barrière observée à l'échéance", _chemins(m, 1.0, 0.70), 1.0, 0.70, False),
                 ("Maturité 6 mois", _chemins(m, 0.5, 0.70), 0.5, 0.70, True)]
    lignes = []
    for nom, ch, T, bar, continu in variantes:
        cpn = _brc(m, ch, T=T, bar=bar, continu=continu)
        toucher, perte = _risques_brc(ch, T, bar, continu)
        lignes.append({"Variante": nom, "Coupon annuel": f"{cpn*100:,.2f}%",
                       "Coupon total": f"{cpn*T*100:,.2f}%",
                       "Proba de perte": _pct(perte), "_c": cpn, "_p": perte})
    a, b = lignes[0], lignes[1]
    _ligne(("Coupon, barrière 70 %", a["Coupon annuel"]),
           ("Coupon, barrière 60 %", b["Coupon annuel"], _pts(b["_c"] - a["_c"])),
           ("Proba de perte, 70 %", a["Proba de perte"]),
           ("Proba de perte, 60 %", b["Proba de perte"], _pts(b["_p"] - a["_p"])))
    st.table([{k: v for k, v in x.items() if not k.startswith("_")} for x in lignes],
             hide_index=True)


def demo_autocall(m):
    st.caption("Autocall Phoenix 3 ans, observations trimestrielles, trigger 100 %, "
               "barrière de coupon 70 %, barrière de capital 60 %, effet mémoire.")
    ch = _chemins(m, AC_T, 0.60)
    cpn, duree, _, perte = _autocall(m, ch)
    i1 = M.idx_obs(ch[0].shape[1] - 1, AC_OBS)[0]
    rappel_1 = (ch[0][:, i1] >= S0).mean()
    _ligne(("Coupon affiché", f"{cpn*100:,.2f}% p.a."),
           ("Durée de vie moyenne", f"{duree:,.2f} ans sur {AC_T:.0f}"),
           ("Rappelé dès le 1er trimestre", _pct(rappel_1)),
           ("Proba de perte en capital", _pct(perte)))


def demo_leviers_garanti(m):
    st.caption("Capital garanti de départ : 5 ans, protection 100 %, sans plafond.")
    p0 = _participation(m)[0]
    leviers = [("8 ans au lieu de 5", _participation(m, T=8.0)[0]),
               ("Protection 90 %", _participation(m, protection=0.90)[0]),
               ("Hausse plafonnée à +50 %", _participation(m, cap=1.5)[0])]
    _ligne(("Participation de départ", _pct(p0, 0)),
           *[(nom, _pct(p, 0), _pts(p - p0)) for nom, p in leviers])
    if p0 <= 0:
        st.caption("Avec ces taux et cette marge, le produit de départ ne peut pas être "
                   "construit : la garantie et la marge consomment tout le nominal.")


def demo_stress(m):
    choc = min(2 * m.vol_atm, 1.0)
    st.caption(f"BRC 1 an, strike 100 % du nouveau niveau. La vol passe de "
               f"{m.vol_atm*100:.0f}% à {choc*100:.0f}%"
               + (" (sous Bates : vol de départ seulement, la vol long terme ne bouge "
                  "pas, comme dans un vrai pic de stress)." if m.BATES else "."))
    avant = _chemins(m, 1.0, 0.70)
    apres70, apres60 = _chemins(m, 1.0, 0.70, vol=choc), _chemins(m, 1.0, 0.60, vol=choc)
    c0, c1, c2 = _brc(m, avant), _brc(m, apres70), _brc(m, apres60, bar=0.60)
    p0, p1, p2 = (_risques_brc(avant)[1], _risques_brc(apres70)[1],
                  _risques_brc(apres60, bar=0.60)[1])
    _ligne(("Avant, barrière 70 %", f"{c0*100:,.2f}% p.a."),
           ("Après, barrière 70 %", f"{c1*100:,.2f}% p.a.", _pts(c1 - c0)),
           ("Après, barrière 60 %", f"{c2*100:,.2f}% p.a.", _pts(c2 - c0)))
    _ligne(("Proba de perte avant", _pct(p0)),
           ("Après, barrière 70 %", _pct(p1)),
           ("Après, barrière 60 %", _pct(p2)))
    st.caption("Après une baisse de 20 %, une barrière à 60 % du nouveau niveau se situe "
               "à 48 % des plus hauts. Les probabilités sont calculées avec la vol de "
               "stress : si la vol réalisée retombe ensuite, comme souvent après un pic, le "
               "risque réel est plus faible que celui que le prix paie.")


def demo_bonus(m):
    T, bar = 2.0, 0.70
    st.caption("Deux ans, barrière 70 % observée en continu. Sans risque émetteur, pour "
               "comparer les deux profils à armes égales.")
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
           ("Bonus : si le marché fait +30 %", _pct(max(1.3, lo))),
           ("BRC : coupon", f"{cpn*100:,.2f}% p.a."),
           ("BRC : si le marché fait +30 %", _pct(1 + cpn * T)))
    st.caption(f"Proba de toucher la barrière sur deux ans : {(1-p_no.mean())*100:,.1f}%. "
               "Dans ce cas, le bonus disparaît et le BRC expose à la baisse.")


def demo_pire(m):
    st.caption("BRC 1 an, barrière 70 % continue, strike 100 %. Remboursement total, "
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
           ("Si l'émetteur fait défaut", _pct(m.recovery, 0)))


def demo_marge(m):
    st.caption("BRC 1 an barrière 70 %, et capital garanti 5 ans à 100 %.")
    ch = _chemins(m, 1.0, 0.70)
    c, c0 = _brc(m, ch), _brc(m, ch, marge=0.0)
    p, p0 = _participation(m)[0], _participation(m, marge=0.0)[0]
    _ligne(("Coupon BRC offert", f"{c*100:,.2f}% p.a."),
           ("BRC sans marge", f"{c0*100:,.2f}% p.a.", _pts(c0 - c)),
           ("Participation capital garanti", _pct(p, 0)),
           ("Capital garanti sans marge", _pct(p0, 0), _pts(p0 - p)))
    st.caption(f"Marge de {m.marge*100:,.1f}% par an, soit {m.nominal*m.marge:,.0f} EUR "
               f"par an sur {m.nominal:,.0f} EUR. Sur un produit de 5 ans, elle est "
               "prélevée cinq fois sur le même budget option.")


def demo_mtm(m):
    st.caption("BRC 1 an, barrière 70 % continue, strike 100 %. Six mois plus tard, le "
               "sous-jacent est à 75. Valeurs en % du nominal, coupon compris.")
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
    _ligne(("Valeur à l'émission", _pct(v0)),
           ("Valeur aujourd'hui", _pct(v1), _pts(v1 - v0)),
           ("Proba de toucher d'ici l'échéance", _pct(toucher)),
           ("Proba de perte en capital", _pct(perte)))
    st.caption("À l'émission, le produit vaut le nominal moins la marge : c'est ainsi que "
               "le coupon est calculé. Six mois plus tard, même modèle, même vue de marché.")


def demo_decomposition(m):
    st.caption("BRC 1 an, barrière 70 % continue, strike 100 %. D'où vient chaque "
               "point de coupon.")
    ch, T = _chemins(m, 1.0, 0.70), 1.0
    total_sans = _brc(m, ch, tau=None)
    total = _brc(m, ch)
    taux = (math.exp(m.r * T) - 1) / T
    marge = -m.marge * math.exp(m.r * T)
    put = total_sans - taux - marge
    parts = [("Taux", _pts(taux)), ("Prime du put vendu", _pts(put))]
    if m.spread > 0:
        parts.append((f"Spread émetteur ({m.spread} bp)", _pts(total - total_sans)))
    parts += [("Marge banque", _pts(marge)), ("Coupon offert", f"{total*100:,.2f}% p.a.")]
    _ligne(*parts)


def demo_capital(m):
    st.caption("Capital garanti 5 ans, protection 100 %, sans plafond. Montants pour "
               f"{m.nominal:,.0f} EUR.")
    p, zc, budget, cout = _participation(m)
    _ligne(("Mis de côté pour la garantie", f"{zc*m.nominal:,.0f} EUR"),
           ("Budget option, après marge", f"{budget*m.nominal:,.0f} EUR"),
           ("Prix du call à la monnaie", f"{cout*m.nominal:,.0f} EUR"),
           ("Participation", _pct(p, 0)))


def demo_memoire(m):
    st.caption("Autocall Phoenix 3 ans, trimestriel, trigger 100 %, barrière de coupon "
               "70 %, barrière de capital 60 %.")
    ch = _chemins(m, AC_T, 0.60)
    avec, sans = _autocall(m, ch, True)[0], _autocall(m, ch, False)[0]
    _ligne(("Coupon sans mémoire", f"{sans*100:,.2f}% p.a."),
           ("Coupon avec mémoire", f"{avec*100:,.2f}% p.a.", _pts(avec - sans)))


def demo_observation(m):
    st.caption("BRC 1 an, barrière 70 %, strike 100 %.")
    ch = _chemins(m, 1.0, 0.70)
    cc, ce = _brc(m, ch), _brc(m, ch, continu=False)
    pc, pe = _risques_brc(ch)[1], _risques_brc(ch, continu=False)[1]
    _ligne(("Coupon, observation continue", f"{cc*100:,.2f}% p.a."),
           ("Coupon, à l'échéance", f"{ce*100:,.2f}% p.a.", _pts(ce - cc)),
           ("Proba de perte, continue", _pct(pc)),
           ("Proba de perte, à l'échéance", _pct(pe), _pts(pe - pc)))


def demo_vol_brc(m):
    st.caption("BRC 1 an, barrière 70 % continue, strike 100 %. Vol choquée de +5 points"
               + (" (sous Bates : vol de départ et vol long terme)." if m.BATES else "."))
    base, choc = _chemins(m, 1.0, 0.70), _chemins(m, 1.0, 0.70, dvol=0.05)
    c0, c1 = _brc(m, base), _brc(m, choc)
    t0, t1 = _risques_brc(base)[0], _risques_brc(choc)[0]
    v = m.vol_atm * 100
    _ligne((f"Coupon, vol à {v:.0f}%", f"{c0*100:,.2f}% p.a."),
           (f"Coupon, vol à {v+5:.0f}%", f"{c1*100:,.2f}% p.a.", _pts(c1 - c0)),
           (f"Proba de toucher, vol à {v:.0f}%", _pct(t0)),
           (f"Proba de toucher, vol à {v+5:.0f}%", _pct(t1), _pts(t1 - t0)))


def demo_taux_garanti(m):
    r_alt = 0.005 if m.r > 0.015 else 0.03
    st.caption("Capital garanti 5 ans, protection 100 %, sans plafond. Montants pour "
               f"{m.nominal:,.0f} EUR.")
    p0, _, b0, _ = _participation(m)
    p1, _, b1, _ = _participation(m, r=r_alt)
    _ligne((f"Budget option a {m.r*100:.2f}%", f"{b0*m.nominal:,.0f} EUR"),
           (f"Participation à {m.r*100:.2f}%", _pct(p0, 0)),
           (f"Budget option a {r_alt*100:.2f}%", f"{b1*m.nominal:,.0f} EUR"),
           (f"Participation à {r_alt*100:.2f}%", _pct(p1, 0), _pts(p1 - p0)))
    if min(b0, b1) <= 0:
        st.caption("Un budget négatif veut dire que la garantie et la marge consomment "
                   "déjà tout le nominal : le produit ne peut pas être construit.")


def demo_worst_of(m):
    st.caption("Autocall Phoenix 3 ans, trimestriel, trigger 100 %, coupon 70 %, capital "
               "60 %. Les trois cas en Black-Scholes multi-actifs, comme dans la page "
               "Autocall, sans risque émetteur.")
    v_p = m.vol_atm if m.BATES else M.vol_au_strike(0.60, m.vol_atm, m.skew)
    n = min(m.n_paths, 20000)
    cas = [("Une seule action", sim_bs(S0, AC_T, m.r, m.q, v_p, n)),
           ("Worst-of 3, corrélation 0,8", sim_panier(3, 0.8, S0, AC_T, m.r, m.q, v_p, n)),
           ("Worst-of 3, corrélation 0,3", sim_panier(3, 0.3, S0, AC_T, m.r, m.q, v_p, n))]
    res = [(nom, _autocall(m, ch, tau=None)) for nom, ch in cas]
    c0, p0 = res[0][1][0], res[0][1][3]
    _ligne(*[(f"Coupon : {nom.lower()}", f"{x[0]*100:,.2f}% p.a.",
              None if i == 0 else _pts(x[0] - c0)) for i, (nom, x) in enumerate(res)])
    _ligne(*[(f"Proba de perte : {nom.lower()}", _pct(x[3]),
              None if i == 0 else _pts(x[3] - p0)) for i, (nom, x) in enumerate(res)])


def demo_skew(m):
    st.caption("BRC 1 an, barrière 70 % continue, strike 100 %.")
    if m.BATES:
        rho_alt = 0.0 if m.rho < 0 else -0.70
        cas = [(f"corrélation spot/vol {rho_alt:+.2f}", rho_alt),
               (f"corrélation spot/vol {m.rho:+.2f}", m.rho)]
        res = []
        for nom, rho in cas:
            ch = _chemins(m, 1.0, 0.70, rho=rho)
            vol70 = M.vol_implicite_bates(ch[0], 1.0, m.r, m.q, [0.70])[0]
            res.append((nom, vol70, _brc(m, ch)))
        st.caption("Sous Bates, le skew n'est pas un réglage : il vient de la corrélation "
                   "spot/vol. La vol au strike 70 % est extraite des prix simulés.")
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
    st.caption("BRC 1 an, barrière 70 % continue, strike 100 %. Taux choqué de +2 points.")
    r1 = m.r + 0.02
    c0 = _brc(m, _chemins(m, 1.0, 0.70))
    c1 = _brc(m, _chemins(m, 1.0, 0.70, r=r1), r=r1)
    _ligne((f"Coupon, taux à {m.r*100:.2f}%", f"{c0*100:,.2f}% p.a."),
           (f"Coupon, taux à {r1*100:.2f}%", f"{c1*100:,.2f}% p.a.", _pts(c1 - c0)),
           ("Part de la hausse transmise", _pct((c1 - c0) / 0.02, 0)))


def demo_spread(m):
    s = m.spread if m.spread > 0 else 150
    st.caption("BRC 1 an, barrière 70 % continue, strike 100 %."
               + ("" if m.spread > 0 else " Le spread de la barre latérale est à 0 : la "
                  "démonstration prend 150 bp."))
    ch = _chemins(m, 1.0, 0.70)
    tau = sim_defaut(len(ch[0]), 1.0, s, m.recovery)
    c0, c1 = _brc(m, ch, tau=None), _brc(m, ch, tau=tau)
    _ligne(("Coupon, émetteur sans risque", f"{c0*100:,.2f}% p.a."),
           (f"Coupon, spread {s} bp", f"{c1*100:,.2f}% p.a.", _pts(c1 - c0)),
           ("Proba de défaut sur 1 an", _pct((tau <= 1.0).mean(), 2)))


def demo_defaut(m):
    s = m.spread if m.spread > 0 else 150
    lam = s / 10000.0 / max(1.0 - m.recovery, 0.01)
    st.caption(f"Spread {s} bp, recouvrement {m.recovery*100:.0f} %."
               + ("" if m.spread > 0 else " Le spread de la barre latérale est à 0 : la "
                  "démonstration prend 150 bp."))
    p3, p5 = 1 - math.exp(-lam * 3), 1 - math.exp(-lam * 5)
    _ligne(("Intensité de défaut", f"{lam*100:,.2f}% par an"),
           ("Proba de défaut sur 3 ans", _pct(p3)),
           ("Proba de défaut sur 5 ans", _pct(p5)),
           ("Perte attendue sur 5 ans", f"{m.nominal*p5*(1-m.recovery):,.0f} EUR"))


def demo_vega(m):
    st.caption(f"Le put down-and-in d'un BRC 1 an, barrière 70 %, strike 100 %, pour "
               f"{m.nominal:,.0f} EUR de nominal.")
    n, sig = m.nominal / 100, _vol_put(m)
    g = _grecques_di(m, 100.0, 1.0, sig)
    choc = (_put_di(m, 100.0, 1.0, sig + 0.05) - g["prix"]) * n
    _ligne(("Valeur du put vendu par le client", f"{g['prix']*n:,.2f} EUR"),
           ("Vega du client", f"{-g['vega']*n:,.2f} EUR par pt"),
           ("Si la vol prend 5 points", f"{-choc:,.2f} EUR"))
    _note_formule(m, sig)


def demo_gamma(m):
    st.caption("Couverture du put down-and-in d'un BRC (barrière 70, strike 100), en "
               "actions détenues par le trader, en % du nominal.")
    sig = _vol_put(m)
    g0, g1 = _grecques_di(m, 100.0, 1.0, sig), _grecques_di(m, 72.0, 1 / 12, sig)
    vanille = lambda S, T, sigma, r: M.bs_put(S, 100.0, T, r, sigma, m.q)
    g2 = M.grecques(vanille, 69.0, 1 / 12, sig, m.r)
    c0, c1, c2 = _couverture(g0, 100), _couverture(g1, 72), _couverture(g2, 69)
    _ligne(("À l'émission, spot 100", _pct(c0, 0)),
           ("À 1 mois de l'échéance, spot 72", _pct(c1, 0)),
           ("Barrière touchée, spot 69", _pct(c2, 0)),
           ("À revendre d'un coup", _pct(c1 - c2, 0)))
    st.caption(f"Gamma à 72 à un mois : {g1['gamma']/max(g0['gamma'], 1e-12):,.0f} fois "
               "celui de l'émission.")
    _note_formule(m, sig)


def demo_theta(m):
    st.caption(f"Le put down-and-in d'un BRC 1 an, barrière 70 %, strike 100 %, pour "
               f"{m.nominal:,.0f} EUR, sous-jacent inchangé à 100.")
    n, sig = m.nominal / 100, _vol_put(m)
    p0, p6 = _put_di(m, 100.0, 1.0, sig) * n, _put_di(m, 100.0, 0.5, sig) * n
    theta = _grecques_di(m, 100.0, 1.0, sig)["theta"] * n
    _ligne(("Put vendu, à l'émission", f"{p0:,.2f} EUR"),
           ("Même spot, 6 mois plus tard", f"{p6:,.2f} EUR"),
           ("Encaissé par le client", f"{p0-p6:,.2f} EUR"),
           ("Par jour, au départ", f"{-theta:,.3f} EUR"))
    _note_formule(m, sig)


def demo_couverture(m):
    st.caption("La banque achète le put down-and-in (barrière 70, strike 100). Actions "
               "détenues par le trader pour être neutre au delta, en % du nominal.")
    sig = _vol_put(m)
    g0, g1 = _grecques_di(m, 100.0, 1.0, sig), _grecques_di(m, 85.0, 0.75, sig)
    c0, c1 = _couverture(g0, 100), _couverture(g1, 85)
    _ligne(("À l'émission, spot 100", _pct(c0, 0)),
           ("3 mois plus tard, spot 85", _pct(c1, 0)),
           ("Racheté dans la baisse", _pct(c1 - c0, 0)))
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
      "répondez-vous ?",
      "Je ne le contredis pas, je lui explique d'où viennent les 9 %. Le coupon n'est pas "
      "un intérêt : c'est la prime que la banque lui verse pour l'assurer contre une forte "
      "baisse du sous-jacent. 9 % quand le taux sans risque est à 3 %, cela veut dire que "
      "la protection qu'il vend a de la valeur : vol élevée, barrière proche, ou émetteur "
      "qui paie un spread. Ensuite je vérifie que sa vue colle au produit : il doit être "
      "prêt à détenir l'action s'il est livré sous le strike. Si oui, le produit lui "
      "convient ; sinon on baisse la barrière et on accepte moins de coupon.",
      "Le coupon est le prix d'un risque, jamais un cadeau. Relier le chiffre à la vue du "
      "client.",
      demo_bonne_affaire),
    Q(CLIENT,
      "Un client veut plus de sécurité sur son BRC. Que lui proposez-vous, et combien ça "
      "coûte ?",
      "Trois leviers, tous payés en coupon. Baisser la barrière, de 70 % à 60 % par "
      "exemple. Observer la barrière seulement à l'échéance au lieu d'en continu. "
      "Raccourcir la maturité, ce qui laisse moins de temps au sous-jacent pour chuter. "
      "Chaque levier réduit la valeur du put que le client vend, donc la prime qu'on lui "
      "reverse. Mon rôle est de chiffrer l'arbitrage, « vous perdez tant de coupon et la "
      "probabilité de perte passe de A à B », puis de le laisser choisir.",
      "Plus de sécurité, c'est toujours moins de coupon. Savoir chiffrer le prix de la "
      "sécurité.",
      demo_securite),
    Q(CLIENT,
      "Expliquez un autocall à un client particulier en trente secondes.",
      "« Vous investissez pour trois ans au plus sur l'Euro Stoxx 50. Chaque trimestre, si "
      "l'indice est au-dessus de son niveau de départ, vous êtes remboursé avec un coupon "
      "et le produit s'arrête. S'il est entre 70 % et 100 %, vous touchez le coupon et on "
      "continue. À l'échéance, vous récupérez votre capital tant que l'indice n'a pas perdu "
      "plus de 40 %. En dessous, vous subissez toute la baisse, comme si vous aviez acheté "
      "l'indice. » Pas de jargon, trois scénarios, et le risque dit clairement.",
      "Clarté et honnêteté sur le scénario de perte. Les mots trigger, Phoenix ou mémoire "
      "ne servent à rien devant un particulier."),
    Q(CLIENT,
      "Le client compare le coupon de 8 % d'un autocall à une obligation à 4 %. "
      "Qu'est-ce qui cloche dans la comparaison ?",
      "Deux choses. Le risque d'abord : l'obligation rembourse le nominal sauf défaut, "
      "l'autocall expose le capital sous la barrière. La durée ensuite : le rappel anticipé "
      "coupe les coupons. Si le produit est rappelé dès la première ou la deuxième "
      "observation, le scénario le plus fréquent, le client touche 8 % par an pendant "
      "quelques mois, puis doit replacer son argent, peut-être dans un marché moins "
      "favorable. Il faut raisonner en durée de vie attendue et en probabilité de perte, "
      "pas en coupon affiché.",
      "Le coupon d'un autocall n'est pas un rendement annuel garanti : durée incertaine et "
      "capital à risque.",
      demo_autocall),
    Q(CLIENT,
      "Un client veut profiter de la hausse du marché sans risquer son capital. Que "
      "proposez-vous ?",
      "Un capital garanti : une obligation zéro-coupon qui assure 100 % à l'échéance, et un "
      "call qui donne la participation à la hausse. Je cadre tout de suite ce qui fait la "
      "participation : la maturité, le niveau de protection et un éventuel plafond. Et je "
      "précise que la garantie est celle de l'émetteur : si la banque fait défaut, la "
      "garantie tombe avec elle. Avec des taux nettement positifs, ce produit redevient "
      "attractif, ce qui n'était pas le cas quand les taux étaient nuls.",
      "Proposer le produit qui colle à la vue, puis nommer le risque émetteur sans qu'on "
      "le demande."),
    Q(CLIENT,
      "Le client trouve trop faible la participation de son capital garanti. Quels leviers "
      "activez-vous ?",
      "Je rappelle que la participation n'est pas fixée par la banque : c'est le budget "
      "restant divisé par le prix du call. Puis je propose les leviers un par un, avec leur "
      "prix. Allonger la maturité : on met moins de côté pour garantir le capital. "
      "Garantir 90 % au lieu de 100 %. Plafonner la hausse, ce qui revient à revendre un "
      "call plus haut. Choisir un sous-jacent moins volatil ou à plus fort dividende, dont "
      "le call coûte moins cher. Chaque levier échange un peu de sécurité ou de potentiel "
      "contre de la participation, et c'est au client d'arbitrer.",
      "La participation est un résultat, pas un choix. Connaître les leviers et leur sens.",
      demo_leviers_garanti),
    Q(CLIENT,
      "Un institutionnel vous dit que votre coupon est 50 bp sous celui d'un concurrent, "
      "sur le même BRC. Que faites-vous ?",
      "Je vérifie d'abord que les structures sont vraiment identiques : observation de la "
      "barrière, strike, dates, livraison physique ou cash, sous-jacent. Si oui, l'écart "
      "vient en général de trois sources : le spread de crédit de l'émetteur, la marge, et "
      "l'appétit du livre du trader pour ce risque à ce moment-la. Je le dis franchement : "
      "un coupon plus haut chez un émetteur moins bien noté rémunère un risque de crédit. "
      "Puis je retourne voir le desk pour savoir si on peut s'aligner, sans rien promettre "
      "avant.",
      "Ne pas paniquer sur le prix : comparer à structure identique, expliquer l'écart, "
      "puis négocier en interne."),
    Q(CLIENT,
      "Le marché vient de perdre 20 % et la volatilité a doublé. Votre client a peur. Que "
      "lui proposez-vous ?",
      "C'est souvent le meilleur moment pour vendre de la volatilité, à condition de le "
      "faire prudemment. Une vol élevée rend les puts chers : à barrière égale, le coupon "
      "d'un BRC est bien plus haut qu'avant la baisse. Et comme le niveau de départ est "
      "déjà 20 % plus bas, une barrière à 60 % correspond à une baisse de 52 % depuis les "
      "plus hauts. Je propose donc une barrière plus profonde et une maturité courte, la "
      "vol courte étant la plus élevée en période de stress. Mais je ne force pas un client "
      "qui a peur : je chiffre, il décide.",
      "Transformer la peur du marché en fenêtre commerciale, sans pousser le client.",
      demo_stress),
    Q(CLIENT,
      "Un client est légèrement haussier sur deux ans et ne veut pas renoncer à la hausse. "
      "BRC ou Bonus Certificate ?",
      "Le Bonus. Il garde toute la hausse et assure un remboursement minimum, le bonus, tant que la barrière tient. Le BRC plafonne le gain au coupon : si le marché monte de 30 %, le client du BRC touche son coupon et rien de plus, ce qui contredit sa vue. Le BRC convient mieux à un client qui pense que le marché va stagner, car il monétise directement l'absence de mouvement. Le travail du Sales, c'est d'abord de reformuler la vue du client, puis de choisir le payoff qui la paie.",
      "Partir de la vue du client, pas du produit qu'on a envie de vendre.",
      demo_bonus),
    Q(CLIENT,
      "Un client vous demande : « Quel est le pire scénario ? » Que répondez-vous ?",
      "Je réponds directement, avec un chiffre. Sur un BRC barrière 70 % : le sous-jacent "
      "touche la barrière et finit à -50 % ; le client récupère 50 % de son capital plus "
      "le coupon. Et indépendamment du marché, si l'émetteur fait défaut, il ne récupère "
      "que le taux de recouvrement. Je n'esquive jamais cette question : un client qui a "
      "entendu le pire scénario et achète quand même est un client qui reste.",
      "Transparence sur la perte maximale, chiffrée. C'est ce qui construit la confiance.",
      demo_pire),
    Q(CLIENT,
      "Le client vous demande combien la banque gagne sur ce produit.",
      "Je réponds franchement : les coûts figurent de toute façon dans le document "
      "d'information (KID PRIIPs) et doivent être détaillés au titre de MiFID II. La marge "
      "est prise sur le budget option, souvent de l'ordre de 0,5 à 1,5 % par an selon le "
      "produit et le client. Elle paie la structuration, le capital immobilisé et les "
      "risques que la banque ne couvre pas parfaitement : saut de delta à la barrière, "
      "skew, corrélation. Un institutionnel la négocie ; l'important est qu'il sache ce "
      "qu'il paie, et pour quoi.",
      "Pas de tabou sur la marge : la justifier par les risques que la banque garde.",
      demo_marge),
    Q(CLIENT,
      "Comment présentez-vous le même produit à un institutionnel et à un client privé ?",
      "À un institutionnel, je parle prix et risques de marché : quelle vol implicite et "
      "quel skew sont dans le prix, quel spread émetteur, comment le produit se comporte "
      "en mark-to-market, quelle liquidité sur le secondaire. Il compare avec ce qu'il "
      "pourrait répliquer lui-même avec des options listées. À un client privé ou à son "
      "banquier, je parle scénarios : que se passe-t-il si le marché monte, stagne, baisse "
      "de 20 %, baisse de 50 %. Le produit est le même, le langage change.",
      "Adapter le discours à l'interlocuteur : grecques pour l'un, scénarios pour l'autre."),
    Q(CLIENT,
      "Votre client a acheté un BRC il y a six mois. Le sous-jacent est à 75 %, la "
      "barrière à 70 %. Que lui dites-vous ?",
      "Je l'appelle avant qu'il m'appelle. Je lui donne les faits : la barrière n'est pas "
      "touchée et le coupon reste du quoi qu'il arrive, mais le produit vaut nettement "
      "moins sur le marché secondaire, car le put qu'il a vendu est proche de s'activer. "
      "Puis je lui présente ses options : conserver le produit en acceptant le risque "
      "d'être remboursé en actions, ou sortir maintenant au prix du secondaire. Je ne lui "
      "promets pas que ça va remonter. La relation se construit sur ces appels-la.",
      "Proactivité et honnêteté dans la mauvaise passe : c'est la que se gagne la "
      "fidélité.",
      demo_mtm),
    Q(CLIENT,
      "Un client veut un placement sans risque qui rapporte 7 %. Que lui répondez-vous ?",
      "Que ça n'existe pas, et je l'explique en une phrase : tout ce qui dépasse le taux "
      "sans risque rémunère un risque. Puis je retourne la question : quel risque êtes-vous "
      "prêt à prendre ? Une baisse du marché au-delà d'un seuil, un gain plafonné, le "
      "crédit d'un émetteur, une durée incertaine. À partir de sa réponse, je construis le "
      "produit. Refuser une promesse impossible, c'est ce qui rend crédible tout le reste.",
      "Pas de rendement sans risque. Le Sales transforme une demande irréaliste en choix "
      "de risque."),
    Q(CLIENT,
      "Un client veut le coupon d'un worst-of sur trois actions mais ne mesure pas le "
      "risque. Comment le lui faire comprendre ?",
      "Une phrase : « Vous n'achetez pas trois actions, vous achetez la pire des trois. » "
      "Il suffit qu'une seule chute pour que le produit se comporte comme elle, même si les "
      "deux autres montent. Puis je chiffre : coupon et probabilité de perte du worst-of "
      "face au même produit sur une seule action. Le coupon plus élevé s'accompagne d'une "
      "probabilité de perte plus élevée, et c'est à lui de juger si l'échange lui convient.",
      "Faire comprendre le worst-of par une image simple, puis par les chiffres.",
      demo_worst_of),

    # ---------------- Mecanique des produits ------------------------
    Q(MECANIQUE,
      "Décomposez un Barrier Reverse Convertible.",
      "Une obligation émise par la banque, plus un put down-and-in que le client vend à la "
      "banque. Le client prête son argent et assure la banque contre une forte baisse du "
      "sous-jacent. Son coupon vient des intérêts de l'obligation et de la prime du put, "
      "plus le spread de crédit si l'émetteur en paie un, moins la marge. Si la barrière "
      "n'est jamais touchée, il récupère 100 % plus le coupon. Si elle est touchée et que "
      "le sous-jacent finit sous le strike, il est remboursé en actions et encaisse toute "
      "la baisse depuis le strike.",
      "Savoir dire qui vend l'option à qui : le client est vendeur du put.",
      demo_decomposition),
    Q(MECANIQUE,
      "Sur un Autocall Phoenix, quelle différence entre le trigger de rappel, la barrière "
      "de coupon et la barrière de capital ?",
      "Trois niveaux, trois rôles. Le trigger, souvent à 100 %, est observé à chaque date : "
      "au-dessus, le produit s'arrête et rembourse le nominal avec le coupon. La barrière "
      "de coupon, par exemple 70 %, décide si le coupon de la période est payé. La barrière "
      "de capital, par exemple 60 %, n'est regardée qu'à l'échéance : en dessous, le client "
      "encaisse toute la baisse depuis 100 %, pas seulement la partie sous la barrière.",
      "Le piège de la barrière de capital : sous 60 %, la perte se compte depuis 100 %."),
    Q(MECANIQUE,
      "Comment est construit un capital garanti, et d'où vient la participation ?",
      "Une obligation zéro-coupon qui reconstitue 100 % à l'échéance, plus un call acheté avec ce qui reste. À 3 % sur 5 ans, garantir 1 000 EUR coûte environ 861 EUR aujourd'hui ; les 139 restants, moins la marge, achètent le call. La participation est donc un résultat : le budget option divisé par le prix du call. La banque ne la choisit pas.",
      "La participation est un reste de budget.",
      demo_capital),
    Q(MECANIQUE,
      "Que se passe-t-il sur un Bonus Certificate si la barrière est touchée ?",
      "Le bonus disparaît définitivement. Le produit, c'est le sous-jacent sans ses "
      "dividendes plus un put down-and-out ; quand la barrière casse, le put s'éteint et le "
      "client se retrouve avec un simple tracker. Il ne perd pas plus que s'il détenait le "
      "sous-jacent, mais il perd sa protection. C'est une protection conditionnelle, pas "
      "une garantie. Sa force face au BRC : la hausse n'est pas plafonnée.",
      "Protection conditionnelle n'est pas garantie : le mot à ne jamais employer."),
    Q(MECANIQUE,
      "Qu'est-ce que l'effet mémoire sur un Phoenix, et combien coûte-t-il ?",
      "Un coupon non versé parce que le sous-jacent était sous la barrière de coupon est "
      "rattrapé à la première date où il repasse au-dessus. Le client ne perd donc ses "
      "coupons que si le sous-jacent reste durablement bas. C'est une option supplémentaire "
      "offerte au client : elle a un prix, et la banque le paie en baissant le coupon par "
      "période.",
      "Tout ce qui protège le client se paie en coupon.",
      demo_memoire),
    Q(MECANIQUE,
      "Sur un BRC, barrière observée en continu ou à l'échéance : qu'est-ce que ça change ?",
      "En continu, un seul passage sous la barrière a n'importe quel moment suffit à armer "
      "le risque, même une mèche en séance. À l'échéance, seul le cours final compte. "
      "L'observation à l'échéance protège mieux le client, donc le put qu'il vend vaut "
      "moins et le coupon baisse. Deux BRC « barrière 70 % » peuvent ainsi avoir des "
      "risques très différents : c'est une ligne du termsheet à toujours lire.",
      "Le mode d'observation fait partie du prix du risque.",
      demo_observation),
    Q(MECANIQUE,
      "Comment un Twin-Win permet-il de gagner à la baisse ?",
      "Le produit, c'est un tracker plus l'achat d'un put down-and-out à la monnaie. Entre "
      "la barrière et le niveau initial, le put paie : la baisse se transforme en gain. Le "
      "financement vient des dividendes abandonnés. Le danger est la discontinuité : avec "
      "une barrière à 65 % et une participation de 100 % à la baisse, le client gagne près "
      "de 35 % juste au-dessus de la barrière, et perd 35 % juste en dessous.",
      "Savoir expliquer la discontinuité à la barrière : le profil le plus dangereux à mal "
      "vendre."),

    # ---------------- Parametres de marche --------------------------
    Q(MARCHE,
      "Pourquoi le coupon d'un BRC monte-t-il quand la volatilité monte ?",
      "Parce que le client est vendeur d'un put. Plus la vol est haute, plus le sous-jacent "
      "a de chances de traverser la barrière, plus ce put vaut cher. La banque paie plus "
      "cher l'assurance que le client lui vend, et ça se voit dans le coupon. Corollaire "
      "commercial : les meilleurs moments pour placer des BRC sont les pics de vol, quand "
      "les clients ont le plus peur.",
      "Relier vol, prix du put et coupon, et en tirer la fenêtre commerciale.",
      demo_vol_brc),
    Q(MARCHE,
      "Pourquoi les capitaux garantis avaient-ils presque disparu entre 2015 et 2021 ?",
      "Parce qu'ils se financent avec les taux. La banque place juste ce qu'il faut pour "
      "rembourser 100 % à l'échéance, et le reste achète le call. A 3 % sur 5 ans, il reste "
      "environ 14 % du nominal ; à 0,5 %, il en reste 2,5 %, avant marge. Avec des taux "
      "nuls ou négatifs, la participation devenait dérisoire et le marché a bascule vers "
      "les BRC et les autocalls. La remontée des taux depuis 2022 les a fait revenir.",
      "Taux hauts, capital garanti vendable ; taux nuls, il disparaît.",
      demo_taux_garanti),
    Q(MARCHE,
      "Pourquoi un autocall worst-of sur trois actions paie-t-il plus qu'un autocall sur "
      "une seule ?",
      "Parce que le client vend de la corrélation. Il faut que les trois actions tiennent : "
      "il suffit qu'une seule chute pour que le produit se comporte comme elle. Moins les "
      "actions sont corrélées, plus il y a de chances qu'il y ait un traînard, plus le "
      "risque que le client vend vaut cher, plus le coupon monte. Le coupon plus élevé "
      "n'est pas un cadeau : la probabilité de perte monte avec lui.",
      "Dire « le client est short corrélation » et le traduire en probabilité de perte.",
      demo_worst_of),
    Q(MARCHE,
      "Quel est l'effet du skew sur le coupon d'un BRC ?",
      "Le skew rend les puts bas plus chers, en vol, que les options à la monnaie. Or le "
      "risque du put que vend le client d'un BRC se concentre autour de la barrière, loin "
      "sous le spot. Avec du skew, ce put est pricé à une vol plus haute que la vol ATM : "
      "le coupon monte. Le skew a deux causes : les marchés baissent plus brutalement "
      "qu'ils ne montent, et les institutionnels achètent massivement des puts pour se "
      "protéger.",
      "Relier skew, vol au niveau de la barrière, prix du put et coupon.",
      demo_skew),
    Q(MARCHE,
      "Un dividende élevé sur le sous-jacent, c'est bon ou mauvais pour un capital "
      "garanti ?",
      "Bon pour la participation. Le client ne touche pas les dividendes : ils abaissent le "
      "prix à terme du sous-jacent, donc le prix du call que la banque doit acheter. Budget "
      "inchangé, call moins cher : la participation monte. Le revers : sur un tracker, "
      "c'est exactement ce que le client abandonne, environ 7,8 % sur 3 ans avec 2,5 % de "
      "dividende.",
      "Les dividendes abandonnés financent la structure.",
      demo_dividende),
    Q(MARCHE,
      "Vol implicite et vol réalisée : pourquoi l'écart compte-t-il pour un produit "
      "structure ?",
      "Sur les indices, la vol implicite est en moyenne 2 à 4 points au-dessus de la vol "
      "qui se réalise ensuite : c'est la prime de risque de variance. Le client qui vend de "
      "la vol à travers un BRC ou un autocall capte cette prime. C'est la justification "
      "économique de ces produits : il est payé pour porter le risque de krach, gagne la "
      "plupart des années et perd beaucoup lors de la mauvaise. Le présenter ainsi est plus "
      "juste que de parler d'un coupon élevé.",
      "Savoir dire pourquoi ces produits existent : la variance risk premium."),
    Q(MARCHE,
      "Que fait une hausse des taux au coupon d'un BRC ?",
      "Elle l'augmente, mais moins que point pour point. Le coupon, c'est d'abord le taux "
      "de l'obligation, puis la prime du put. Quand les taux montent de 2 points, la "
      "composante obligataire monte d'environ 2 points ; mais le prix à terme du "
      "sous-jacent monte aussi, ce qui rend le put moins cher et freine la hausse. C'est "
      "pour cela qu'à vol égale, les BRC affichaient des coupons bien plus élevés en 2023 "
      "qu'en 2020.",
      "Séparer la composante taux de la composante option.",
      demo_taux_brc),

    # ---------------- Risque emetteur -------------------------------
    Q(EMETTEUR,
      "Pourquoi un spread de crédit plus large augmente-t-il le coupon ?",
      "Un produit structuré est une dette de la banque. Si l'émetteur paie 150 bp "
      "au-dessus du taux sans risque pour se financer sur le marché obligataire, il peut "
      "reverser ce financement plus cher dans le produit. Le client est donc rémunéré pour "
      "un risque de crédit en plus du risque de marché. A structure identique, un coupon "
      "supérieur à la concurrence doit faire poser la question : l'émetteur est-il moins "
      "bien noté ?",
      "Coupon plus haut peut vouloir dire signature plus faible.",
      demo_spread),
    Q(EMETTEUR,
      "Que récupère le client si l'émetteur fait défaut, même si la barrière n'a jamais "
      "été touchée ?",
      "Seulement le taux de recouvrement de la dette senior de la banque, 40 % par "
      "convention, parfois bien moins. Le client n'est pas propriétaire des actions, il est "
      "créancier de l'émetteur. En 2008, des porteurs de produits Lehman, y compris à "
      "capital garanti, n'ont récupéré qu'une petite fraction de leur mise, quel que soit "
      "le comportement de leur sous-jacent. « Capital garanti » veut dire garanti par "
      "l'émetteur, pas sans risque.",
      "Un produit structuré est une dette de la banque, pas une détention d'actions."),
    Q(EMETTEUR,
      "Comment estimer le risque de défaut d'un émetteur a partir de son spread ?",
      "Règle simple : intensité de défaut égale spread divisé par un moins le taux de "
      "recouvrement. À 150 bp avec 40 % de recouvrement, cela donne 2,5 % par an, soit "
      "environ 7 % de probabilité de défaut sur 3 ans. La perte attendue, probabilité fois "
      "perte en cas de défaut, retombe à peu près sur le spread : c'est ce que le marché "
      "fait payer. Ces probabilités sont risque-neutres, donc plus élevées que les défauts "
      "historiques, car elles incluent une prime de risque.",
      "Savoir passer d'un spread à une probabilité de défaut, et nuancer.",
      demo_defaut),
    Q(EMETTEUR,
      "Deux BRC identiques, deux émetteurs, 60 bp d'écart de coupon. Comment conseillez-vous "
      "un institutionnel ?",
      "À structure identique, l'écart vient surtout de la signature : l'émetteur qui paie plus se finance plus cher, le marché le juge plus risqué. Je compare notations et CDS, et je regarde l'exposition que le client a déjà sur chaque banque : un institutionnel ne veut pas concentrer son risque de contrepartie. On peut répartir les tickets entre plusieurs émetteurs, ou, si le risque de crédit le gêne vraiment, proposer une version collatéralisée en échange d'un coupon plus faible.",
      "Le coupon supérieur se paie en risque de crédit. Penser diversification des "
      "contreparties."),
    Q(EMETTEUR,
      "Qu'est-ce que le wrong-way risk, et pourquoi compte-t-il sur un produit structuré ?",
      "C'est quand le risque de défaut de la contrepartie augmente précisément quand "
      "l'exposition est la plus forte. Une banque a plus de chances de faire défaut en "
      "plein krach, au moment où le sous-jacent d'un BRC s'effondre. Le client cumule "
      "alors la perte de marché et la perte de crédit. Les modèles simples, dont celui de "
      "cette appli, supposent le défaut indépendant du marché : ils sous-estiment ce "
      "risque.",
      "Comprendre que crédit et marché sont corrélés au pire moment."),

    # ---------------- Grecques et couverture ------------------------
    Q(GRECQUES,
      "Quelle est la position en vega d'un client qui achète un BRC ?",
      "Short vega : il a vendu un put. Si la vol implicite monte après l'achat, le put vaut plus cher et la valeur de marché de son BRC baisse, même si le sous-jacent n'a pas bougé. En face, la banque est long vega et revend de la vol sur le marché pour se couvrir. C'est un phénomène connu : les desks qui ont vendu beaucoup d'autocalls se retrouvent longs de vol longue et la revendent, ce qui pèse sur ces vols.",
      "Le client d'un BRC ou d'un autocall est vendeur de volatilité.",
      demo_vega),
    Q(GRECQUES,
      "Pourquoi le gamma explose-t-il près de la barrière ?",
      "Juste au-dessus de la barrière, à l'approche de l'échéance, la valeur du put "
      "down-and-in change énormément pour un petit mouvement : un point de plus ou de moins "
      "décide si le client perd 30 % ou rien. Le delta du trader devient très grand, il "
      "peut détenir plusieurs fois le nominal en actions. Si la barrière est touchée, le "
      "put devient un put ordinaire et le delta retombe d'un coup : il doit revendre une "
      "grosse partie de sa couverture, dans un marché qui baisse. Ce risque se paie dans le "
      "prix : les barrières proches du spot sont moins bien traitées.",
      "Lier gamma, barrière et coût de couverture, et comprendre pourquoi le desk dégrade "
      "le prix.",
      demo_gamma),
    Q(GRECQUES,
      "Le client d'un BRC est-il long ou short theta ?",
      "Long theta, comme tout vendeur d'option. Chaque jour qui passe sans que la barrière "
      "soit touchée, le put qu'il a vendu perd de la valeur, et c'est lui qui l'encaisse. "
      "La contrepartie : il est short gamma, il souffre des grands mouvements. Theta et "
      "gamma sont les deux faces de la même position.",
      "Vendeur d'option : long theta, short gamma.",
      demo_theta),
    Q(GRECQUES,
      "Comment la banque couvre-t-elle un BRC qu'elle vient de vendre ?",
      "La banque est acheteuse du put down-and-in. Un put perd de la valeur quand le "
      "sous-jacent monte : pour neutraliser ce risque, le trader achète des actions, en "
      "quantité égale au delta, et ajuste chaque jour. Quand le marché baisse, le delta du "
      "put s'alourdit et il en rachète ; quand il remonte, il revend. Le vega et le skew se "
      "couvrent en revendant de la vol sur le marché d'options. Ce qui reste, le saut de "
      "delta à la barrière ou la corrélation d'un worst-of, est gardé dans le livre et "
      "payé par la marge.",
      "Le Sales ne porte pas le risque, mais doit savoir comment le trader le couvre.",
      demo_couverture),
    Q(GRECQUES,
      "Pourquoi la couverture en delta coûte-t-elle de l'argent, alors qu'en théorie elle "
      "est parfaite ?",
      "Black-Scholes suppose un rebalancement continu et gratuit : dans ce monde, la "
      "couverture réplique exactement l'option. En pratique, on rebalance à intervalles "
      "discrets, ce qui laisse une erreur de couverture, et chaque transaction paie la "
      "fourchette achat-vente. Rebalancer plus souvent réduit l'erreur mais augmente les "
      "frais : c'est un arbitrage. Ce coût fait partie de ce que la marge doit couvrir, "
      "elle n'est pas un pur profit.",
      "La marge paie aussi un coût de couverture réel."),
    Q(GRECQUES,
      "Comment calcule-t-on une grecque sur un produit pricé par Monte Carlo ?",
      "Par bump and revalue : on décale un paramètre de peu, on reprice, et on divise "
      "l'écart par le choc. Le détail qui compte : réutiliser exactement les mêmes tirages "
      "aléatoires pour les deux prix. Sinon le bruit de simulation, souvent plus grand que "
      "l'effet cherché, domine la différence et la grecque est inutilisable. Pour un "
      "gamma, c'est encore pire, d'où l'intérêt des formules fermées quand elles existent, "
      "comme pour le put down-and-in.",
      "Connaître la technique standard et son piège numérique."),
]


# =====================================================================
# JEU « DEVINE LE COUPON »
# L'appli tire un produit et un marche, l'utilisateur estime le resultat,
# puis le moteur le calcule exactement comme la page produit (memes
# fonctions, meme graine, 5 000 trajectoires). Les reglages tires portent
# les cles des curseurs : le bouton « Refaire ce produit dans sa page »
# les ecrit tels quels dans la barre laterale et dans la page.
# =====================================================================

QUESTIONS_ENTRETIEN, JEU = "Questions d'entretien", "Devine le coupon"
DV_BRC, DV_CP = "Barrier Reverse Convertible", "Capital Protection"
DV_AC, DV_DC = "Autocall Phoenix", "Discount Certificate"
DV_PRODUITS = [DV_BRC, DV_CP, DV_AC, DV_DC]
DV_HASARD = "Au hasard"
DV_PAGES = {DV_BRC: "3 - Barrier Reverse Convertible", DV_CP: "2 - Capital Protection",
            DV_AC: "4 - Autocall Phoenix", DV_DC: "7 - Discount Certificate"}
DV_N = 5000
CONTINUE, EUROPEENNE = "Continue (américaine)", "À l'échéance (européenne)"
FREQUENCES = {"Trimestrielle": 4, "Semestrielle": 2, "Annuelle": 1}
CLASSIQUE = "Indice classique"

# Le point de depart : reglages par defaut de la barre laterale et des pages produits
DV_REF_MARCHE = {"taux": 3.0, "dividende": 2.5, "vol": 22.0, "skew": 2.0, "spread": 0}
DV_REF_PRODUIT = {
    DV_BRC: {"brc_maturite": 1.0, "brc_barriere": 70, "brc_observation": CONTINUE},
    DV_CP: {"cp_maturite": 5.0, "cp_protection": 100, "cp_cap": 300},
    DV_AC: {"ac_maturite": 3.0, "ac_frequence": "Trimestrielle", "ac_trigger": 100,
            "ac_bar_coupon": 70, "ac_bar_capital": 60, "ac_memoire": True},
    DV_DC: {"dc_maturite": 1.0, "dc_cap": 100},
}
# Les parametres qui comptent pour chaque produit (le Discount est hors risque emetteur)
DV_MARCHE = {DV_BRC: ["vol", "skew", "taux", "dividende", "spread"],
             DV_CP: ["taux", "spread", "vol", "skew", "dividende"],
             DV_AC: ["vol", "skew", "taux", "dividende", "spread"],
             DV_DC: ["vol", "skew", "taux", "dividende"]}

# Ce qu'on devine
DV_GRANDEUR = {
    DV_BRC: "le coupon du BRC, en % par an",
    DV_CP: "la participation à la hausse, en %",
    DV_AC: "le coupon de l'autocall, en équivalent annuel (% par an)",
    DV_DC: "la décote du certificat, en % du prix du sous-jacent",
}
DV_CIBLE = {DV_BRC: "au coupon", DV_CP: "à la participation", DV_AC: "au coupon",
            DV_DC: "à la décote"}

DV_NOMS = {
    "vol": "Volatilité ATM", "skew": "Skew", "taux": "Taux sans risque",
    "dividende": "Dividende", "spread": "Spread émetteur",
    "brc_maturite": "Maturité", "brc_barriere": "Barrière", "brc_observation": "Observation",
    "cp_maturite": "Maturité", "cp_protection": "Protection", "cp_cap": "Cap",
    "ac_maturite": "Maturité maximale", "ac_frequence": "Observation", "ac_trigger": "Trigger",
    "ac_bar_coupon": "Barrière de coupon", "ac_bar_capital": "Barrière de capital",
    "ac_memoire": "Effet mémoire", "dc_maturite": "Maturité", "dc_cap": "Cap",
}

# Les memes, dans une phrase
DV_PHRASE = {
    "vol": "la volatilité ATM", "skew": "le skew", "taux": "le taux sans risque",
    "dividende": "le dividende", "spread": "le spread émetteur",
    "brc_maturite": "la maturité", "brc_barriere": "la barrière",
    "brc_observation": "l'observation de la barrière", "cp_maturite": "la maturité",
    "cp_protection": "le niveau de protection", "cp_cap": "le cap",
    "ac_maturite": "la maturité maximale", "ac_frequence": "la fréquence d'observation",
    "ac_bar_coupon": "la barrière de coupon", "ac_bar_capital": "la barrière de capital",
    "ac_memoire": "l'effet mémoire", "dc_maturite": "la maturité", "dc_cap": "le cap",
}


def _dv_format(cle, v):
    """Un reglage tire, comme on le lirait sur le curseur."""
    if cle == "vol":
        return f"{v:.0f} %"
    if cle in ("taux", "dividende"):
        return f"{v:.2f} %"
    if cle == "skew":
        return f"{v:.1f} pt / -10 %"
    if cle == "spread":
        return f"{v} bp"
    if cle.endswith("maturite"):
        return G.duree(v)
    if cle == "ac_memoire":
        return "Oui" if v else "Non"
    if cle == "cp_cap" and v >= 300:
        return "aucun (300 %)"
    if isinstance(v, str):
        return v
    return f"{v} %"


# Ce qui explique chaque parametre, produit par produit (vrai dans les deux sens)
DV_POURQUOI = {
    (DV_BRC, "vol"): "Plus la vol est haute, plus le put que le client vend à la banque vaut cher : la prime reversée en coupon grossit.",
    (DV_BRC, "skew"): "Le skew rend les puts bas plus chers, et c'est justement un put à barrière basse que le client vend.",
    (DV_BRC, "taux"): "Le coupon contient d'abord le taux sans risque : la banque place le nominal, chaque point de taux s'y retrouve presque tel quel.",
    (DV_BRC, "dividende"): "Le client ne touche pas les dividendes : plus ils sont élevés, plus le forward est bas, plus le put vendu vaut cher.",
    (DV_BRC, "spread"): "Le client porte le risque de l'émetteur : la banque se finance à travers le produit et lui reverse son spread.",
    (DV_BRC, "brc_maturite"): "La maturité change le temps laissé au sous-jacent pour toucher la barrière, et la durée sur laquelle la prime est étalée.",
    (DV_BRC, "brc_barriere"): "Une barrière plus haute se touche plus facilement : le put vendu vaut plus cher, le coupon monte.",
    (DV_BRC, "brc_observation"): "Observée en continu, la barrière se touche bien plus souvent qu'à l'échéance seule : le put vaut plus, le coupon aussi.",
    (DV_CP, "taux"): "Des taux plus hauts rendent le zéro-coupon moins cher : il reste plus de budget pour acheter le call.",
    (DV_CP, "spread"): "La garantie est actualisée au taux de financement de la banque : un spread plus large libère du budget, contre un risque de crédit porté par le client.",
    (DV_CP, "vol"): "Le client achète un call : plus la vol est haute, plus il coûte cher, moins le budget en achète.",
    (DV_CP, "skew"): "Avec le skew, les strikes hauts ont une vol plus basse : le call vendu au cap rapporte moins, le call spread coûte plus cher.",
    (DV_CP, "dividende"): "Les dividendes restent à la banque et abaissent le forward : le call coûte moins cher, le budget en achète plus.",
    (DV_CP, "cp_maturite"): "Plus c'est long, moins la banque doit placer pour garantir le capital : le budget option grossit, mais le call coûte aussi plus cher.",
    (DV_CP, "cp_protection"): "Chaque point de protection abandonné rend le zéro-coupon moins cher et finance plus de call.",
    (DV_CP, "cp_cap"): "Un cap plus bas revend à la banque la hausse au-delà du cap : ce call vendu finance une participation plus forte.",
    (DV_AC, "vol"): "Le client vend de la vol : plus elle est haute, plus le risque de finir sous la barrière de capital vaut cher, plus le coupon monte.",
    (DV_AC, "skew"): "Le skew rend la baisse plus chère à assurer : la barrière de capital que porte le client rapporte plus.",
    (DV_AC, "taux"): "Les taux s'ajoutent au coupon : la banque place le nominal pendant toute la vie du produit.",
    (DV_AC, "dividende"): "Le client ne touche pas les dividendes : ils abaissent le forward, le rappel devient moins probable et la perte plus probable.",
    (DV_AC, "spread"): "Le client porte le risque de l'émetteur : la banque se finance à travers le produit et lui reverse son spread.",
    (DV_AC, "ac_maturite"): "Une maturité plus longue laisse plus de temps pour finir sous la barrière de capital, et plus de dates pour être rappelé.",
    (DV_AC, "ac_frequence"): "Des observations plus rares, ce sont moins d'occasions d'être rappelé et de toucher un coupon : le produit vit plus longtemps et son risque change.",
    (DV_AC, "ac_bar_coupon"): "Une barrière de coupon plus haute fait sauter des coupons plus souvent : pour le même prix, chaque coupon payé est plus gros.",
    (DV_AC, "ac_bar_capital"): "Une barrière de capital plus haute rend la perte plus probable : le client est payé plus cher pour ce risque.",
    (DV_AC, "ac_memoire"): "Sans mémoire, un coupon sauté est perdu pour de bon : la banque peut promettre un coupon plus gros.",
    (DV_DC, "vol"): "Le client vend un call au cap : plus la vol est haute, plus ce call vaut cher, plus la décote est forte.",
    (DV_DC, "skew"): "Avec le skew, la vol au cap dépend du niveau du cap : plus haute sous 100 %, plus basse au-dessus, et le call vendu suit.",
    (DV_DC, "taux"): "Les taux rendent le call vendu plus cher et le remboursement futur moins cher aujourd'hui : la décote grandit.",
    (DV_DC, "dividende"): "Le client abandonne les dividendes : plus ils sont élevés, plus la décote grandit.",
    (DV_DC, "dc_maturite"): "Plus c'est long, plus le call vendu et les dividendes abandonnés valent cher : la décote grandit.",
    (DV_DC, "dc_cap"): "Un cap plus bas vend un call plus cher : grosse décote, mais un rendement maximal plus faible.",
}


def dv_tirage(graine, choix):
    """(produit, reglages) tires d'une graine : des marches et des produits realistes."""
    rng = random.Random(graine)
    reg = {"taux": rng.randint(0, 18) * 0.25,           # 0 a 4,5 %
           "dividende": rng.randint(6, 18) * 0.25,      # 1,5 a 4,5 %
           "vol": float(rng.randint(12, 40)),           # 12 a 40 %
           "skew": rng.randint(2, 8) * 0.5,             # 1 a 4 pts
           "spread": rng.choice([0, 25, 50, 75, 100, 150])}
    produit = rng.choice(DV_PRODUITS) if choix not in DV_PRODUITS else choix
    if produit == DV_BRC:
        reg.update(brc_maturite=rng.choice([0.5, 1.0, 1.5, 2.0]),
                   brc_barriere=rng.choice([55, 60, 65, 70, 75, 80, 85]),
                   brc_observation=rng.choice([CONTINUE, CONTINUE, EUROPEENNE]))
    elif produit == DV_CP:
        reg.update(cp_maturite=rng.choice([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                   cp_protection=rng.choice([90, 95, 100]),
                   cp_cap=rng.choice([140, 160, 180, 200, 250, 300]))
    elif produit == DV_AC:
        bar_cp = rng.choice([60, 65, 70, 75, 80])
        reg.update(ac_maturite=rng.choice([2.0, 3.0, 4.0, 5.0]),
                   ac_frequence=rng.choice(list(FREQUENCES)), ac_trigger=100,
                   ac_bar_coupon=bar_cp,
                   ac_bar_capital=rng.choice([b for b in (50, 55, 60, 65, 70) if b <= bar_cp]),
                   ac_memoire=rng.choice([True, False]))
    else:
        reg.update(dc_maturite=rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]),
                   dc_cap=rng.choice([85, 90, 95, 100, 105, 110, 115]))
    return produit, reg


def dv_valeur(produit, reg, bates, marge, recovery, nominal):
    """
    La valeur affichee par la page produit avec ces reglages (en fraction) :
    memes appels au moteur, dans le meme ordre, 5 000 trajectoires, graine 42.
    bates : None (Black-Scholes + skew) ou (theta, kappa, xi, rho, lam, mu_j, sig_j).
    """
    r, q = reg["taux"] / 100, reg["dividende"] / 100
    vol, skew, spread = reg["vol"] / 100, reg["skew"], reg["spread"]

    def chemins(T):
        if bates:
            theta, kappa, xi, rho, lam, mu_j, sig_j = bates
            return M.simule_bates(S0, T, r, q, DV_N, vol ** 2, theta ** 2, kappa, xi, rho,
                                  lam, mu_j, sig_j, seed=42)
        return M.simule_lv(S0, T, r, q, vol, skew, DV_N, seed=42)

    def defauts(n, T):
        return M.tire_defaut(n, T, spread, recovery, 123) if spread > 0 else None

    if produit == DV_BRC:
        T, bar = reg["brc_maturite"], reg["brc_barriere"]
        ch, sg = chemins(T)
        return M.coupon_brc(ch, T, r, nominal, bar / 100, 100 / 100,
                            reg["brc_observation"] == CONTINUE, marge, sg,
                            defauts(len(ch), T), recovery)
    if produit == DV_CP:
        T, protection, cap = reg["cp_maturite"], reg["cp_protection"], reg["cp_cap"]
        zc = protection / 100 * math.exp(-(r + spread / 10000.0) * T)
        budget = 1.0 - zc - marge * T
        v_call = vol if bates else M.vol_au_strike(1.0, vol, skew)
        v_cap = vol if bates else M.vol_au_strike(cap / 100, vol, skew)
        cout = max((M.bs_call(S0, S0, T, r, v_call, q)
                    - M.bs_call(S0, cap / 100 * S0, T, r, v_cap, q)) / S0, 1e-9)
        return max(budget / cout, 0.0)
    if produit == DV_AC:
        T, par_an = reg["ac_maturite"], FREQUENCES[reg["ac_frequence"]]
        n_obs = max(int(T * par_an), 1)
        ch, _ = chemins(T)
        cpn = M.coupon_autocall(ch, T, r, nominal, n_obs, reg["ac_trigger"] / 100,
                                reg["ac_bar_coupon"] / 100, reg["ac_bar_capital"] / 100,
                                reg["ac_memoire"], marge, defauts(len(ch), T), recovery)
        return cpn * par_an
    T, cap = reg["dc_maturite"], reg["dc_cap"]
    if bates:
        return M.discount_mc(chemins(T)[0], T, r, cap / 100, marge)["decote"]
    sigma = M.vol_au_strike(cap / 100, vol, skew)
    return M.discount_certificate(S0, cap / 100 * S0, T, r, sigma, q, marge)["decote"]


def dv_resultat(produit, reg, bates, marge, recovery, nominal):
    """
    La vraie valeur, celle des reglages de reference, et l'effet de chaque
    parametre tire pris seul (on part de la reference et on ne change que lui).
    """
    ref = {**DV_REF_MARCHE, **DV_REF_PRODUIT[produit]}
    v_ref = dv_valeur(produit, ref, bates, marge, recovery, nominal)
    cles = [c for c in DV_MARCHE[produit] if not (bates and c == "skew")]
    cles += [c for c in DV_REF_PRODUIT[produit] if c != "ac_trigger"]
    effets = []
    for c in cles:
        if reg[c] != ref[c]:
            seul = {**ref, c: reg[c]}
            effets.append((c, dv_valeur(produit, seul, bates, marge, recovery, nominal) - v_ref))
    effets.sort(key=lambda x: -abs(x[1]))
    return {"valeur": dv_valeur(produit, reg, bates, marge, recovery, nominal),
            "reference": v_ref, "effets": effets}


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _dv_resultat(produit, reg, bates, marge, recovery, nominal):
    return dv_resultat(produit, reg, bates, marge, recovery, nominal)


def _dv_bates(m):
    return (m.theta_v, m.kappa, m.xi, m.rho, m.lam, m.mu_j, m.sig_j) if m.BATES else None


def _dv_verdict(ecart):
    """Ecart en points de % : dans le mille a 1 point, proche a 3."""
    return "mille" if abs(ecart) <= 1 else ("proche" if abs(ecart) <= 3 else "loin")


def _dv_graine():
    if "ent_dv_graine" not in st.session_state:
        lue = st.query_params.get("ent_dv_graine", "")
        st.session_state["ent_dv_graine"] = int(lue) if lue.isdigit() else random.randrange(10 ** 6)
    return st.session_state["ent_dv_graine"]


def _dv_nouveau():
    st.session_state["ent_dv_graine"] = (_dv_graine() + random.randrange(1, 10 ** 6)) % 10 ** 6
    st.session_state["ent_dv_estimation"] = None
    st.session_state["ent_dv_revele"] = None


def _dv_reveler(cle):
    """Seule la premiere revelation d'un tirage compte : l'estimation est figee."""
    if (st.session_state.get("ent_dv_revele") or {}).get("cle") == cle:
        return
    st.session_state["ent_dv_revele"] = {"cle": cle,
                                         "estimation": st.session_state.get("ent_dv_estimation")}


def _dv_zero():
    st.session_state["ent_dv_scores"] = {}


def _dv_refaire(produit, reg):
    """Avant le rendu des widgets : la barre laterale et la page produit prennent le tirage."""
    for c, v in reg.items():
        st.session_state[c] = v
    st.session_state["trajectoires"] = DV_N
    if produit == DV_BRC:
        st.session_state["brc_strike"] = 100
    elif produit == DV_AC:
        st.session_state.update(ac_step_down=False, ac_indice=CLASSIQUE, ac_sous_jacents=1)
    st.session_state["page"] = DV_PAGES[produit]


def _dv_tableau(reg, cles):
    return [{"Paramètre": DV_NOMS[c], "Valeur": _dv_format(c, reg[c])} for c in cles]


def jeu(m):
    """Le jeu « Devine le coupon »."""
    st.caption("Un produit, un marché tirés au hasard : estime le résultat avant que le "
               "moteur le calcule. C'est l'ordre de grandeur qu'un Sales doit avoir en tête "
               "au téléphone.")
    choix = st.selectbox(
        "Produit à deviner", [DV_HASARD] + DV_PRODUITS, key="ent_dv_produit",
        bind="query-params",
        help="Au hasard : le produit change à chaque tirage. Sinon, le marché et les réglages "
             "du produit choisi sont tirés au hasard, pour s'entraîner sur un seul produit.")
    graine = _dv_graine()
    st.query_params["ent_dv_graine"] = str(graine)
    produit, reg = dv_tirage(graine, choix)
    bates = _dv_bates(m)
    cle = f"{graine}-{DV_PRODUITS.index(produit)}-{int(m.BATES)}"
    quoi = DV_GRANDEUR[produit]

    st.subheader(produit, help="Les conditions sont tirées dans des bornes réalistes : vol 12 "
                 "à 40 %, taux 0 à 4,5 %, dividende 1,5 à 4,5 %, spread 0 à 150 bp, et des "
                 "réglages de produit qu'on voit réellement sur le marché. La marge banque et "
                 "le taux de recouvrement viennent de la barre latérale.")
    marche = [c for c in DV_MARCHE[produit] if not (m.BATES and c == "skew")]
    c1, c2 = st.columns(2)
    c1.markdown("**Le marché**")
    c1.table(_dv_tableau(reg, marche))
    c2.markdown("**Le produit**")
    c2.table(_dv_tableau(reg, list(DV_REF_PRODUIT[produit])))
    if m.BATES:
        st.caption(f"Sous Bates : vol de départ tirée, autres paramètres Bates de la barre "
                   f"latérale. Marge {m.marge*100:.1f} %/an, recouvrement "
                   f"{m.recovery*100:.0f} %, Monte Carlo sur 5 000 trajectoires.")
    else:
        st.caption(f"Sous Black-Scholes + skew (vol locale). Marge {m.marge*100:.1f} %/an, "
                   f"recouvrement {m.recovery*100:.0f} %"
                   + (", formule fermée." if produit in (DV_CP, DV_DC)
                      else ", Monte Carlo sur 5 000 trajectoires."))

    st.markdown(f"### Devine : {quoi}")
    e1, e2 = st.columns([1, 2])
    e1.number_input(
        "Ton estimation (en %)", min_value=0.0, max_value=500.0, value=None, step=0.1,
        format="%.2f", key="ent_dv_estimation", bind="query-params",
        help="Un chiffre en %, sans le signe. Dans le mille à 1 point près, proche à 3 "
             "points.")
    repere = e2.checkbox(
        "Montrer un point de repère", key="ent_dv_repere", bind="query-params",
        help="Affiche la valeur du même produit avec les réglages par défaut de l'appli. Le "
             "réflexe d'un desk : partir d'un chiffre connu et l'ajuster paramètre par "
             "paramètre.")
    b = st.columns(3)
    b[0].button("Révéler", key="ent_dv_reveler", type="primary", on_click=_dv_reveler,
                args=(cle,), help="Le moteur calcule la vraie valeur, l'écart avec ton "
                                  "estimation et le paramètre qui a le plus pesé.")
    b[1].button("Nouveau tirage", key="ent_dv_nouveau", on_click=_dv_nouveau,
                help="Un autre marché et un autre produit (ou le même produit si tu l'as "
                     "choisi au-dessus).")
    b[2].button("Remettre le score à zéro", key="ent_dv_zero", on_click=_dv_zero,
                help="Oublie les tirages joués pendant cette session.")

    revele = st.session_state.get("ent_dv_revele") or {}
    revele = revele if revele.get("cle") == cle else {}
    res = None
    if revele or repere:
        res = _dv_resultat(produit, reg, bates, m.marge, m.recovery, m.nominal)
    if repere:
        st.caption(f"Point de repère : avec les réglages par défaut de l'appli "
                   f"({_dv_texte_ref(produit)}), ce produit vaut "
                   f"{_dv_affiche(produit, res['reference'])}.")

    if revele:
        v = res["valeur"] * 100
        x = revele["estimation"]
        scores = st.session_state.setdefault("ent_dv_scores", {})
        k = st.columns(3)
        k[0].metric("Vraie valeur", _dv_affiche(produit, res["valeur"]))
        if x is None:
            k[1].metric("Ton estimation", "—")
            st.info("Pas d'estimation saisie : ce tirage ne compte pas dans le score.")
        else:
            ecart = x - v
            scores.setdefault(cle, ecart)
            k[1].metric("Ton estimation", f"{x:,.2f}%")
            k[2].metric("Écart", f"{ecart:+,.2f} pts", delta_color="off")
            sens = "trop haut" if ecart > 0 else "trop bas"
            verdict = _dv_verdict(ecart)
            if verdict == "mille":
                st.success(f"Dans le mille : à {abs(ecart):,.2f} point près.")
            elif verdict == "proche":
                st.info(f"Proche : {abs(ecart):,.2f} points {sens}.")
            else:
                st.warning(f"Loin : {abs(ecart):,.2f} points {sens}.")
        st.markdown("**Pourquoi.** " + _dv_pourquoi(produit, reg, res))
        if res["effets"]:
            with st.expander("Paramètre par paramètre"):
                st.table([{"Paramètre": DV_NOMS[c], "Tiré": _dv_format(c, reg[c]),
                           "Référence": _dv_format(c, {**DV_REF_MARCHE,
                                                       **DV_REF_PRODUIT[produit]}[c]),
                           "Effet seul": f"{e*100:+,.2f} pts"} for c, e in res["effets"]])
                st.caption("Chaque effet part des réglages par défaut et ne change que ce "
                           "paramètre. Leur somme n'est pas exactement l'écart total : les "
                           "paramètres interagissent.")
        st.button("Refaire ce produit dans sa page", key="ent_dv_refaire",
                  on_click=_dv_refaire, args=(produit, reg),
                  help="Charge ce marché dans la barre latérale (avec 5 000 trajectoires) et "
                       "ouvre la page du produit avec les mêmes réglages : elle affiche la "
                       "même valeur.")
    else:
        st.info("Écris ton chiffre, puis clique sur « Révéler ».")

    scores = list(st.session_state.get("ent_dv_scores", {}).values())
    st.markdown("---")
    s = st.columns(4)
    s[0].metric("Tirages joués", f"{len(scores)}",
                help="Un tirage compte dès qu'il est révélé avec une estimation. Le score dure "
                     "le temps de la session.")
    s[1].metric("Dans le mille", f"{sum(_dv_verdict(e) == 'mille' for e in scores)}",
                help="Estimations à 1 point ou moins de la vraie valeur.")
    s[2].metric("Proches", f"{sum(_dv_verdict(e) == 'proche' for e in scores)}",
                help="Estimations entre 1 et 3 points de la vraie valeur.")
    s[3].metric("Écart moyen",
                f"{sum(abs(e) for e in scores) / len(scores):,.2f} pts" if scores else "—",
                help="Moyenne des écarts, en valeur absolue, sur les tirages joués.")


def _dv_affiche(produit, v):
    """Le format de la page produit."""
    if produit in (DV_BRC, DV_AC):
        return f"{v*100:,.2f}% p.a."
    if produit == DV_CP:
        return f"{v*100:,.1f}%"
    return f"{v*100:,.2f}%"


def _dv_texte_ref(produit):
    marche = "vol 22 %, taux 3 %, dividende 2,5 %, skew 2, spread nul"
    if produit == DV_BRC:
        return f"1 an, barrière 70 % continue, {marche}"
    if produit == DV_CP:
        return f"5 ans, protection 100 %, sans cap, {marche}"
    if produit == DV_AC:
        return f"3 ans, trimestriel, barrières 70 / 60 %, mémoire, {marche}"
    return f"1 an, cap 100 %, {marche}"


def _dv_pourquoi(produit, reg, res):
    """La phrase chiffree : le parametre qui a le plus pese, et son mecanisme."""
    ref = {**DV_REF_MARCHE, **DV_REF_PRODUIT[produit]}
    texte = (f"Avec les réglages par défaut de l'appli, ce produit vaudrait "
             f"{_dv_affiche(produit, res['reference'])}, contre "
             f"{_dv_affiche(produit, res['valeur'])} ici. ")
    if not res["effets"]:
        return texte + "Ce tirage tombe sur les réglages par défaut."
    c, e = res["effets"][0]
    texte += (f"Ce qui a le plus pesé : {DV_PHRASE[c]}, {_dv_format(c, reg[c])} au lieu "
              f"de {_dv_format(c, ref[c])}. ")
    if e > 0:
        texte += f"À lui seul, ce paramètre ajoute {e*100:,.2f} points {DV_CIBLE[produit]}. "
    else:
        texte += f"À lui seul, ce paramètre retire {-e*100:,.2f} points {DV_CIBLE[produit]}. "
    return texte + DV_POURQUOI[(produit, c)]


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
    st.title("Entraînement", help="**Ce que c'est.** Une banque de questions d'entretien pour un poste de Sales en\nproduits structurés, classées en cinq thèmes. Le plus fourni est le discours client :\nc'est ce qu'un desk teste en premier chez un Sales.\n\n**Comment s'en servir.** Lis la question, réponds à voix haute en trente secondes, puis\naffiche la réponse. Elle est rédigée comme on la dirait en entretien : courte, juste, avec\nle pourquoi financier. La ligne « Ce que le recruteur vérifie » donne l'idée à ne pas\nrater.\n\n**Les démonstrations chiffrées.** Quand la réponse repose sur un mécanisme de prix, le\nmoteur le calcule avec la vue de marché de la barre latérale. Change la vol, les taux, le\nspread ou le modèle : les chiffres suivent, et tu peux citer des ordres de grandeur justes.\n\n**Le compteur.** Une question compte comme vue dès que sa réponse est affichée.\nLe bouton « Question au hasard » privilégie les questions pas encore vues.\n\n**Devine le coupon.** Un jeu d'intuition : l'appli tire un produit et un marché,\ntu estimes le coupon, la participation ou la décote, puis le moteur révèle la vraie\nvaleur, ton écart et le paramètre qui a le plus pesé. Un Sales doit connaître ces\nordres de grandeur sans calculatrice.")
    exercice = st.segmented_control(
        "Exercice", [QUESTIONS_ENTRETIEN, JEU], default=QUESTIONS_ENTRETIEN, required=True,
        key="ent_exercice", bind="query-params",
        help="Questions d'entretien : la banque de questions, réponses rédigées. Devine le "
             "coupon : estimer le résultat d'un produit tiré au hasard, avant le calcul.")
    if exercice == JEU:
        jeu(m)
        return
    st.caption("Réponds à voix haute avant de regarder : c'est la formulation qui se "
               "travaille, pas seulement l'idée.")

    c1, c2 = st.columns([1, 3])
    theme = c1.selectbox(
        "Thème", [TOUS] + THEMES, key="ent_theme", bind="query-params",
        help="Filtre la banque de questions. Le discours client est le thème prioritaire : "
             "savoir adapter un produit au besoin du client et l'expliquer simplement.")
    liste = [i for i, x in enumerate(QUESTIONS) if theme == TOUS or x["theme"] == theme]
    if st.session_state.get("ent_q") not in liste:
        st.session_state["ent_q"] = liste[0]
    q = c2.selectbox(
        "Question", liste, key="ent_q", bind="query-params", format_func=lambda i: f"{i+1}. {QUESTIONS[i]['q']}",
        help="Choisis une question précise, ou laisse les boutons en dessous te guider. "
             "Le numéro est celui de la question dans la banque complète.")

    b = st.columns(4)
    b[0].button("Voir la réponse", key="ent_voir", type="primary", on_click=_revele,
                help="Affiche la réponse rédigée et, si la question s'y prête, une "
                     "démonstration chiffrée calculée avec la vue de marché actuelle.")
    b[1].button("Question suivante", key="ent_suivante", on_click=_suivante, args=(liste,),
                help="Passe à la question suivante du thème choisi, dans l'ordre de la "
                     "banque.")
    b[2].button("Question au hasard", key="ent_hasard", on_click=_hasard, args=(liste,),
                help="Tire une question du thème choisi, en priorité parmi celles dont tu "
                     "n'as pas encore vu la réponse. Le plus proche d'un vrai entretien.")
    b[3].button("Remettre le compteur à zéro", key="ent_zero", on_click=_zero,
                help="Oublie les questions vues pendant cette session.")

    x = QUESTIONS[q]
    st.markdown(f"### {x['q']}")
    st.caption(f"{x['theme']} · question {liste.index(q) + 1} sur {len(liste)}"
               + (" · avec démonstration chiffrée" if x["demo"] else ""))

    if st.session_state.get("ent_revele") == q:
        with st.container(border=True):
            st.markdown(x["r"])
            st.caption("Ce que le recruteur vérifie : " + x["cle"])
        if x["demo"]:
            st.subheader("Démonstration chiffrée",
                         help="Calculée par le moteur de l'appli avec la vue de marché de "
                              "la barre latérale. Les paramètres du produit sont fixes et "
                              "rappelés sous ce titre.")
            m.bandeau_modele()
            x["demo"](m)
    else:
        st.info("Prends trente secondes, réponds à voix haute, puis clique sur « Voir la "
                "réponse ».")

    vues = _vues()
    st.markdown("---")
    k = st.columns(2)
    k[0].metric("Questions vues", f"{len(vues)} / {len(QUESTIONS)}",
                help="Une question compte comme vue dès que sa réponse a été affichée. Le "
                     "compteur dure le temps de la session : il repart à zéro si tu "
                     "recharges la page.")
    k[1].metric("Vues dans ce thème", f"{len(vues & set(liste))} / {len(liste)}",
                help="Progression dans le thème sélectionné.")
