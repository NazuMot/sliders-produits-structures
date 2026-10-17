"""5 - Bonus Certificate."""

import math
import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0


def affiche(m):
    bandeau_modele, marge, nominal, r, trajectoires = m.bandeau_modele, m.marge, m.nominal, m.r, m.trajectoires

    st.title("Bonus Certificate", help="**Ce que c'est.** Un niveau de remboursement minimum (le bonus) tant que la barriere n'est\njamais touchee, plus toute la hausse du sous-jacent sans plafond.\n\n**Comment c'est construit.** Detention du sous-jacent sans dividendes, plus achat d'un put\ndown-and-out. Les dividendes abandonnes financent le put.\n\n**Ce qui le distingue du BRC.** Pas de plafond a la hausse : si le sous-jacent explose, le\nclient suit.\n\n**Pourquoi un client l'achete.** Vue neutre a haussiere, avec l'envie de garder le potentiel\nde hausse.\n\n**A ne jamais oublier.** La protection est conditionnelle. Si la barriere casse, le bonus\ndisparait definitivement et le client se retrouve avec un simple tracker. Ce n'est pas une\ngarantie.")
    st.caption("Un niveau bonus garanti si la barriere tient, hausse illimitee.")
    bandeau_modele()

    c1, c2 = st.columns(2)
    T = c1.slider("Maturite (annees)", 0.5, 4.0, 2.0, 0.25,
                  key="bonus_maturite", bind="query-params")
    bar = c2.slider("Barriere (% du spot)", 50, 90, 70, 5,
                    key="bonus_barriere", bind="query-params")

    ch, sg = trajectoires(T, bar / 100)
    ST = ch[:, -1].astype(np.float64)
    p_no = M.proba_non_touche(ch, bar / 100 * S0, sg, T)

    def pv_bonus(niveau):
        haut = np.maximum(nominal * ST / S0, nominal * niveau)
        bas = nominal * ST / S0
        return math.exp(-r * T) * (p_no * haut + (1 - p_no) * bas).mean()

    lo, hi, cible = 1.0, 2.5, nominal * (1.0 - marge * T)
    for _ in range(40):
        mid = (lo + hi) / 2
        hi, lo = (mid, lo) if pv_bonus(mid) > cible else (hi, mid)
    bonus = lo

    m = st.columns(3)
    m[0].metric("Niveau bonus offert", f"{bonus*100:,.1f}%")
    m[1].metric("Rendement si marche lateral", f"{(bonus-1)*100:,.1f}%")
    m[2].metric("Proba de toucher la barriere", f"{(1-p_no.mean())*100:,.1f}%")

    spots = np.linspace(30, 200, 500)
    intact = np.maximum(nominal * spots / S0, nominal * bonus)
    direct = nominal * spots / S0
    graphe = G.payoff(
        spots, intact, direct, nominal, "Si la barriere a tenu",
        titre=f"Plancher a {bonus*100:,.0f}% tant que la barriere tient",
        sous_titre="Si la barriere casse, la courbe rouge disparait et le client se "
                   "retrouve avec la ligne noire : un simple tracker",
        niveaux=[(bar / 100 * S0, f"barriere  {bar/100*S0:,.0f}"),
                 (S0, f"depart  {S0:,.0f}")],
        label_direct="Si la barriere a casse (ou detention directe)",
        echantillon=ST, seuil_densite=bar / 100 * S0,
        niveaux_h=[(nominal * bonus, f"bonus  {nominal*bonus:,.0f}", G.GAIN)],
        legende_zones=True)
    G.affiche(graphe)

    st.info(
        "**Decomposition** : detention du sous-jacent sans dividendes, plus achat d'un put "
        "down-and-out de strike egal au niveau bonus. Les dividendes financent le put.\n\n"
        "**Ce qui le distingue du BRC** : pas de plafond a la hausse.\n\n"
        "**A ne jamais laisser passer** : la protection est conditionnelle et disparait "
        "si la barriere casse. Ce n'est pas une garantie.")
