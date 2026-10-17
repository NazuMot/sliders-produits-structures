"""6 - Twin-Win."""

import math
import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0


def affiche(m):
    bandeau_modele, marge, nominal, r, trajectoires = m.bandeau_modele, m.marge, m.nominal, m.r, m.trajectoires

    st.title("Twin-Win", help="**Ce que c'est.** Le client gagne a la hausse ET a la baisse moderee, tant que la barriere\ntient.\n\n**Comment c'est construit.** Tracker plus achat d'un put down-and-out. Le put transforme la\nbaisse en gain, et s'evapore si la barriere casse.\n\n**Pourquoi un client l'achete.** Il pense que le marche va bouger sans savoir dans quel sens,\nmais sans krach.\n\n**Le point dangereux.** Entre la barriere et le niveau initial, le client gagne. Sous la\nbarriere, il perd d'un coup toute cette zone de gain. La discontinuite est brutale : c'est\nle profil le plus difficile a bien expliquer a un client.")
    st.caption("Gagne a la hausse ET a la baisse moderee, tant que la barriere tient.")
    bandeau_modele()

    c1, c2 = st.columns(2)
    T = c1.slider("Maturite (annees)", 0.5, 4.0, 2.0, 0.25)
    bar = c2.slider("Barriere (% du spot)", 50, 90, 65, 5)

    ch, sg = trajectoires(T, bar / 100)
    ST = ch[:, -1].astype(np.float64)
    p_no = M.proba_non_touche(ch, bar / 100 * S0, sg, T)
    perf_st = ST / S0 - 1

    def pv_twin(pb):
        haut = np.where(perf_st >= 0, nominal * (1 + perf_st),
                        nominal * (1 + pb * (-perf_st)))
        bas = nominal * ST / S0
        return math.exp(-r * T) * (p_no * haut + (1 - p_no) * bas).mean()

    lo, hi, cible = 0.0, 3.0, nominal * (1.0 - marge * T)
    for _ in range(40):
        mid = (lo + hi) / 2
        hi, lo = (mid, lo) if pv_twin(mid) > cible else (hi, mid)
    part_bas = lo

    m = st.columns(3)
    m[0].metric("Participation a la baisse", f"{part_bas*100:,.0f}%")
    m[1].metric("Participation a la hausse", "100%")
    m[2].metric("Proba de toucher", f"{(1-p_no.mean())*100:,.1f}%")

    spots = np.linspace(30, 200, 500)
    perf = spots / S0 - 1
    intact = np.where(perf >= 0, nominal * (1 + perf),
                      nominal * (1 + part_bas * (-perf)))
    direct = nominal * spots / S0
    y_bar = np.interp(bar / 100 * S0, spots, intact)
    graphe = G.payoff(
        spots, intact, direct, nominal, "Si la barriere a tenu",
        titre="La baisse rapporte, jusqu'a la barriere",
        sous_titre=f"Entre {bar/100*S0:,.0f} et {S0:,.0f} le client gagne. Sous la "
                   "barriere il perd d'un coup toute cette zone.",
        niveaux=[(bar / 100 * S0, f"barriere  {bar/100*S0:,.0f}"),
                 (S0, f"depart  {S0:,.0f}")],
        label_direct="Si la barriere a casse (ou detention directe)",
        echantillon=ST, seuil_densite=bar / 100 * S0,
        notes=[(bar / 100 * S0, y_bar, "chute brutale\nsi la barriere casse", G.PERTE, 10, -10)],
        legende_zones=True)
    G.affiche(graphe)

    st.info(
        "**Decomposition** : tracker plus achat d'un put down-and-out. Le put transforme "
        "la baisse en gain et s'evapore si la barriere casse.\n\n"
        "**Le point a ne jamais laisser passer** : entre la barriere et le niveau initial, "
        "le client gagne. Sous la barriere, il perd d'un coup toute cette zone de gain. "
        "La discontinuite est brutale : le profil le plus dangereux a mal expliquer.")
