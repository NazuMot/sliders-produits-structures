"""2 - Capital Protection."""

import math
import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0


def affiche(m):
    BATES, bandeau_modele, marge, nominal, q = m.BATES, m.bandeau_modele, m.marge, m.nominal, m.q
    r, skew, spread, trajectoires, vol_atm = m.r, m.skew, m.spread, m.trajectoires, m.vol_atm

    st.title("Capital Protection Certificate", help="**Ce que c'est.** Le client recupere au minimum son capital a l'echeance, et participe a une\npartie de la hausse.\n\n**Comment c'est construit.** Une obligation zero-coupon qui reconstitue le capital, plus un\ncall avec ce qui reste du budget.\n\n**Le mecanisme cle.** La participation n'est pas choisie, elle est calculee. La banque place\njuste ce qu'il faut pour rembourser le nominal ; ce qui reste achete des options. Plus les\ntaux sont hauts, plus ce reste est gros, plus la participation monte.\n\n**Pourquoi un client l'achete.** Vue haussiere, mais refus de risquer le capital.\n\n**Ce qu'il faut tester.** Baisse les taux dans la barre laterale et regarde la participation\ns'effondrer. C'est exactement pourquoi ces produits ont quasiment disparu entre 2015 et\n2021, avant de revenir avec la remontee des taux.")
    st.caption("La participation n'est pas un choix commercial. C'est un reste de budget.")
    bandeau_modele()

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 1.0, 10.0, 5.0, 0.5,
                  help="Parametre cle : plus c'est long, moins la banque doit mettre de "
                       "cote, donc plus le budget option est gros.",
                       key="cp_maturite", bind="query-params")
    protection = c2.slider("Niveau de protection (%)", 80, 100, 100, 5,
                           help="90% = le client accepte de perdre 10% maximum. Chaque "
                                "point abandonne remonte la participation.",
                                key="cp_protection", bind="query-params")
    cap = c3.slider("Cap sur la hausse (%)", 110, 300, 300, 10,
                    help="Capper vend un call de strike haut, ce qui libere du budget.",
                    key="cp_cap", bind="query-params")

    # le spread emetteur reduit le cout de la garantie : la banque actualise
    # a son propre taux de financement
    r_fin = r + spread / 10000.0
    zc = protection / 100 * math.exp(-r_fin * T)
    budget = 1.0 - zc - marge * T
    v_call = vol_atm if BATES else M.vol_au_strike(1.0, vol_atm, skew)
    v_cap = vol_atm if BATES else M.vol_au_strike(cap / 100, vol_atm, skew)
    cout = max((M.bs_call(S0, S0, T, r, v_call, q)
                - M.bs_call(S0, cap / 100 * S0, T, r, v_cap, q)) / S0, 1e-9)
    part = max(budget / cout, 0.0)

    m = st.columns(4)
    m[0].metric("Cout de la garantie", f"{zc*nominal:,.0f}")
    m[1].metric("Budget option", f"{budget*nominal:,.0f}")
    m[2].metric("Cout du call spread", f"{cout*nominal:,.0f}")
    m[3].metric("Participation offerte", f"{part*100:,.0f}%")

    if part < 0.35:
        st.error("Participation sous 35% : invendable. Baisse la protection, allonge la "
                 "maturite, ou mets un cap.")
    elif part > 1.0:
        st.success("Participation au-dessus de 100% : configuration de taux favorable.")

    if spread > 0:
        st.caption(f"La garantie est actualisee au taux de financement de la banque "
                   f"({r*100:,.2f}% + {spread} bp), pas au taux sans risque. C'est ce qui "
                   "permet a un emetteur moins bien note d'offrir une meilleure "
                   "participation — au prix d'un risque de credit que le client porte.")

    spots = np.linspace(30, 220, 500)
    perf = np.minimum(spots / S0 - 1, cap / 100 - 1)
    pay = nominal * (protection / 100 + part * np.maximum(perf, 0))
    direct = nominal * spots / S0
    niv = [(S0, f"depart  {S0:,.0f}")]
    if cap < 300:
        niv.append((cap / 100 * S0, f"cap  {cap:,.0f}"))
    graphe = G.payoff(
        spots, pay, direct, nominal, "Ce que touche le client",
        titre=f"Plancher a {protection}% du nominal, {part*100:,.0f}% de la hausse",
        sous_titre="Le client est protege a la baisse et abandonne une partie de la "
                   "hausse : c'est le meme arbitrage vu des deux cotes",
        niveaux=niv,
        echantillon=trajectoires(T, 1.0)[0][:, -1],
        niveaux_h=[(nominal * protection / 100,
                    f"plancher  {nominal*protection/100:,.0f}", G.GAIN)],
        legende_zones=True)
    G.affiche(graphe)

    st.subheader("Sensibilite aux taux")
    lignes = []
    for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        z = protection / 100 * math.exp(-(t + spread / 10000.0) * T)
        b = 1 - z - marge * T
        c_ = max((M.bs_call(S0, S0, T, t, v_call, q)
                  - M.bs_call(S0, cap / 100 * S0, T, t, v_cap, q)) / S0, 1e-9)
        lignes.append({"Taux": f"{t*100:.0f}%", "Mis de cote": f"{z*nominal:,.0f}",
                       "Budget option": f"{b*nominal:,.0f}",
                       "Participation": f"{max(b/c_, 0)*100:,.0f}%"})
    st.table(lignes)

    st.info(
        "**Le mecanisme** : la banque doit rembourser le nominal a l'echeance. Elle place "
        "juste ce qu'il faut pour y arriver. Ce qui reste achete un call.\n\n"
        "**A savoir dire en entretien** : taux hauts = capital garanti vendable. Taux a "
        "zero = participation ridicule, le marche bascule vers BRC et autocalls. "
        "Mouvement observe entre 2015 et 2021, puis l'inverse depuis.")
