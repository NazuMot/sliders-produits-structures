"""Comparateur des produits."""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0


def affiche(m):
    BATES, bandeau_modele, defauts, marge, nominal = m.BATES, m.bandeau_modele, m.defauts, m.marge, m.nominal
    q, r, recovery, skew, trajectoires = m.q, m.r, m.recovery, m.skew, m.trajectoires
    vol_atm = m.vol_atm

    st.title("Comparateur", help="**Ce que tu regardes.** Quatre structures pricees sous la meme vue de marche, avec la meme\nmaturite, cote a cote.\n\n**Pourquoi c'est utile.** Un produit structure n'est jamais bon ou mauvais dans l'absolu :\nil est adapte ou non a une vue de marche. Le comparateur montre ce que chaque structure\ndonne et ce qu'elle prend en echange.\n\n**La regle a retenir.** Protection, participation, coupon : on n'en a jamais trois sur trois.\nLe budget est fixe, il ne fait que changer de forme.\n\n**Ce qu'il faut faire.** Bouge la volatilite ou les taux dans la barre laterale et regarde\nquelles offres s'ameliorent et lesquelles se degradent. Elles ne bougent pas toutes dans\nle meme sens.")
    bandeau_modele()

    T = st.slider("Maturite (annees)", 1.0, 5.0, 2.0, 0.5)

    ch_brc, s_brc = trajectoires(T, 0.70)
    ch_ac, s_ac = trajectoires(T, 0.65, seed=43)
    tau_brc = defauts(len(ch_brc), T)
    tau_ac = defauts(len(ch_ac), T)

    zc = math.exp(-r * T)
    budget = 1.0 - zc - marge * T
    vol_call = vol_atm if BATES else M.vol_au_strike(1.0, vol_atm, skew)
    call_u = M.bs_call(S0, S0, T, r, vol_call, q) / S0
    part = max(budget / call_u, 0.0) if call_u > 0 else 0.0

    cpn_brc = M.coupon_brc(ch_brc, T, r, nominal, 0.70, 1.0, True, marge, s_brc,
                           tau_brc, recovery)
    n_obs = max(int(T * 4), 1)
    cpn_ac = M.coupon_autocall(ch_ac, T, r, nominal, n_obs, 1.0, 0.70, 0.65, True,
                               marge, tau_ac, recovery)
    duree, _, p_perte = M.stats_autocall(ch_ac, T, n_obs, 1.0, 0.65)

    st.subheader("Ce que le desk peut offrir")
    st.table([
        {"Produit": "Tracker", "Offre": "Participation 100%", "Protection": "aucune",
         "Plafond hausse": "aucun", "Vue client": "haussier franc"},
        {"Produit": "Capital Protection", "Offre": f"Participation {part*100:,.0f}%",
         "Protection": "capital garanti", "Plafond hausse": "participation reduite",
         "Vue client": "haussier prudent"},
        {"Produit": "Barrier Reverse Conv.", "Offre": f"Coupon {cpn_brc*100:,.2f}% p.a.",
         "Protection": "barriere 70%", "Plafond hausse": f"{cpn_brc*100:,.2f}% p.a.",
         "Vue client": "neutre"},
        {"Produit": "Autocall Phoenix", "Offre": f"Coupon {cpn_ac*100:,.2f}% / trimestre",
         "Protection": "barriere 65% a l'echeance", "Plafond hausse": "coupons cumules",
         "Vue client": "neutre a legerement haussier"},
    ])

    c = st.columns(3)
    c[0].metric("Proba de hausse du sous-jacent",
                f"{(ch_ac[:, -1] > S0).mean()*100:,.1f}%",
                help="Sous mesure risque-neutre. Ce n'est pas une prevision.")
    c[1].metric("Duree de vie moyenne autocall", f"{duree:,.2f} ans")
    c[2].metric("Proba de perte autocall", f"{p_perte*100:,.1f}%")

    spots = np.linspace(30, 190, 400)
    perf = spots / S0 - 1
    direct = nominal * (1 + perf)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(spots, direct, color=G.DIRECT, ls="--", lw=1.5, zorder=4,
            label="Detention directe du sous-jacent")
    ax.plot(spots, nominal * (1 + part * np.maximum(perf, 0)), lw=2.8, color=G.ACCENT,
            zorder=3, label=f"Capital Protection — plancher, {part*100:,.0f}% de la hausse")
    ax.plot(spots, np.where(spots < 70, nominal * spots / S0, nominal)
            + cpn_brc * nominal * T, lw=2.8, color=G.PRODUIT, zorder=3,
            label=f"BRC — coupon {cpn_brc*100:,.2f}%, capital a risque sous 70")
    ax.axhline(nominal, color="#D8D8D8", lw=1.0, zorder=0)
    ax.set_xlim(spots.min(), spots.max())
    G.style(ax, "Trois facons d'exprimer la meme vue de marche",
            "Chaque courbe protege quelque part et abandonne ailleurs",
            "Niveau du sous-jacent a l'echeance", "Remboursement (EUR)")
    G.format_eur(ax)
    G.niveau_vertical(ax, 70, "barriere du BRC  70")
    G.niveau_vertical(ax, S0, f"depart  {S0:,.0f}")
    leg = ax.legend(fontsize=9.5, loc="upper left", frameon=True, framealpha=0.95,
                    edgecolor="#E0E0E0")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "**La regle qui structure tout le metier** : protection, participation, coupon. "
        "Jamais trois sur trois. Le budget est fixe : il vient des taux, des dividendes, "
        "du spread emetteur et de ce que le client accepte de vendre comme optionalite.\n\n"
        "**Ta question au client** : quelle est ta vue, et qu'es-tu pret a abandonner "
        "pour l'exprimer ?")
