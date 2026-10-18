"""5 - Bonus Certificate."""

import math
import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0


def affiche(m):
    bandeau_modele, marge, nominal, r, trajectoires = m.bandeau_modele, m.marge, m.nominal, m.r, m.trajectoires

    st.title("Bonus Certificate", help="**Ce que c'est.** Un niveau de remboursement minimum (le bonus) tant que la barrière n'est\njamais touchée, plus toute la hausse du sous-jacent sans plafond.\n\n**Comment c'est construit.** Détention du sous-jacent sans dividendes, plus achat d'un put\ndown-and-out. Les dividendes abandonnés financent le put.\n\n**Ce qui le distingue du BRC.** Pas de plafond à la hausse : si le sous-jacent explose, le\nclient suit.\n\n**Pourquoi un client l'achète.** Vue neutre à haussière, avec l'envie de garder le potentiel\nde hausse.\n\n**À ne jamais oublier.** La protection est conditionnelle. Si la barrière casse, le bonus\ndisparaît définitivement et le client se retrouve avec un simple tracker. Ce n'est pas une\ngarantie.")
    st.caption("Un niveau bonus garanti si la barrière tient, hausse illimitée.")
    G.en_30_secondes(
        "Un rendement minimum (le bonus) tant que la barrière n'est jamais touchée, et toute "
        "la hausse du sous-jacent, sans plafond.",
        "Si le sous-jacent passe une seule fois sous la barrière, le bonus disparaît : le "
        "client suit alors toute la baisse. Il renonce aussi aux dividendes.",
        "Un client qui pense que le marché va stagner ou monter, sans vouloir plafonner son "
        "gain.")
    bandeau_modele()

    c1, c2 = st.columns(2)
    T = c1.slider("Maturité (années)", 0.5, 4.0, 2.0, 0.25,
                  key="bonus_maturite", bind="query-params")
    bar = c2.slider("Barrière (% du spot)", 50, 90, 70, 5,
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
    m[1].metric("Rendement si marché latéral", f"{(bonus-1)*100:,.1f}%")
    m[2].metric("Proba de toucher la barrière", f"{(1-p_no.mean())*100:,.1f}%")

    spots = np.linspace(30, 200, 500)
    intact = np.maximum(nominal * spots / S0, nominal * bonus)
    direct = nominal * spots / S0
    graphe = G.payoff(
        spots, intact, direct, nominal, "Si la barrière a tenu",
        titre=f"Plancher à {bonus*100:,.0f}% tant que la barrière tient",
        sous_titre="Si la barrière casse, la courbe rouge disparaît et le client se "
                   "retrouve avec la ligne noire : un simple tracker",
        niveaux=[(bar / 100 * S0, f"barrière  {bar/100*S0:,.0f}"),
                 (S0, f"départ  {S0:,.0f}")],
        label_direct="Si la barrière a cassé (ou détention directe)",
        echantillon=ST, seuil_densite=bar / 100 * S0,
        niveaux_h=[(nominal * bonus, f"bonus  {nominal*bonus:,.0f}", G.GAIN)],
        legende_zones=True)
    G.affiche(graphe)
    G.en_clair(
        f"Pour {G.eur(nominal)} investis sur {G.duree(T)} : si le sous-jacent ne passe jamais "
        f"sous {bar/100*S0:,.0f}, le client récupère au moins {G.eur(nominal * bonus)}, et "
        f"{G.eur(max(nominal * 1.3, nominal * bonus))} si le marché monte de 30 %. S'il passe "
        f"une seule fois dessous, le bonus disparaît : fini à -20 %, il récupère "
        f"{G.eur(nominal * 0.8)}. La barrière casse dans {(1-p_no.mean())*100:,.0f} % des "
        "scénarios simulés.")

    st.info(
        "**Décomposition** : détention du sous-jacent sans dividendes, plus achat d'un put "
        "down-and-out de strike égal au niveau bonus. Les dividendes financent le put.\n\n"
        "**Ce qui le distingue du BRC** : pas de plafond à la hausse.\n\n"
        "**À ne jamais laisser passer** : la protection est conditionnelle et disparaît "
        "si la barrière casse. Ce n'est pas une garantie.")
