"""1 - Tracker Certificate."""

import math
import numpy as np
import streamlit as st

import graphiques as G
from moteur import S0


def affiche(m):
    nominal, q, trajectoires = m.nominal, m.q, m.trajectoires

    st.title("Tracker Certificate", help="**Ce que c'est.** Un produit qui suit le sous-jacent a 100%, à la hausse comme à la baisse.\nPas de protection, pas de coupon.\n\n**Comment c'est construit.** Une simple exposition linéaire. Les dividendes du sous-jacent\nrestent à la banque et financent la structure et les frais.\n\n**Pourquoi un client l'achète.** Pour *l'accès*, pas pour le rendement : un panier\nthématique, un indice sur mesure, un marché difficile d'accès, en une seule ligne de\nportefeuille.\n\n**Ce qu'il faut regarder.** L'écart entre la courbe du produit et la détention directe :\nc'est le coût cumulé des frais et des dividendes abandonnés.\n\n**Le risque à nommer.** Le client est créancier de la banque, il ne détient pas les actions.\nC'est la différence majeure avec un ETF.")
    st.caption("Exposition linéaire. Produit d'accès, pas produit de rendement.")
    G.en_30_secondes(
        "Toute la hausse du sous-jacent, en une seule ligne de portefeuille : un indice, un "
        "panier ou un thème difficile d'accès autrement.",
        "Toute la baisse, sans aucune protection, plus les frais et les dividendes "
        "abandonnés. Et la faillite de la banque : le client ne détient pas les actions.",
        "Un client qui veut un accès simple à un marché ou à un thème, pas un rendement.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturité (années)", 1.0, 10.0, 3.0, 0.5,
                  key="tracker_maturite", bind="query-params")
    participation = c2.slider("Participation (%)", 50, 150, 100, 5,
                              help="100% = 1:1. Au-dessus, on parle d'outperformance "
                                   "certificate : la banque finance le levier en "
                                   "capturant les dividendes.",
                                   key="tracker_participation", bind="query-params")
    frais_pa = c3.slider("Frais annuels (%)", 0.0, 2.0, 0.6, 0.1,
                         key="tracker_frais", bind="query-params")

    spots = np.linspace(30, 180, 400)
    perf = spots / S0 - 1
    payoff = nominal * (1 + participation / 100 * perf) * (1 - frais_pa / 100 * T)

    direct = nominal * spots / S0
    perte_frais = frais_pa / 100 * T * 100
    graphe = G.payoff(
        spots, payoff, direct, nominal, "Ce que touche le client",
        titre=("Le tracker suit le sous-jacent, moins les frais"
               if participation == 100 else
               f"Le tracker suit le sous-jacent a {participation}%"),
        sous_titre=f"Écart permanent de {perte_frais:,.1f}% dû au cumul des frais "
                   f"sur {T:,.1f} ans",
        niveaux=[(S0, f"départ  {S0:,.0f}")],
        echantillon=trajectoires(T, 1.0)[0][:, -1], legende_zones=True)
    G.affiche(graphe)
    haut, bas = (nominal * (1 + participation / 100 * p) * (1 - frais_pa / 100 * T)
                 for p in (0.2, -0.2))
    G.en_clair(
        f"Pour {G.eur(nominal)} investis sur {G.duree(T)} : si le sous-jacent monte de 20 %, "
        f"le client récupère {G.eur(haut)} ; s'il baisse de 20 %, {G.eur(bas)}. En direct, ce "
        f"serait {G.eur(nominal * 1.2)} et {G.eur(nominal * 0.8)} : l'écart vient "
        + ("des frais." if participation == 100 else "des frais et de la participation."))

    st.table([{"Sous-jacent": f"{p:+d}%",
               "Tracker": f"{nominal*(1+participation/100*p/100)*(1-frais_pa/100*T):,.0f}",
               "Direct": f"{nominal*(1+p/100):,.0f}",
               "Écart": f"{nominal*(1+participation/100*p/100)*(1-frais_pa/100*T) - nominal*(1+p/100):+,.0f}"}
              for p in [-40, -20, 0, 20, 40]])

    st.info(
        f"**Le dividende est le nerf du produit.** Le sous-jacent verse {q*100:,.1f}% par "
        f"an que le client ne touche pas : sur {T:,.1f} ans cela représente environ "
        f"{(math.exp(q*T)-1)*100:,.1f}% de rendement abandonné.\n\n"
        "**Le risque à nommer en clientèle** : le client est créancier de la banque. "
        "Différence majeure avec un ETF.")
