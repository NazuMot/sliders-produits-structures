"""1 - Tracker Certificate."""

import math
import numpy as np
import streamlit as st

import graphiques as G
from moteur import S0


def affiche(m):
    nominal, q, trajectoires = m.nominal, m.q, m.trajectoires

    st.title("Tracker Certificate", help="**Ce que c'est.** Un produit qui suit le sous-jacent a 100%, a la hausse comme a la baisse.\nPas de protection, pas de coupon.\n\n**Comment c'est construit.** Une simple exposition lineaire. Les dividendes du sous-jacent\nrestent a la banque et financent la structure et les frais.\n\n**Pourquoi un client l'achete.** Pour *l'acces*, pas pour le rendement : un panier\nthematique, un indice sur mesure, un marche difficile d'acces, en une seule ligne de\nportefeuille.\n\n**Ce qu'il faut regarder.** L'ecart entre la courbe du produit et la detention directe :\nc'est le cout cumule des frais et des dividendes abandonnes.\n\n**Le risque a nommer.** Le client est creancier de la banque, il ne detient pas les actions.\nC'est la difference majeure avec un ETF.")
    st.caption("Exposition lineaire. Produit d'acces, pas produit de rendement.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 1.0, 10.0, 3.0, 0.5)
    participation = c2.slider("Participation (%)", 50, 150, 100, 5,
                              help="100% = 1:1. Au-dessus, on parle d'outperformance "
                                   "certificate : la banque finance le levier en "
                                   "capturant les dividendes.")
    frais_pa = c3.slider("Frais annuels (%)", 0.0, 2.0, 0.6, 0.1)

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
        sous_titre=f"Ecart permanent de {perte_frais:,.1f}% du au cumul des frais "
                   f"sur {T:,.1f} ans",
        niveaux=[(S0, f"depart  {S0:,.0f}")],
        echantillon=trajectoires(T, 1.0)[0][:, -1], legende_zones=True)
    G.affiche(graphe)

    st.table([{"Sous-jacent": f"{p:+d}%",
               "Tracker": f"{nominal*(1+participation/100*p/100)*(1-frais_pa/100*T):,.0f}",
               "Direct": f"{nominal*(1+p/100):,.0f}",
               "Ecart": f"{nominal*(1+participation/100*p/100)*(1-frais_pa/100*T) - nominal*(1+p/100):+,.0f}"}
              for p in [-40, -20, 0, 20, 40]])

    st.info(
        f"**Le dividende est le nerf du produit.** Le sous-jacent verse {q*100:,.1f}% par "
        f"an que le client ne touche pas : sur {T:,.1f} ans cela represente environ "
        f"{(math.exp(q*T)-1)*100:,.1f}% de rendement abandonne.\n\n"
        "**Le risque a nommer en clientele** : le client est creancier de la banque. "
        "Difference majeure avec un ETF.")
