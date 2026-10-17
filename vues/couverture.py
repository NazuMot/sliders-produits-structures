"""Cout de couverture."""

import math
import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0
from communs import MAX_CACHE


def affiche(m):
    nominal, q, r, vol_atm = m.nominal, m.q, m.r, m.vol_atm

    st.title("Cout de couverture", help="**Ce que tu regardes.** Une simulation de ce que fait reellement le trader : couvrir sa\nposition en delta, avec un rebalancement discret et des frais a chaque transaction.\n\n**Pourquoi c'est la.** En theorie de Black-Scholes, une couverture continue et gratuite\ndonne un resultat exactement nul. Dans la realite il y a deux fuites : on ne rebalance\npas en continu, et chaque transaction coute.\n\n**Les trois sources du resultat du trader.**\n1. L'ecart entre la volatilite a laquelle il a achete l'option et celle qui se realise.\n2. L'erreur liee au rebalancement discret.\n3. Les frais de transaction, proportionnels au gamma.\n\n**Le graphe de droite.** Couvrir plus souvent reduit le risque mais coute plus cher en\nfrais. Il existe une frequence optimale, et elle depend du cout de transaction.\n\n**Ce que ca t'apprend.** Quand le desk degrade son prix sur un sous-jacent illiquide ou une\nbarriere proche, ce n'est pas de la mauvaise volonte : c'est ce cout-la, et il se mesure.")
    st.caption("Pourquoi la marge n'est pas un simple prelevement : le trader doit "
               "couvrir, et couvrir coute de l'argent.")

    st.markdown(
        "On prend un **Reverse Convertible sans barriere** : le client vend un put "
        "a la monnaie, la banque l'achete. Le trader delta-couvre cette position "
        "jusqu'a l'echeance. En theorie de Black-Scholes, avec une couverture continue "
        "et sans frais, le resultat est exactement nul. On enleve ces deux hypotheses.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 0.25, 2.0, 1.0, 0.25)
    freq_h = c2.select_slider("Rebalancement",
                              ["Mensuel", "Hebdomadaire", "2x par semaine", "Quotidien"],
                              "Hebdomadaire",
                              help="Plus on couvre souvent, plus l'erreur de couverture "
                                   "est faible, mais plus on paie de frais.")
    cout_bp = c3.slider("Cout de transaction (bp par transaction)", 0, 50, 10, 1,
                        help="Fourchette achat-vente plus impact de marche, en points "
                             "de base du montant echange.")

    par_an = {"Mensuel": 12, "Hebdomadaire": 52, "2x par semaine": 104, "Quotidien": 252}[freq_h]
    n_reb = max(int(round(par_an * T)), 4)

    vol_reelle = st.slider(
        "Volatilite realisee (%)", 5.0, 60.0, vol_atm * 100, 1.0,
        help="Ce qui se passe VRAIMENT. La banque a vendu le produit sur la base de la "
             "vol implicite. L'ecart entre les deux est son resultat principal.") / 100

    @st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
    def couverture(T, n_reb, vol_impl, vol_reel, cout_bp, r, q, nominal, n_sim=4000):
        """
        Simule la couverture en delta d'un put vendu au client.

        A chaque date : on recalcule le delta Black-Scholes avec la vol
        IMPLICITE (celle utilisee pour vendre), on ajuste la position en
        actions, on paie les frais, on capitalise le cash.

        Trois sources de resultat pour le trader :
          1. vol implicite > vol realisee : il a vendu cher, il gagne
          2. rebalancement discret : il rate une partie des mouvements,
             ca cree une erreur qui ne se compense pas
          3. les frais de transaction : cout pur, proportionnel au gamma
        """
        dt = T / n_reb
        rng = np.random.default_rng(7)
        z = rng.standard_normal((n_sim // 2, n_reb))
        z = np.vstack([z, -z])
        S = np.empty((z.shape[0], n_reb + 1))
        S[:, 0] = S0
        for t in range(n_reb):
            S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * vol_reel ** 2) * dt
                                           + vol_reel * math.sqrt(dt) * z[:, t])

        n_parts = nominal / S0

        def delta_put(s, tau):
            if tau <= 1e-9:
                return np.where(s < S0, -1.0, 0.0)
            d1 = (np.log(s / S0) + (r - q + 0.5 * vol_impl ** 2) * tau) / (vol_impl * math.sqrt(tau))
            return -math.exp(-q * tau) * (1.0 - M.norm_cdf_vec(d1))

        # la banque est ACHETEUSE du put -> delta du put achete = delta_put
        # pour etre neutre elle detient -delta actions
        prime = M.bs_call(S0, S0, T, r, vol_impl, q) - S0 * math.exp(-q * T) + S0 * math.exp(-r * T)
        cash = np.full(z.shape[0], -prime * n_parts)     # elle a paye la prime
        pos = np.zeros(z.shape[0])
        frais_tot = np.zeros(z.shape[0])

        for t in range(n_reb + 1):
            tau = T - t * dt
            cible = -delta_put(S[:, t], tau) * n_parts
            if t == n_reb:
                cible = np.zeros_like(cible)
            d = cible - pos
            frais = np.abs(d) * S[:, t] * cout_bp / 10000.0
            frais_tot += frais
            cash -= d * S[:, t] + frais
            pos = cible
            if t < n_reb:
                cash *= math.exp(r * dt)
                cash += pos * S[:, t] * (math.exp(q * dt) - 1.0)   # dividendes percus

        payoff_put = np.maximum(S0 - S[:, -1], 0.0) * n_parts      # elle encaisse le put
        pnl = cash + payoff_put
        return pnl, frais_tot, S

    pnl, frais, S = couverture(T, n_reb, vol_atm, vol_reelle, cout_bp, r, q, nominal)

    m = st.columns(4)
    m[0].metric("Resultat moyen du trader", f"{pnl.mean():+,.1f} EUR")
    m[1].metric("Ecart-type du resultat", f"{pnl.std():,.1f} EUR")
    m[2].metric("Frais de transaction moyens", f"{frais.mean():,.1f} EUR")
    m[3].metric("Pire cas sur 5%", f"{np.percentile(pnl, 5):+,.1f} EUR")

    graphe = G.histogramme(
        pnl, 70, "Couvrir ne garantit pas un resultat nul",
        f"{(pnl < 0).mean()*100:,.0f}% des scenarios se terminent en perte pour "
        "le trader, malgre une position couverte",
        "Resultat de couverture (EUR)",
        couleur_bin=lambda c: G.GAIN if c >= 0 else G.PERTE,
        niveaux=[(0, "", G.DIRECT), (pnl.mean(), f"moyenne  {pnl.mean():+,.0f}", G.DIRECT)])
    G.affiche(graphe)

    freqs = [12, 26, 52, 104, 252]
    moy, ect = [], []
    for f in freqs:
        p, _, _ = couverture(T, max(int(f * T), 4), vol_atm, vol_reelle, cout_bp,
                             r, q, nominal, 3000)
        moy.append(p.mean())
        ect.append(p.std())
    graphe = G.courbes(
        freqs, [{"nom": "Risque : ecart-type du resultat", "y": ect, "couleur": G.ACCENT,
                 "points": True},
                {"nom": "Resultat moyen (ronge par les frais)", "y": moy,
                 "couleur": G.PRODUIT, "points": True}],
        "Couvrir plus souvent reduit le risque mais coute plus cher",
        "C'est l'arbitrage quotidien du trader, et il depend du cout de transaction",
        "Rebalancements par an", "EUR", nom_x="Rebalancements par an",
        fmt_x=lambda v: f"{v:,.0f} par an", fmt_y=lambda v: G.eur(v, signe=True),
        x_log=True, ticks_x=freqs, x_domaine=(10, 300), zero=True, hauteur=320)
    G.affiche(graphe)

    ecart = (vol_atm - vol_reelle) * 100
    st.info(
        f"**Lecture du resultat.** La banque a achete le put a {vol_atm*100:,.0f}% de vol "
        f"implicite, et le marche a realise {vol_reelle*100:,.0f}%. "
        + (f"Elle a donc paye trop cher de {-ecart:,.0f} points : elle perd sur cette jambe."
           if ecart < 0 else
           f"Elle a paye {ecart:,.0f} points de moins que ce que le marche a realise : "
           "elle perd, car elle est acheteuse de vol et la vol realisee est basse.")
        + "\n\n**L'arbitrage du graphe de droite** : couvrir plus souvent reduit le risque "
          "(courbe orange qui descend) mais coute plus cher en frais (courbe verte qui "
          "descend aussi). Il existe une frequence optimale, et elle depend du cout de "
          "transaction. C'est un vrai sujet de desk.\n\n"
        "**Ce que ca t'apprend en tant que Sales** : quand le desk degrade son prix sur un "
        "sous-jacent illiquide ou une barriere proche du spot, ce n'est pas de la mauvaise "
        "volonte. C'est ce cout-la, et il est mesurable.")

    st.warning(
        "**Limite assumee de cette page** : on couvre ici un put vanille, pas un put a "
        "barriere. Sur un vrai BRC, le gamma explose autour de la barriere et le cout de "
        "couverture est nettement superieur a ce qui est affiche ici. L'ordre de grandeur "
        "et le mecanisme sont justes ; le niveau est un plancher.")
