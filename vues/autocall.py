"""4 - Autocall Phoenix."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0
from communs import sim_bs, sim_panier


def affiche(m):
    BATES, bandeau_modele, defauts, marge, n_paths = m.BATES, m.bandeau_modele, m.defauts, m.marge, m.n_paths
    nominal, q, r, recovery, skew = m.nominal, m.q, m.r, m.recovery, m.skew
    trajectoires, vol_atm = m.trajectoires, m.vol_atm

    st.title("Autocall Phoenix", help="**Ce que c'est.** Le produit phare des desks. Coupons conditionnels, et remboursement\nanticipe automatique si le sous-jacent est au-dessus d'un niveau a une date\nd'observation.\n\n**Les trois niveaux a distinguer.**\n- **Trigger** : au-dessus, le produit s'arrete et le client est rembourse.\n- **Barriere de coupon** : au-dessus, le coupon est paye.\n- **Barriere de capital** : observee a l'echeance seulement ; en dessous, le client encaisse\n  toute la baisse.\n\n**Le piege commercial.** Le rappel anticipe coupe les coupons futurs. Regarde la duree de\nvie moyenne : le produit dure rarement jusqu'au bout.\n\n**Le worst-of.** Active plusieurs sous-jacents : le plus mauvais decide de tout. Le coupon\ngrimpe, et la probabilite de perte aussi. C'est la version reellement dominante sur les\ndesks.")
    st.caption("Le produit phare des desks.")
    bandeau_modele()

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite maximale (annees)", 1.0, 6.0, 3.0, 0.5,
                  help="Duree MAXIMALE : le produit est presque toujours rappele avant.")
    freq = c1.selectbox("Frequence d'observation",
                        ["Trimestrielle", "Semestrielle", "Annuelle"])
    trigger = c2.slider("Trigger de rappel (%)", 80, 110, 100, 5)
    bar_cp = c2.slider("Barriere de coupon (%)", 50, 100, 70, 5,
                       help="Le coupon n'est paye que si le sous-jacent est au-dessus a "
                            "la date d'observation. C'est ce qui distingue un Phoenix "
                            "d'un BRC.")
    bar_cap = c3.slider("Barriere de capital (%)", 40, 90, 60, 5,
                        help="Observee uniquement A L'ECHEANCE. Le vrai risque du produit.")
    memoire = c3.checkbox("Effet memoire", value=True)

    st.markdown("---")
    w1, w2 = st.columns(2)
    n_actifs = w1.select_slider(
        "Nombre de sous-jacents (worst-of)", [1, 2, 3, 4], 1,
        help="Le plus mauvais des sous-jacents decide de tout. C'est la structure "
             "reellement dominante sur les desks.")
    correl = w2.slider(
        "Correlation entre sous-jacents", 0.0, 0.95, 0.60, 0.05,
        disabled=(n_actifs == 1),
        help="Le parametre de pricing dominant sur un worst-of. Correlation haute : les "
             "actifs bougent ensemble, le panier ressemble a un actif unique, coupon "
             "faible. Correlation basse : il y a presque toujours un trainard, coupon "
             "eleve et risque eleve.")

    par_an = {"Trimestrielle": 4, "Semestrielle": 2, "Annuelle": 1}[freq]
    n_obs = max(int(T * par_an), 1)

    if n_actifs > 1:
        v_panier = vol_atm if BATES else M.vol_au_strike(bar_cap / 100, vol_atm, skew)
        ch, sg = sim_panier(n_actifs, correl, S0, T, r, q, v_panier, n_paths)
        if BATES:
            st.warning("Le worst-of est simule en Black-Scholes multi-actifs : un Bates "
                       "multi-dimensionnel demanderait une matrice de correlation entre "
                       "les volatilites de chaque actif, ce qui depasse le cadre de cet "
                       "outil. Le mecanisme de correlation reste juste.")
    else:
        ch, sg = trajectoires(T, bar_cap / 100)

    tau = defauts(len(ch), T)
    cpn = M.coupon_autocall(ch, T, r, nominal, n_obs, trigger / 100, bar_cp / 100,
                            bar_cap / 100, memoire, marge, tau, recovery)
    duree, p_rappel, p_perte = M.stats_autocall(ch, T, n_obs, trigger / 100, bar_cap / 100)

    m = st.columns(4)
    m[0].metric("Coupon par observation", f"{cpn*100:,.2f}%")
    m[1].metric("Equivalent annuel", f"{cpn*par_an*100:,.2f}% p.a.")
    m[2].metric("Duree de vie moyenne", f"{duree:,.2f} ans")
    m[3].metric("Proba de perte en capital", f"{p_perte*100:,.1f}%")

    indices = M.idx_obs(ch.shape[1] - 1, n_obs)
    tt = np.linspace(0, T, ch.shape[1])

    # --- graphe 1 : trajectoires et rappels
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axhspan(0, bar_cap / 100 * S0, color=G.FOND_PERTE, alpha=0.07, zorder=0)
    n_v = n_r = n_p = 0
    for i in range(60):
        chem = ch[i]
        rappel_a = next((j for j in indices[:-1] if chem[j] >= trigger / 100 * S0), None)
        if rappel_a is not None:
            ax.plot(tt[:rappel_a + 1], chem[:rappel_a + 1], lw=0.9, alpha=0.75,
                    color=G.GAIN, zorder=2,
                    label="Rappele par anticipation" if n_r == 0 else None)
            ax.scatter([tt[rappel_a]], [chem[rappel_a]], s=26, color=G.GAIN,
                       zorder=5, edgecolor="white", lw=0.8)
            n_r += 1
        elif chem[-1] < bar_cap / 100 * S0:
            ax.plot(tt, chem, lw=0.9, alpha=0.8, color=G.PERTE, zorder=3,
                    label="Perte en capital" if n_p == 0 else None)
            n_p += 1
        else:
            ax.plot(tt, chem, lw=0.9, alpha=0.6, color=G.NIVEAU, zorder=1,
                    label="Va au terme sans perte" if n_v == 0 else None)
            n_v += 1
    for j in indices[:-1]:
        ax.axvline(tt[j], color="#DCDCDC", lw=0.9, zorder=0)
    ax.set_xlim(0, T * 1.02)
    G.style(ax, f"Duree de vie moyenne : {duree:,.2f} ans sur {T:,.1f} possibles",
            "Les traits verticaux gris sont les dates d'observation",
            "Temps (annees)", "Worst-of du panier" if n_actifs > 1 else "Sous-jacent")
    G.niveau_horizontal(ax, trigger / 100 * S0,
                        f"trigger de rappel  {trigger/100*S0:,.0f}", G.GAIN)
    G.niveau_horizontal(ax, bar_cp / 100 * S0,
                        f"barriere coupon  {bar_cp/100*S0:,.0f}", G.ACCENT, x=0.62)
    G.niveau_horizontal(ax, bar_cap / 100 * S0,
                        f"barriere capital  {bar_cap/100*S0:,.0f}", G.PERTE)
    leg = ax.legend(fontsize=9.5, loc="upper left", frameon=True, framealpha=0.95,
                    edgecolor="#E0E0E0")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- graphe 2 : quand le produit se termine
    probas, vivant = [], np.ones(len(ch), dtype=bool)
    for j in indices[:-1]:
        rappel = vivant & (ch[:, j] >= trigger / 100 * S0)
        probas.append(rappel.mean())
        vivant = vivant & ~rappel
    probas.append(vivant.mean())
    labels = [f"Obs {k+1}" for k in range(len(probas) - 1)] + ["Terme"]
    couleurs = [G.GAIN] * (len(probas) - 1) + [G.NIVEAU]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    barres = ax.bar(labels, np.array(probas) * 100, color=couleurs, width=0.68)
    for b, p in zip(barres, probas):
        if p > 0.012:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{p*100:,.0f}%", ha="center", fontsize=9,
                    color="#444444", fontweight="bold")
    ax.set_ylim(0, max(probas) * 118)
    G.style(ax, f"{(1-probas[-1])*100:,.0f}% des scenarios se terminent avant l'echeance",
            "Chaque rappel anticipe coupe tous les coupons restants",
            None, "Probabilite (%)")
    ax.tick_params(axis="x", rotation=45 if len(labels) > 8 else 0, labelsize=8.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if st.checkbox("Effet du nombre de sous-jacents"):
        v_p = vol_atm if BATES else M.vol_au_strike(bar_cap / 100, vol_atm, skew)
        lignes = []
        for na in [1, 2, 3, 4]:
            for co in ([0.0] if na == 1 else [0.80, 0.50, 0.25]):
                if na == 1:
                    c_, _ = sim_bs(S0, T, r, q, v_p, min(n_paths, 20000))
                else:
                    c_, _ = sim_panier(na, co, S0, T, r, q, v_p, min(n_paths, 20000))
                cp = M.coupon_autocall(c_, T, r, nominal, n_obs, trigger / 100,
                                       bar_cp / 100, bar_cap / 100, memoire, marge)
                _, _, pp = M.stats_autocall(c_, T, n_obs, trigger / 100, bar_cap / 100)
                lignes.append({"Sous-jacents": str(na),
                               "Correlation": "-" if na == 1 else f"{co:.2f}",
                               "Coupon annuel": f"{cp*par_an*100:,.2f}%",
                               "Proba de perte": f"{pp*100:,.1f}%"})
        st.table(lignes)
        st.caption("Lis les deux colonnes ensemble. Le coupon double entre un mono "
                   "sous-jacent et un worst-of decorrele — et la probabilite de perte "
                   "aussi. Un client qui ne regarde que le coupon ne voit qu'une moitie "
                   "du produit.")

    st.info(
        f"**Duree de vie moyenne : {duree:,.2f} ans sur {T:,.1f} possibles.** Le rappel "
        "anticipe coupe les coupons futurs. Un client qui compare le coupon affiche a un "
        "rendement obligataire se trompe : il ne le touchera peut-etre qu'une ou deux fois."
        + ("\n\n**Sur un worst-of, le coupon spectaculaire a une contrepartie simple** : "
           "il faut que TOUS les sous-jacents tiennent, pas un seul. Plus ils sont "
           "decorreles, plus il est probable qu'au moins un decroche."
           if n_actifs > 1 else ""))
