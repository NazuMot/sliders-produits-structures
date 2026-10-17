"""Laboratoire modele."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0
from communs import sim_bates, sim_bs


def affiche(m):
    BATES, kappa, lam, marge, mu_j = m.BATES, m.kappa, m.lam, m.marge, m.mu_j
    n_paths, nominal, q, r, recovery = m.n_paths, m.nominal, m.q, m.r, m.recovery
    rho, sig_j, theta_v, vol_atm, xi = m.rho, m.sig_j, m.theta_v, m.vol_atm, m.xi

    st.title("Laboratoire modele", help="**Ce que tu regardes.** Le meme produit price par deux modeles differents : Black-Scholes a\nvolatilite constante, et Bates (volatilite stochastique + sauts).\n\n**Pourquoi c'est important.** Le choix du modele n'est pas un detail academique. Sur un BRC\na barriere 70%, il change le coupon de pres de 3 points. Black-Scholes sous-estime la\nprobabilite de toucher la barriere, donc sous-estime la valeur du put que vend le client,\ndonc affiche un coupon trop faible.\n\n**Les quatre blocs.**\n1. Le smile : Bates genere un skew realiste, Black-Scholes donne une droite plate.\n2. Les queues de distribution : Bates produit de vrais krachs.\n3. L'effet sur le prix du BRC.\n4. L'effet du spread emetteur sur le coupon.\n\n**A retenir.** Un coupon anormalement eleve doit faire poser deux questions : quel modele\nle desk utilise-t-il, et quelle est la signature de l'emetteur ?")
    st.caption("Pourquoi le choix du modele n'est pas un detail academique : "
               "il change le coupon de plusieurs points.")

    T = st.slider("Maturite (annees)", 0.5, 3.0, 1.0, 0.25)
    bar = st.slider("Barriere du BRC de reference (%)", 50, 90, 70, 5)

    ch_bs, s_bs = sim_bs(S0, T, r, q, vol_atm, n_paths)
    if BATES:
        ch_bt, s_bt = sim_bates(S0, T, r, q, n_paths, vol_atm ** 2, theta_v ** 2,
                                kappa, xi, rho, lam, mu_j, sig_j)
    else:
        ch_bt, s_bt = sim_bates(S0, T, r, q, n_paths, vol_atm ** 2, (vol_atm + 0.02) ** 2,
                                2.0, 0.5, -0.70, 0.3, -0.10, 0.15)
        st.warning("Le modele actif est Black-Scholes. Les colonnes Bates ci-dessous "
                   "utilisent des parametres typiques d'indice actions, a titre de "
                   "comparaison. Bascule le modele dans la barre laterale pour les regler.")

    st.subheader("1. Le smile genere par le modele")
    ks = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
    vols = M.vol_implicite_bates(ch_bt, T, r, q, ks)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    xs = [k * 100 for k in ks]
    ys = [v * 100 for v in vols]
    ax.fill_between(xs, [vol_atm * 100] * len(xs), ys, color=G.PRODUIT, alpha=0.10)
    ax.plot(xs, ys, "o-", lw=2.8, color=G.PRODUIT, ms=7, mfc="white", mew=2,
            label="Volatilite implicite generee par Bates", zorder=3)
    ax.axhline(vol_atm * 100, color=G.DIRECT, ls="--", lw=1.5,
               label="Volatilite unique de Black-Scholes", zorder=2)
    ax.annotate(f"{(vols[0]-vols[3])*100:+.1f} pts\nde skew", (xs[0], ys[0]),
                textcoords="offset points", xytext=(32, -4), fontsize=9.5,
                color=G.PRODUIT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=G.PRODUIT, lw=1.3))
    G.style(ax, "Bates genere le skew, Black-Scholes l'ignore",
            "Les puts bas coutent plus cher en vol que les options a la monnaie",
            "Strike (% du spot)", "Volatilite implicite (%)")
    G.niveau_vertical(ax, 100, "a la monnaie")
    leg = ax.legend(fontsize=9.5, loc="upper right", frameon=True, framealpha=0.95,
                    edgecolor="#E0E0E0")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    pente = (vols[0] - vols[3]) * 100
    st.markdown(
        f"Pente 70 % / 100 % : **{pente:+.1f} points de vol**. "
        "Black-Scholes donne une droite plate — il price le put a 70 % au meme prix "
        "relatif que l'option a la monnaie, ce que le marche ne fait jamais.")

    st.subheader("2. Queues de distribution")
    st_bs = ch_bs[:, -1].astype(float)
    st_bt = ch_bt[:, -1].astype(float)
    c = st.columns(4)
    c[0].metric("P(S < 60) Black-Scholes", f"{(st_bs < 60).mean()*100:,.2f}%")
    c[1].metric("P(S < 60) Bates", f"{(st_bt < 60).mean()*100:,.2f}%")
    c[2].metric("1er centile BS", f"{np.percentile(st_bs, 1):,.1f}")
    c[3].metric("1er centile Bates", f"{np.percentile(st_bt, 1):,.1f}")

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bins = np.linspace(20, 200, 120)
    ax.hist(st_bs, bins=bins, alpha=0.45, color=G.DIRECT,
            label="Black-Scholes", density=True)
    ax.hist(st_bt, bins=bins, alpha=0.55, color=G.PRODUIT,
            label="Bates", density=True)
    ax.axvspan(20, bar, color=G.FOND_PERTE, alpha=0.08, zorder=0)
    ax.set_xlim(20, 200)
    G.style(ax, "Bates produit des krachs, Black-Scholes presque pas",
            f"Sous la barriere : {(st_bs < bar).mean()*100:,.1f}% des scenarios en "
            f"Black-Scholes contre {(st_bt < bar).mean()*100:,.1f}% en Bates",
            "Sous-jacent a l'echeance", "Densite de probabilite")
    G.niveau_vertical(ax, bar, f"barriere  {bar:,.0f}", G.PERTE)
    leg = ax.legend(fontsize=9.5, loc="upper right", frameon=True, framealpha=0.95,
                    edgecolor="#E0E0E0")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("3. Effet sur le prix")
    lignes = []
    for nom, ch_, s_ in [("Black-Scholes, vol plate", ch_bs, vol_atm),
                         ("Bates (vol stochastique + sauts)", ch_bt, s_bt)]:
        pn = M.proba_non_touche(ch_, bar / 100 * S0, s_, T)
        cp = M.coupon_brc(ch_, T, r, nominal, bar / 100, 1.0, True, marge, s_)
        lignes.append({"Modele": nom,
                       "Proba de toucher": f"{(1-pn.mean())*100:,.1f}%",
                       "Coupon equitable": f"{cp*100:,.2f}% p.a."})
    st.table(lignes)

    st.error(
        "**C'est le point le plus important de cette appli.** Black-Scholes sous-estime "
        "la probabilite de franchir la barriere, donc sous-estime la valeur du put que le "
        "client vend, donc affiche un coupon trop faible. Un desk qui pricerait en "
        "Black-Scholes plat vendrait systematiquement trop cher pour le client et se ferait "
        "arbitrer par la concurrence.\n\n"
        "Inversement, un Sales qui recoit un coupon nettement au-dessus du marche doit se "
        "demander ce que le modele du desk suppose, ou quel est le spread de l'emetteur.")

    st.subheader("4. Spread emetteur")
    lignes = []
    for sp in [0, 50, 100, 200, 400]:
        tau = M.tire_defaut(len(ch_bt), T, sp, recovery)
        cp = M.coupon_brc(ch_bt, T, r, nominal, bar / 100, 1.0, True, marge, s_bt,
                          tau, recovery)
        lignes.append({"Spread emetteur": f"{sp} bp",
                       "Proba de defaut sur la periode": f"{(tau <= T).mean()*100:,.1f}%",
                       "Coupon": f"{cp*100:,.2f}% p.a."})
    st.table(lignes)
    st.info(
        "**Le contre-intuitif a retenir** : le spread emetteur augmente le coupon. La "
        "banque se finance moins cher via le produit structure que sur le marche "
        "obligataire, et reverse l'economie au client.\n\n"
        "**Ce que ca implique en clientele** : comparer deux coupons de deux emetteurs "
        "differents sans regarder leur signature n'a aucun sens. En 2008, les porteurs de "
        "produits Lehman ont recupere environ 9 cents par dollar, barrieres intactes ou non.")
