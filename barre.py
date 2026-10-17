"""Barre laterale : la vue de marche unique, partagee par toutes les pages."""

from types import SimpleNamespace

import streamlit as st

import moteur as M
from moteur import S0
from communs import sim_bs, sim_bates, sim_defaut


def barre_laterale():
    """Affiche la barre laterale et renvoie tous les reglages de marche."""
    st.sidebar.markdown("---")
    st.sidebar.title("Parametres de marche")

    modele = st.sidebar.radio(
        "Modele de pricing", ["Black-Scholes + skew", "Bates (vol stochastique + sauts)"],
        help="Black-Scholes est rapide et suffit pour comprendre les payoffs. Bates est le "
             "standard de marche pour les exotiques actions : il genere le skew au lieu de "
             "le postuler, et produit de vrais krachs. Compare les deux dans l'onglet "
             "Laboratoire modele.")
    BATES = modele.startswith("Bates")

    r = st.sidebar.slider(
        "Taux sans risque (%)", 0.0, 7.0, 3.0, 0.25,
        help="Carburant du budget option. Pour garantir 1 000 EUR dans 5 ans a 3%, la banque "
             "ne met que 861 aujourd'hui ; les 139 restants achetent des options. A 0,5% elle "
             "doit mettre 975 et il ne reste que 25.") / 100

    q = st.sidebar.slider(
        "Taux de dividende (%)", 0.0, 7.0, 2.5, 0.25,
        help="Le client d'un produit structure ne le touche PAS : il reste a la banque et "
             "finance la structure. Source de financement la plus invisible.") / 100

    vol_atm = st.sidebar.slider(
        "Volatilite implicite ATM (%)", 8.0, 70.0, 22.0, 1.0,
        help="Ce n'est pas une prevision : c'est le prix d'une option dans une autre unite. "
             "Sous Bates, c'est la volatilite de depart du processus.") / 100

    theta_v = kappa = xi = rho = lam = mu_j = sig_j = None
    if not BATES:
        skew = st.sidebar.slider(
            "Skew (pts de vol / -10% de strike)", 0.0, 6.0, 2.0, 0.5,
            help="Skew impose a la main. Sous Bates ce reglage disparait : le modele le "
                 "genere lui-meme via la correlation spot-vol et les sauts.")
    else:
        skew = 0.0
        with st.sidebar.expander("Parametres Bates", expanded=False):
            st.caption("Valeurs typiques d'un indice actions. Chaque parametre a un effet "
                       "identifiable sur le smile.")
            theta_v = st.slider("Vol long terme (%)", 10.0, 50.0, 24.0, 1.0,
                                help="Niveau vers lequel la volatilite revient.") / 100
            kappa = st.slider("Vitesse de retour a la moyenne", 0.2, 6.0, 2.0, 0.2,
                              help="Plus c'est eleve, plus la vol revient vite a son niveau "
                                   "long terme. Cree la structure par terme.")
            xi = st.slider("Vol de la vol", 0.05, 1.5, 0.50, 0.05,
                           help="Donne de la convexite au smile : les options tres hors de "
                                "la monnaie deviennent cheres dans les deux sens.")
            rho = st.slider("Correlation spot / vol", -0.95, 0.0, -0.70, 0.05,
                            help="LE parametre du skew. Negatif = la vol monte quand le "
                                 "marche baisse. Mets-le a 0 et le skew disparait presque "
                                 "entierement.")
            st.markdown("---")
            lam = st.slider("Frequence des sauts (par an)", 0.0, 2.0, 0.3, 0.1,
                            help="0,3 = un saut tous les trois ans en moyenne.")
            mu_j = st.slider("Taille moyenne d'un saut (%)", -30.0, 5.0, -10.0, 1.0,
                             help="Negatif : les krachs sont des gaps a la baisse.") / 100
            sig_j = st.slider("Dispersion des sauts (%)", 1.0, 40.0, 15.0, 1.0) / 100

    st.sidebar.markdown("---")
    st.sidebar.subheader("Risque emetteur")
    spread = st.sidebar.slider(
        "Spread de credit emetteur (bp)", 0, 500, 0, 25,
        help="Le point contre-intuitif : un spread eleve AUGMENTE le coupon. La banque se "
             "finance moins cher via le produit structure que sur le marche obligataire, et "
             "reverse l'economie au client. Un coupon superieur a la concurrence peut donc "
             "simplement signifier un emetteur moins bien note.")
    recovery = st.sidebar.slider(
        "Taux de recouvrement (%)", 0, 80, 40, 5,
        help="Ce que le client recupere si l'emetteur fait defaut. 40% est la convention "
             "senior non securise. Les porteurs de produits Lehman ont recupere environ "
             "9 cents par dollar.") / 100

    marge = st.sidebar.slider("Marge banque (% par an)", 0.0, 2.5, 0.8, 0.1,
                              help="Prelevee sur le budget option, invisible dans le "
                                   "payoff.") / 100

    st.sidebar.markdown("---")
    nominal = float(st.sidebar.select_slider("Nominal (EUR)",
                                             [1000, 10000, 100000, 1000000], 1000))
    n_paths = st.sidebar.select_slider("Trajectoires Monte Carlo",
                                       [5000, 10000, 20000, 40000], 10000)

    st.sidebar.caption("Sous-jacent initial fixe a 100.")


    # --------------------------------------------------------------------
    def trajectoires(T, niveau_ref=1.0, seed=42):
        """
        Renvoie (chemins, sigma) selon le modele choisi.
        niveau_ref sert uniquement en Black-Scholes, pour appliquer le skew
        au strike pertinent du produit.
        """
        if BATES:
            return sim_bates(S0, T, r, q, n_paths, vol_atm ** 2, theta_v ** 2, kappa,
                             xi, rho, lam, mu_j, sig_j, seed)
        v = M.vol_au_strike(niveau_ref, vol_atm, skew)
        return sim_bs(S0, T, r, q, v, n_paths, seed)


    def defauts(n, T):
        return sim_defaut(n, T, spread, recovery) if spread > 0 else None




    def bandeau_modele():
        if BATES:
            st.caption(f"Pricing sous Bates — vol de depart {vol_atm*100:.0f}%, "
                       f"correlation spot/vol {rho:+.2f}, {lam:.1f} saut(s) par an."
                       + (f" Spread emetteur {spread} bp." if spread else ""))
        else:
            st.caption(f"Pricing sous Black-Scholes — vol ATM {vol_atm*100:.0f}%, "
                       f"skew {skew:.1f} pt/10%."
                       + (f" Spread emetteur {spread} bp." if spread else ""))

    return SimpleNamespace(**locals())
