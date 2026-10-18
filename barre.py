"""Barre laterale : la vue de marche unique, partagee par toutes les pages."""

from types import SimpleNamespace

import streamlit as st

import moteur as M
from moteur import S0
from communs import sim_bs, sim_bates, sim_defaut


# Conditions de marche typiques, en un clic. Ordres de grandeur pour un indice actions
# europeen : vol d'apres le VSTOXX de data/eurostoxx50.csv (ramenee a une maturite de
# quelques annees, plus basse que la vol 30 jours en crise), taux et dividendes de la zone
# euro (plancher a 0 du curseur quand les taux etaient negatifs), spread d'une banque.
SCENARIOS = {
    "Reglages par defaut": {
        "texte": "Le point de depart de l'appli.",
        "valeurs": {"taux": 3.0, "dividende": 2.5, "vol": 22.0, "skew": 2.0, "vol_lt": 24.0,
                    "spread": 0}},
    "Octobre 2026 (photo Eurex)": {
        "texte": "Taux autour de 2,75 %, vol basse (VSTOXX ~17 %), skew modere.",
        "valeurs": {"taux": 2.75, "dividende": 3.0, "vol": 17.0, "skew": 2.5, "vol_lt": 20.0,
                    "spread": 50}},
    "Marche calme (2017)": {
        "texte": "Taux nuls, vol tres basse (VSTOXX ~15 %) : peu de budget, peu de prime.",
        "valeurs": {"taux": 0.0, "dividende": 3.0, "vol": 15.0, "skew": 3.0, "vol_lt": 18.0,
                    "spread": 50}},
    "Taux zero (2020-2021)": {
        "texte": "Taux nuls ou negatifs : le capital garanti n'a presque plus de budget.",
        "valeurs": {"taux": 0.0, "dividende": 2.5, "vol": 21.0, "skew": 2.5, "vol_lt": 22.0,
                    "spread": 50}},
    "Krach Covid (mars 2020)": {
        "texte": "Vol qui explose (VSTOXX ~55 % en moyenne, 86 % au pic), taux nuls.",
        "valeurs": {"taux": 0.0, "dividende": 3.5, "vol": 45.0, "skew": 4.0, "vol_lt": 25.0,
                    "spread": 125}},
    "Crise financiere (automne 2008)": {
        "texte": "Vol extreme (VSTOXX ~55 %), taux encore eleves, banques sous pression.",
        "valeurs": {"taux": 3.75, "dividende": 5.0, "vol": 45.0, "skew": 4.0, "vol_lt": 25.0,
                    "spread": 300}},
    "Remontee des taux (2022-2023)": {
        "texte": "Les taux remontent vers 3 % : le retour des produits a capital garanti.",
        "valeurs": {"taux": 3.0, "dividende": 3.5, "vol": 24.0, "skew": 3.0, "vol_lt": 24.0,
                    "spread": 75}},
}


def _applique_scenario():
    """
    Ecrit les valeurs du scenario dans les curseurs, avant qu'ils soient redessines.
    Le selecteur revient ensuite a vide : c'est une action, pas un reglage. Ce sont les
    curseurs, relies au lien, qui gardent les valeurs.
    """
    nom = st.session_state.scenario
    if nom is None:
        return
    for cle, valeur in SCENARIOS[nom]["valeurs"].items():
        st.session_state[cle] = valeur
    st.session_state.scenario_applique = nom
    st.session_state.scenario = None


def barre_laterale():
    """Affiche la barre laterale et renvoie tous les reglages de marche."""
    st.sidebar.markdown("---")
    st.sidebar.title("Parametres de marche")

    st.sidebar.selectbox(
        "Conditions de marche", list(SCENARIOS), index=None, key="scenario",
        placeholder="Charger un scenario de marche...", on_change=_applique_scenario,
        help="Charge en un clic des conditions de marche typiques : taux, dividende, "
             "volatilite, skew et spread emetteur. C'est un point de depart : tu peux "
             "ensuite bouger chaque curseur librement. Compare par exemple le capital "
             "garanti en 2021 et en 2023.")
    applique = st.session_state.get("scenario_applique")
    if applique:
        st.sidebar.caption(f"**{applique}** : {SCENARIOS[applique]['texte']} Point de "
                           "depart charge, chaque curseur reste libre.")

    modele = st.sidebar.radio(
        "Modele de pricing", ["Black-Scholes + skew", "Bates (vol stochastique + sauts)"],
        help="Black-Scholes est rapide et suffit pour comprendre les payoffs. Bates est le "
             "standard de marche pour les exotiques actions : il genere le skew au lieu de "
             "le postuler, et produit de vrais krachs. Compare les deux dans l'onglet "
             "Laboratoire modele.", key="modele", bind="query-params")
    BATES = modele.startswith("Bates")

    r = st.sidebar.slider(
        "Taux sans risque (%)", 0.0, 7.0, 3.0, 0.25,
        help="Carburant du budget option. Pour garantir 1 000 EUR dans 5 ans a 3%, la banque "
             "ne met que 861 aujourd'hui ; les 139 restants achetent des options. A 0,5% elle "
             "doit mettre 975 et il ne reste que 25.", key="taux", bind="query-params") / 100

    q = st.sidebar.slider(
        "Taux de dividende (%)", 0.0, 7.0, 2.5, 0.25,
        help="Le client d'un produit structure ne le touche PAS : il reste a la banque et "
             "finance la structure. Source de financement la plus invisible.",
             key="dividende", bind="query-params") / 100

    vol_atm = st.sidebar.slider(
        "Volatilite implicite ATM (%)", 8.0, 70.0, 22.0, 1.0,
        help="Ce n'est pas une prevision : c'est le prix d'une option dans une autre unite. "
             "Sous Bates, c'est la volatilite de depart du processus.",
             key="vol", bind="query-params") / 100

    theta_v = kappa = xi = rho = lam = mu_j = sig_j = None
    if not BATES:
        skew = st.sidebar.slider(
            "Skew (pts de vol / -10% de strike)", 0.0, 6.0, 2.0, 0.5,
            help="Skew impose a la main. Sous Bates ce reglage disparait : le modele le "
                 "genere lui-meme via la correlation spot-vol et les sauts.",
                 key="skew", bind="query-params")
    else:
        skew = 0.0
        with st.sidebar.expander("Parametres Bates", expanded=False):
            st.caption("Valeurs typiques d'un indice actions. Chaque parametre a un effet "
                       "identifiable sur le smile.")
            theta_v = st.slider("Vol long terme (%)", 10.0, 50.0, 24.0, 1.0,
                                help="Niveau vers lequel la volatilite revient.",
                                key="vol_lt", bind="query-params") / 100
            kappa = st.slider("Vitesse de retour a la moyenne", 0.2, 6.0, 2.0, 0.2,
                              help="Plus c'est eleve, plus la vol revient vite a son niveau "
                                   "long terme. Cree la structure par terme.",
                                   key="kappa", bind="query-params")
            xi = st.slider("Vol de la vol", 0.05, 1.5, 0.50, 0.05,
                           help="Donne de la convexite au smile : les options tres hors de "
                                "la monnaie deviennent cheres dans les deux sens.",
                                key="vol_vol", bind="query-params")
            rho = st.slider("Correlation spot / vol", -0.95, 0.0, -0.70, 0.05,
                            help="LE parametre du skew. Negatif = la vol monte quand le "
                                 "marche baisse. Mets-le a 0 et le skew disparait presque "
                                 "entierement.", key="rho", bind="query-params")
            st.markdown("---")
            lam = st.slider("Frequence des sauts (par an)", 0.0, 2.0, 0.3, 0.1,
                            help="0,3 = un saut tous les trois ans en moyenne.",
                            key="sauts", bind="query-params")
            mu_j = st.slider("Taille moyenne d'un saut (%)", -30.0, 5.0, -10.0, 1.0,
                             help="Negatif : les krachs sont des gaps a la baisse.",
                             key="taille_saut", bind="query-params") / 100
            sig_j = st.slider("Dispersion des sauts (%)", 1.0, 40.0, 15.0, 1.0,
                              key="disp_saut", bind="query-params") / 100

    st.sidebar.markdown("---")
    st.sidebar.subheader("Risque emetteur")
    spread = st.sidebar.slider(
        "Spread de credit emetteur (bp)", 0, 500, 0, 25,
        help="Le point contre-intuitif : un spread eleve AUGMENTE le coupon. La banque se "
             "finance moins cher via le produit structure que sur le marche obligataire, et "
             "reverse l'economie au client. Un coupon superieur a la concurrence peut donc "
             "simplement signifier un emetteur moins bien note.",
             key="spread", bind="query-params")
    recovery = st.sidebar.slider(
        "Taux de recouvrement (%)", 0, 80, 40, 5,
        help="Ce que le client recupere si l'emetteur fait defaut. 40% est la convention "
             "senior non securise. Les porteurs de produits Lehman ont recupere environ "
             "9 cents par dollar.", key="recouvrement", bind="query-params") / 100

    marge = st.sidebar.slider("Marge banque (% par an)", 0.0, 2.5, 0.8, 0.1,
                              help="Prelevee sur le budget option, invisible dans le "
                                   "payoff.", key="marge", bind="query-params") / 100

    st.sidebar.markdown("---")
    nominal = float(st.sidebar.select_slider("Nominal (EUR)",
                                             [1000, 10000, 100000, 1000000], 1000,
                                             key="nominal", bind="query-params"))
    n_paths = st.sidebar.select_slider("Trajectoires Monte Carlo",
                                       [5000, 10000, 20000, 40000], 10000,
                                       key="trajectoires", bind="query-params")

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
