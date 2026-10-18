"""Barre laterale : la vue de marche unique, partagee par toutes les pages."""

from types import SimpleNamespace

import streamlit as st

from moteur import S0
from communs import sim_lv, sim_bates, sim_defaut


# Conditions de marche typiques, en un clic. Ordres de grandeur pour un indice actions
# europeen : vol d'apres le VSTOXX de data/eurostoxx50.csv (ramenee a une maturite de
# quelques annees, plus basse que la vol 30 jours en crise), taux et dividendes de la zone
# euro (plancher a 0 du curseur quand les taux etaient negatifs), spread d'une banque.
SCENARIOS = {
    "Réglages par défaut": {
        "texte": "Le point de départ de l'appli.",
        "valeurs": {"taux": 3.0, "dividende": 2.5, "vol": 22.0, "skew": 2.0, "vol_lt": 24.0,
                    "spread": 0}},
    "Octobre 2026 (photo Eurex)": {
        "texte": "Taux autour de 2,75 %, vol basse (VSTOXX ~17 %), skew modéré.",
        "valeurs": {"taux": 2.75, "dividende": 3.0, "vol": 17.0, "skew": 2.5, "vol_lt": 20.0,
                    "spread": 50}},
    "Marché calme (2017)": {
        "texte": "Taux nuls, vol très basse (VSTOXX ~15 %) : peu de budget, peu de prime.",
        "valeurs": {"taux": 0.0, "dividende": 3.0, "vol": 15.0, "skew": 3.0, "vol_lt": 18.0,
                    "spread": 50}},
    "Taux zéro (2020-2021)": {
        "texte": "Taux nuls ou négatifs : le capital garanti n'a presque plus de budget.",
        "valeurs": {"taux": 0.0, "dividende": 2.5, "vol": 21.0, "skew": 2.5, "vol_lt": 22.0,
                    "spread": 50}},
    "Krach Covid (mars 2020)": {
        "texte": "Vol qui explose (VSTOXX ~55 % en moyenne, 86 % au pic), taux nuls.",
        "valeurs": {"taux": 0.0, "dividende": 3.5, "vol": 45.0, "skew": 4.0, "vol_lt": 25.0,
                    "spread": 125}},
    "Crise financière (automne 2008)": {
        "texte": "Vol extrême (VSTOXX ~55 %), taux encore élevés, banques sous pression.",
        "valeurs": {"taux": 3.75, "dividende": 5.0, "vol": 45.0, "skew": 4.0, "vol_lt": 25.0,
                    "spread": 300}},
    "Remontée des taux (2022-2023)": {
        "texte": "Les taux remontent vers 3 % : le retour des produits à capital garanti.",
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
    st.sidebar.title("Paramètres de marché")

    st.sidebar.selectbox(
        "Conditions de marché", list(SCENARIOS), index=None, key="scenario",
        placeholder="Charger un scénario de marché...", on_change=_applique_scenario,
        help="Charge en un clic des conditions de marché typiques : taux, dividende, "
             "volatilité, skew et spread émetteur. C'est un point de départ : tu peux "
             "ensuite bouger chaque curseur librement. Compare par exemple le capital "
             "garanti en 2021 et en 2023.")
    applique = st.session_state.get("scenario_applique")
    if applique:
        st.sidebar.caption(f"**{applique}** : {SCENARIOS[applique]['texte']} Point de "
                           "départ chargé, chaque curseur reste libre.")

    modele = st.sidebar.radio(
        "Modèle de pricing", ["Black-Scholes + skew", "Bates (vol stochastique + sauts)"],
        help="Black-Scholes est rapide et suffit pour comprendre les payoffs. Bates est le "
             "standard de marché pour les exotiques actions : il génère le skew au lieu de "
             "le postuler, et produit de vrais krachs. Compare les deux dans l'onglet "
             "Laboratoire modèle.", key="modele", bind="query-params")
    BATES = modele.startswith("Bates")

    r = st.sidebar.slider(
        "Taux sans risque (%)", 0.0, 7.0, 3.0, 0.25,
        help="Carburant du budget option. Pour garantir 1 000 EUR dans 5 ans à 3%, la banque "
             "ne met que 861 aujourd'hui ; les 139 restants achètent des options. À 0,5% elle "
             "doit mettre 975 et il ne reste que 25.", key="taux", bind="query-params") / 100

    q = st.sidebar.slider(
        "Taux de dividende (%)", 0.0, 7.0, 2.5, 0.25,
        help="Le client d'un produit structuré ne le touche PAS : il reste à la banque et "
             "finance la structure. Source de financement la plus invisible.",
             key="dividende", bind="query-params") / 100

    vol_atm = st.sidebar.slider(
        "Volatilité implicite ATM (%)", 8.0, 70.0, 22.0, 1.0,
        help="Ce n'est pas une prévision : c'est le prix d'une option dans une autre unité. "
             "Sous Bates, c'est la volatilité de départ du processus.",
             key="vol", bind="query-params") / 100

    theta_v = kappa = xi = rho = lam = mu_j = sig_j = None
    if not BATES:
        skew = st.sidebar.slider(
            "Skew (pts de vol / -10% de strike)", 0.0, 6.0, 2.0, 0.5,
            help="Skew imposé à la main. Sous Bates ce réglage disparaît : le modèle le "
                 "génère lui-même via la corrélation spot-vol et les sauts.",
                 key="skew", bind="query-params")
    else:
        skew = 0.0
        with st.sidebar.expander("Paramètres Bates", expanded=False):
            st.caption("Valeurs typiques d'un indice actions. Chaque paramètre a un effet "
                       "identifiable sur le smile.")
            theta_v = st.slider("Vol long terme (%)", 10.0, 50.0, 24.0, 1.0,
                                help="Niveau vers lequel la volatilité revient.",
                                key="vol_lt", bind="query-params") / 100
            kappa = st.slider("Vitesse de retour à la moyenne", 0.2, 6.0, 2.0, 0.2,
                              help="Plus c'est élevé, plus la vol revient vite à son niveau "
                                   "long terme. Crée la structure par terme.",
                                   key="kappa", bind="query-params")
            xi = st.slider("Vol de la vol", 0.05, 1.5, 0.50, 0.05,
                           help="Donne de la convexité au smile : les options très hors de "
                                "la monnaie deviennent chères dans les deux sens.",
                                key="vol_vol", bind="query-params")
            rho = st.slider("Corrélation spot / vol", -0.95, 0.0, -0.70, 0.05,
                            help="LE paramètre du skew. Négatif = la vol monte quand le "
                                 "marché baisse. Mets-le à 0 et le skew disparaît presque "
                                 "entièrement.", key="rho", bind="query-params")
            st.markdown("---")
            lam = st.slider("Fréquence des sauts (par an)", 0.0, 2.0, 0.3, 0.1,
                            help="0,3 = un saut tous les trois ans en moyenne.",
                            key="sauts", bind="query-params")
            mu_j = st.slider("Taille moyenne d'un saut (%)", -30.0, 5.0, -10.0, 1.0,
                             help="Négatif : les krachs sont des gaps à la baisse.",
                             key="taille_saut", bind="query-params") / 100
            sig_j = st.slider("Dispersion des sauts (%)", 1.0, 40.0, 15.0, 1.0,
                              key="disp_saut", bind="query-params") / 100

    st.sidebar.markdown("---")
    st.sidebar.subheader("Risque émetteur")
    spread = st.sidebar.slider(
        "Spread de crédit émetteur (bp)", 0, 500, 0, 25,
        help="Le point contre-intuitif : un spread élevé AUGMENTE le coupon. La banque se "
             "finance moins cher via le produit structuré que sur le marché obligataire, et "
             "reverse l'économie au client. Un coupon supérieur à la concurrence peut donc "
             "simplement signifier un émetteur moins bien noté.",
             key="spread", bind="query-params")
    recovery = st.sidebar.slider(
        "Taux de recouvrement (%)", 0, 80, 40, 5,
        help="Ce que le client récupère si l'émetteur fait défaut. 40% est la convention "
             "senior non sécurisé. Les porteurs de produits Lehman ont récupéré environ "
             "9 cents par dollar.", key="recouvrement", bind="query-params") / 100

    marge = st.sidebar.slider("Marge banque (% par an)", 0.0, 2.5, 0.8, 0.1,
                              help="Prélevée sur le budget option, invisible dans le "
                                   "payoff.", key="marge", bind="query-params") / 100

    st.sidebar.markdown("---")
    nominal = float(st.sidebar.select_slider("Nominal (EUR)",
                                             [1000, 10000, 100000, 1000000], 1000,
                                             key="nominal", bind="query-params"))
    n_paths = st.sidebar.select_slider("Trajectoires Monte Carlo",
                                       [5000, 10000, 20000, 40000], 10000,
                                       key="trajectoires", bind="query-params")

    st.sidebar.caption("Sous-jacent initial fixé à 100.")


    # --------------------------------------------------------------------
    def trajectoires(T, niveau_ref=1.0, seed=42):
        """
        Renvoie (chemins, sigma) selon le modele choisi. En Black-Scholes, le
        skew passe par une volatilite locale (moteur.simule_lv) : niveau_ref
        n'est plus utilise, garde pour compatibilite.
        """
        if BATES:
            return sim_bates(S0, T, r, q, n_paths, vol_atm ** 2, theta_v ** 2, kappa,
                             xi, rho, lam, mu_j, sig_j, seed)
        return sim_lv(S0, T, r, q, vol_atm, skew, n_paths, seed)


    def defauts(n, T):
        return sim_defaut(n, T, spread, recovery) if spread > 0 else None




    def bandeau_modele():
        if BATES:
            st.caption(f"Pricing sous Bates — vol de départ {vol_atm*100:.0f}%, "
                       f"corrélation spot/vol {rho:+.2f}, {lam:.1f} saut(s) par an."
                       + (f" Spread émetteur {spread} bp." if spread else ""))
        else:
            st.caption(f"Pricing sous Black-Scholes — vol ATM {vol_atm*100:.0f}%, "
                       f"skew {skew:.1f} pt/10%."
                       + (f" Spread émetteur {spread} bp." if spread else ""))

    return SimpleNamespace(**locals())
