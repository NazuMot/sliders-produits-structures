"""Les grecques en mouvement."""

import math
import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0
from communs import MAX_CACHE


def _va_au_jour(cle):
    """Clic sur un graphe : place le curseur Jour sur le point clique."""
    idx = G.point_clique(cle)
    if idx is not None:
        st.session_state.jour_grecques = int(idx)


def affiche(m):
    BATES, nominal, q, r, skew = m.BATES, m.nominal, m.q, m.r, m.skew
    vol_atm = m.vol_atm

    st.title("Les grecques en mouvement", help="**Ce que tu regardes.** La position que porte la banque apres avoir vendu un BRC : elle est\nacheteuse du put a barriere que le client lui a vendu. On suit cette position jour par jour.\n\n**Comment l'utiliser.** Choisis un scenario, puis deplace le curseur *Jour*. A chaque\ninstant tu vois les cinq grecques et l'ordre exact que le trader doit passer pour rester\ncouvert.\n\n**Ce qu'il faut observer.**\n- Le **delta** bouge en permanence : c'est le delta-hedging quotidien.\n- Le **gamma** monte quand le sous-jacent s'approche de la barriere, surtout pres de\n  l'echeance : le trader doit alors rebalancer beaucoup et souvent.\n- Au moment ou la barriere casse, **toutes les courbes sautent en meme temps**. L'option\n  devient un put vanille, definitivement. C'est le gap risk.\n\n**Le meilleur scenario pour commencer.** *Franchissement puis rebond* : la rupture y est\nnette et visible sur les quatre graphes.")
    st.caption("Une trajectoire, jour par jour. Ce que mesure chaque grecque, "
               "et ce que le trader doit faire en face.")

    st.markdown(
        "On suit la position que la banque porte apres avoir vendu un **Barrier "
        "Reverse Convertible** : elle est acheteuse du put down-and-in que le client "
        "lui a vendu. Deplace le curseur de temps et regarde les cinq grecques bouger.")

    c1, c2, c3 = st.columns(3)
    scenario = c1.selectbox(
        "Scenario de marche",
        ["Marche calme", "Glissade vers la barriere", "Franchissement puis rebond",
         "Krach"],
        help="Quatre trajectoires choisies parmi des milliers de simulations, "
             "selon ce qu'elles font par rapport a la barriere.",
             key="gr_scenario", bind="query-params")
    T_g = c2.slider("Maturite (annees)", 0.5, 2.0, 1.0, 0.5,
                    key="gr_maturite", bind="query-params")
    bar_g = c3.slider("Barriere (% du spot)", 50, 90, 70, 5,
                      key="gr_barriere", bind="query-params")

    K_g = S0
    H_g = bar_g / 100 * S0
    vol_g = vol_atm if BATES else M.vol_au_strike(bar_g / 100, vol_atm, skew)

    @st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
    def chemin_scenario(scenario, T, H, vol, r, q, seed=5):
        """Choisit une trajectoire representative du scenario demande."""
        rng = np.random.default_rng(seed)
        n, pas = 3000, max(int(252 * T), 120)
        dt = T / pas
        z = rng.standard_normal((n, pas))
        ch = S0 * np.exp(np.cumsum((r - q - 0.5 * vol ** 2) * dt
                                   + vol * math.sqrt(dt) * z, axis=1))
        ch = np.hstack([np.full((n, 1), S0), ch])
        mini, fin = ch.min(axis=1), ch[:, -1]
        if scenario == "Marche calme":
            score = np.where(mini > H * 1.15, np.abs(fin - S0), 1e9)
        elif scenario == "Glissade vers la barriere":
            score = np.where((mini > H) & (mini < H * 1.08), np.abs(fin - H * 1.1), 1e9)
        elif scenario == "Franchissement puis rebond":
            score = np.where((mini < H) & (fin > S0), -fin, 1e9)
        else:
            score = np.where(mini < H, fin, 1e9)
        return ch[int(np.argmin(score))]

    chemin = chemin_scenario(scenario, T_g, H_g, vol_g, r, q)
    n_pas = len(chemin) - 1
    jours = np.linspace(0, T_g, n_pas + 1)

    @st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
    def suite_grecques(chemin, T, K, H, vol, r, q):
        """
        Les cinq grecques le long de la trajectoire.

        Point cle : une fois la barriere franchie, le knock-in est acquis
        et l'option DEVIENT un put vanille, definitivement. Le prix et le
        delta sautent a cet instant precis : c'est le gap risk, et c'est
        le moment ou la couverture derape.
        """
        n = len(chemin) - 1
        touche = np.minimum.accumulate(chemin) <= H
        out = {k: np.zeros(n + 1) for k in
               ("prix", "delta", "gamma", "vega", "theta", "rho")}
        for i, S in enumerate(chemin):
            tau = max(T * (1 - i / n), 1e-6)
            if touche[i]:
                f = lambda S, T, sigma, r: M.bs_put(S, K, T, r, sigma, q)
            else:
                f = lambda S, T, sigma, r: M.put_down_and_in(S, K, H, T, r, sigma, q)
            g = M.grecques(f, float(S), tau, vol, r)
            for k in out:
                out[k][i] = g[k]
        return out, touche

    G_, touche = suite_grecques(chemin, T_g, K_g, H_g, vol_g, r, q)

    # Le jour est pilote par le curseur ou par un clic sur un graphe : le clic
    # ecrit dans la cle du curseur avant que celui-ci soit redessine.
    if "jour_grecques" not in st.session_state:
        st.session_state.jour_grecques = int(n_pas * 0.55)
    st.session_state.jour_grecques = min(int(st.session_state.jour_grecques), n_pas)
    i = st.slider("Jour", 0, n_pas, key="jour_grecques",
                  help="Deplace-toi dans la vie du produit. Tout se recalcule. Tu peux "
                       "aussi cliquer directement sur un des graphes ci-dessous.",
                       bind="query-params")

    n_opt = nominal / S0          # nombre d'options pour le nominal choisi
    spot_i = float(chemin[i])
    parts = -G_["delta"][i] * n_opt        # actions detenues pour etre neutre
    parts_prec = -G_["delta"][max(i - 1, 0)] * n_opt
    ordre = parts - parts_prec

    m = st.columns(4)
    m[0].metric("Sous-jacent", f"{spot_i:,.1f}",
                delta=f"{spot_i/S0-1:+.1%} depuis le depart")
    m[1].metric("Jours restants", f"{int((T_g - jours[i]) * 365)}")
    m[2].metric("Valeur de la position", f"{G_['prix'][i] * n_opt:,.0f} EUR")
    m[3].metric("Barriere", "franchie" if touche[i] else "intacte")

    st.subheader("Les cinq grecques a cet instant")
    g = st.columns(5)
    g[0].metric("Delta", f"{G_['delta'][i]:+.3f}",
                help="Variation de la position pour +1 point de sous-jacent. "
                     "C'est ce que le trader doit compenser en actions.")
    g[1].metric("Gamma", f"{G_['gamma'][i]:+.4f}",
                help="Variation du delta pour +1 point de sous-jacent. Gamma eleve "
                     "= le delta change vite = il faut rebalancer souvent.")
    g[2].metric("Vega", f"{G_['vega'][i] * n_opt:+.1f} EUR",
                help="Gain ou perte de la position pour +1 point de volatilite "
                     "implicite.")
    g[3].metric("Theta", f"{G_['theta'][i] * n_opt:+.2f} EUR",
                help="Ce que la position gagne ou perd par jour qui passe, "
                     "a marche inchange.")
    g[4].metric("Rho", f"{G_['rho'][i] * n_opt:+.1f} EUR",
                help="Gain ou perte pour +1 point de taux sans risque.")

    st.subheader("Ce que le trader doit faire aujourd'hui")
    if abs(ordre) < 0.01:
        st.success(f"Position en actions inchangee : **{parts:,.2f} actions**. "
                   "Rien a faire aujourd'hui.")
    elif ordre > 0:
        st.info(f"**Acheter {ordre:,.2f} actions** pour passer de {parts_prec:,.2f} "
                f"a {parts:,.2f}. Le sous-jacent a baisse, le put achete par la banque "
                "a pris de la valeur, il faut plus d'actions en face pour rester neutre.")
    else:
        st.info(f"**Vendre {abs(ordre):,.2f} actions** pour passer de {parts_prec:,.2f} "
                f"a {parts:,.2f}. Le sous-jacent est monte, le put vaut moins, "
                "la couverture doit etre allegee.")

    if touche[i] and not touche[max(i - 1, 0)]:
        st.error("**La barriere vient d'etre franchie.** Le knock-in est acquis : "
                 "l'option devient un put vanille, definitivement. Le delta saute d'un "
                 "coup et le trader doit ajuster massivement en une seule fois. C'est le "
                 "gap risk, et c'est ce qui rend les barrieres cheres a couvrir.")

    # ---------------- trajectoire
    j_casse = int(np.argmax(touche)) if touche.any() else None
    deja = np.where(np.arange(len(chemin)) <= i, chemin, np.nan)
    infos_jour = [("Jour", [f"{k} sur {n_pas}" for k in range(n_pas + 1)]),
                  ("Barriere", ["franchie" if t else "intacte" for t in touche])]
    graphe = G.courbes(
        jours, [{"nom": "Trajectoire du sous-jacent", "y": chemin, "couleur": G.PRODUIT,
                 "epaisseur": 1.8},
                {"nom": "Deja vecu", "y": deja, "couleur": G.ACCENT, "epaisseur": 2.6}],
        scenario, "La partie orange est deja vecue ; le point est aujourd'hui. "
                  "Clique sur le graphe pour aller a ce jour-la.",
        "Temps (annees)", "Niveau du sous-jacent", nom_x="Date",
        fmt_x=lambda v: f"{v:,.2f} an", fmt_y=lambda v: G.niveau(v, S0),
        x_domaine=(0, T_g), zone_y=(min(chemin.min() * 0.95, H_g * 0.9), H_g, G.FOND_PERTE),
        y_domaine=(min(chemin.min() * 0.95, H_g * 0.9), max(chemin.max(), S0) * 1.04),
        niveaux_h=[(H_g, f"barriere  {H_g:,.0f}", G.PERTE),
                   (S0, f"strike  {S0:,.0f}", G.NIVEAU)],
        marqueurs=[(jours[i], spot_i, G.ACCENT)], lignes_infobulle=infos_jour,
        legende=False, clic=True, hauteur=260)
    G.affiche(graphe, cle="g_chemin", sur_clic=lambda: _va_au_jour("g_chemin"))

    # ---------------- les grecques dans le temps
    series = [("Delta", G_["delta"], G.PRODUIT,
               "Combien d'actions le trader doit detenir", lambda v: f"{v:+.3f}"),
              ("Gamma", G_["gamma"], G.PERTE,
               "A quelle vitesse cette quantite change", lambda v: f"{v:+.4f}"),
              ("Vega  (EUR par point de vol)", G_["vega"] * n_opt, G.GAIN,
               "Sensibilite a la volatilite implicite", lambda v: G.eur(v, signe=True)),
              ("Theta  (EUR par jour)", G_["theta"] * n_opt, G.ACCENT,
               "Ce que le temps coute ou rapporte", lambda v: f"{v:+.2f} EUR")]
    cols = st.columns(2)
    for k, (nom, serie, coul, sous, fmt) in enumerate(series):
        niv = [(jours[i], "aujourd'hui", G.NIVEAU)]
        if j_casse:
            niv.append((jours[j_casse], "barriere franchie" if k == 0 else "", G.PERTE))
        graphe = G.courbes(
            jours, [{"nom": nom, "y": serie, "couleur": coul, "epaisseur": 2.0}],
            nom, sous, "Temps (annees)", None, nom_x="Date",
            fmt_x=lambda v: f"{v:,.2f} an", fmt_y=fmt, x_domaine=(0, T_g),
            niveaux_v=niv, marqueurs=[(jours[i], float(serie[i]), coul)],
            lignes_infobulle=infos_jour, legende=False, zero=True, clic=True, hauteur=230)
        cle = f"g_grecque_{k}"
        with cols[k % 2]:
            G.affiche(graphe, cle=cle, sur_clic=lambda c=cle: _va_au_jour(c))

    st.info(
        "**Comment lire ces quatre courbes.**\n\n"
        "Le **delta** dit combien d'actions detenir. Il bouge tous les jours : c'est le "
        "delta-hedging, l'activite quotidienne du trader.\n\n"
        "Le **gamma** dit a quelle vitesse ce nombre change. Quand il monte, le trader "
        "doit rebalancer plus souvent, donc payer plus de frais. C'est la raison "
        "chiffree pour laquelle une barriere proche du spot fait degrader le prix.\n\n"
        "Le **vega** s'eteint a mesure que l'echeance approche : une option qui expire "
        "demain ne depend plus de la volatilite.\n\n"
        "Le **theta** est le miroir du gamma. Celui qui possede du gamma paie du theta "
        "chaque jour, et inversement. Un trader ne choisit jamais l'un sans l'autre.")

    if st.checkbox("Voir pourquoi une barriere proche de l'echeance est intenable",
                   key="gr_voir_echeance", bind="query-params"):
        lignes = []
        for tau in [1.0, 0.25, 0.08, 0.02]:
            for S in [H_g * 1.15, H_g * 1.04, H_g * 1.01]:
                gg = M.grecques(
                    lambda S, T, sigma, r: M.put_down_and_in(S, K_g, H_g, T, r, sigma, q),
                    float(S), tau, vol_g, r)
                lignes.append({
                    "Maturite restante": f"{tau:.2f} an" if tau >= 0.25 else f"{int(tau*365)} jours",
                    "Sous-jacent": f"{S:,.1f}",
                    "Delta": f"{gg['delta']:+.2f}",
                    "Gamma": f"{gg['gamma']:+.3f}",
                    "Actions a detenir": f"{-gg['delta'] * n_opt:,.1f}"})
        st.table(lignes)
        st.caption("A un an du terme, le delta reste sage. A une semaine, avec le spot "
                   "juste au-dessus de la barriere, il depasse 4 en valeur absolue : le "
                   "trader doit detenir quatre fois plus d'actions que le nominal ne le "
                   "suggere, et tout basculera d'un coup si la barriere cede. Aucun desk "
                   "ne cote une telle structure sans charger une marge consequente.")
