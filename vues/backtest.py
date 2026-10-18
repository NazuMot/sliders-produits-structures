"""Backtest historique : le produit relance chaque mois sur le vrai Euro Stoxx 50."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import backtest as B
import moteur as M
import graphiques as G
from moteur import S0
from communs import MAX_CACHE, sim_bs, sim_bates


SOUS_JACENTS = {
    "Indice des prix (SX5E)": "prix",
    "Net return (dividendes réinvestis)": "net_return",
    "Indice a décrément (50 points)": "decrement",
}
NOMS_SOUS_JACENT = {"prix": "l'Euro Stoxx 50 (indice des prix)",
                    "net_return": "l'Euro Stoxx 50 dividendes réinvestis",
                    "decrement": "l'Euro Stoxx 50 à décrément de 50 points"}
# 50 points par an sur un indice relance a 1000 = 5 points pour un spot a 100 (unite du moteur)
DECREMENT = 5.0
ONGLETS = {"autocall": "Autocall Phoenix", "brc": "Barrier Reverse Convertible"}
SOURCES = ["Backtest historique", "Monte Carlo, vue de marché actuelle"]

# Une couleur = un sens, comme sur la page Autocall
RAPPEL, TERME, PERTE = "Rappelé par anticipation", "Va au terme sans perte", "Perte en capital"
INTACTE, TOUCHEE = "Barrière intacte", "Barrière touchée, sans perte"
COULEURS_AC = {RAPPEL: G.GAIN, TERME: G.NIVEAU, PERTE: G.PERTE}
COULEURS_BRC = {INTACTE: G.GAIN, TOUCHEE: G.NIVEAU, PERTE: G.PERTE}

# Pour dire au client dans quel contexte le pire cas a ete emis
EPISODES = [("1999-06-01", "2001-12-31", "en pleine bulle internet"),
            ("2007-01-01", "2008-09-30", "à la veille de la crise financière"),
            ("2010-01-01", "2011-12-31", "pendant la crise des dettes souveraines"),
            ("2019-06-01", "2020-02-29", "juste avant le krach du Covid")]


# =====================================================================
# DONNEES ET CALCULS EN CACHE
# =====================================================================

@st.cache_resource(show_spinner=False)
def _historique():
    """Lu une seule fois pour tous les utilisateurs (0,3 Mo, jamais modifie)."""
    return B.charge_historique()


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _autocall(T, par_an, trigger, bar_cp, bar_cap, memoire, coupon_pa, sous_jacent, debut):
    return B.backtest_autocall(_historique(), T, par_an, trigger, bar_cp, bar_cap, memoire,
                               coupon_pa, sous_jacent, debut)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _brc(T, barriere, strike, continu, coupon_pa, sous_jacent, debut):
    return B.backtest_brc(_historique(), T, barriere, strike, continu, coupon_pa, sous_jacent,
                          debut)


@st.cache_data(show_spinner=False, max_entries=2)
def _sim_decrement(T, r, n, sigma, bates):
    """Indice a decrement simule : dividendes reinvestis (q = 0) moins 5 points par an."""
    if bates is None:
        return M.simule_bs(S0, T, r, 0.0, sigma, n, decrement=DECREMENT)
    return M.simule_bates(S0, T, r, 0.0, n, *bates, decrement=DECREMENT)


@st.cache_data(show_spinner=False, max_entries=4)
def _courbe_indice(serie, debut):
    """Cours de l'indice une seance sur cinq (graphe leger), un peu avant la 1re emission."""
    h = _historique()
    v = h.prix if serie == "prix" else h.net_return
    i0 = max(int(np.searchsorted(h.dates, np.datetime64(debut))) - 60, 0)
    idx = np.unique(np.r_[np.arange(i0, len(h.dates), 5), len(h.dates) - 1])
    return pd.DataFrame({"date": pd.to_datetime(h.dates[idx]), "niveau": v[idx]})


def _trajectoires(m, T, niveau_ref, sous_jacent):
    """
    Monte Carlo de la vue de marche actuelle, sur le meme sous-jacent que le
    backtest. Indice des prix : les trajectoires des pages produits (meme
    graine, donc meme coupon). Net return : dividendes reinvestis, q = 0.
    Decrement : q = 0 et 5 points retires chaque annee.
    """
    if sous_jacent == "prix":
        return m.trajectoires(T, niveau_ref)
    bates = ((m.vol_atm ** 2, m.theta_v ** 2, m.kappa, m.xi, m.rho, m.lam, m.mu_j, m.sig_j)
             if m.BATES else None)
    sigma = 0.0 if bates else M.vol_au_strike(niveau_ref, m.vol_atm, m.skew)
    if sous_jacent == "net_return":
        if bates:
            return sim_bates(S0, T, m.r, 0.0, m.n_paths, *bates)
        return sim_bs(S0, T, m.r, 0.0, sigma, m.n_paths)
    return _sim_decrement(T, m.r, min(m.n_paths, 20000), sigma, bates)


# =====================================================================
# FORMATS
# =====================================================================

def _date(d):
    s = str(d)
    return f"{s[8:10]}/{s[5:7]}/{s[:4]}"


def _contexte(d):
    s = str(d)
    return next((" " + txt for a, b, txt in EPISODES if a <= s <= b), "")


def _p(x, decimales=1):
    """Fraction -> texte en % : 0.577 -> '57.7 %'."""
    return G.pct(x * 100, decimales)


def _rendement(x):
    return f"{x * 100:+.1f} % par an"


def _derive(m, sous_jacent):
    """Ce que le Monte Carlo suppose pour la tendance du sous-jacent (risque-neutre)."""
    if sous_jacent == "prix":
        return f"taux moins dividende, {(m.r - m.q) * 100:+.2f} % par an"
    if sous_jacent == "net_return":
        return f"taux sans risque, {m.r * 100:+.2f} % par an"
    return (f"taux sans risque moins le décrément, {(m.r - DECREMENT / 100) * 100:+.2f} % par an "
            "au départ")


def _vol_modele(m, niveau_ref):
    if m.BATES:
        return f"Bates, vol de départ {m.vol_atm * 100:.0f} %"
    return (f"Black-Scholes, vol {M.vol_au_strike(niveau_ref, m.vol_atm, m.skew) * 100:.0f} % "
            f"(vol ATM + skew au niveau de la barrière)")


# =====================================================================
# GRAPHES
# =====================================================================

def _graphe_emissions(res, serie, issues, couleurs, details, barriere, titre, sous_titre,
                      i_pire, texte_pire):
    """
    L'indice dans le temps, un point par emission colore selon l'issue pour
    le client. Au survol : ligne verticale, infobulle de l'emission, et sa vie
    (trait bleu au niveau de depart jusqu'a la fin, tirete rouge a la barriere).
    """
    h = _historique()
    v = h.prix if serie == "prix" else h.net_return
    i0 = np.searchsorted(h.dates, res.emission)
    i1 = np.searchsorted(h.dates, res.fin)
    em = pd.DataFrame({"idx": np.arange(len(i0)), "date": pd.to_datetime(res.emission),
                       "fin": pd.to_datetime(res.fin), "niveau": v[i0], "niveau_fin": v[i1],
                       "barriere": v[i0] * barriere, "issue": issues})
    infobulle = []
    for j, (nom, textes) in enumerate(details):
        em[f"d{j}"] = list(textes)
        infobulle.append(alt.Tooltip(f"d{j}:N", title=nom))
    courbe = _courbe_indice(serie, str(res.emission[0]))

    lo = min(float(courbe["niveau"].min()), float(em["barriere"].min())) * 0.92
    hi = float(courbe["niveau"].max()) * 1.06
    ech_y = alt.Scale(domain=[lo, hi], nice=False, zero=False)
    ech_c = alt.Scale(domain=list(couleurs), range=list(couleurs.values()))
    x = alt.X("date:T", scale=alt.Scale(nice=False),
              axis=alt.Axis(title=None, format="%Y", tickCount=14))
    y = alt.Y("niveau:Q", scale=ech_y)
    nom_y = "Euro Stoxx 50 (points)" if serie == "prix" else "Euro Stoxx 50 net return (points)"

    survol = alt.selection_point(name="survol", nearest=True, on="pointerover", fields=["idx"],
                                 empty=False, clear="pointerout")
    base = alt.Chart(em)
    pire = em.iloc[[i_pire]].assign(txt=texte_pire)
    couches = [
        alt.Chart(courbe).mark_line(color=G.DIRECT, strokeWidth=1.1, opacity=0.6).encode(
            x, alt.Y("niveau:Q", scale=ech_y, axis=alt.Axis(title=nom_y, labelExpr=G.AXE_EUR))),
        base.mark_point(filled=True, size=42, opacity=1, stroke="white", strokeWidth=0.6).encode(
            x, y, color=alt.Color("issue:N", scale=ech_c, legend=G._legende())),
        alt.Chart(pire).mark_point(size=320, strokeWidth=1.8, color=G.PERTE, opacity=1).encode(
            x, y),
        alt.Chart(pire).mark_text(align="left", baseline="bottom", dx=12, dy=-12, fontSize=11.5,
                                  fontWeight="bold", color=G.PERTE, lineBreak="\n").encode(
            x, y, text="txt:N"),
        # vie de l'emission survolee
        base.mark_rule(color=G.PRODUIT, strokeWidth=3).encode(
            x, x2="fin:T", y=y).transform_filter(survol),
        base.mark_rule(color=G.PERTE, strokeWidth=2, strokeDash=[5, 4]).encode(
            x, x2="fin:T", y=alt.Y("barriere:Q", scale=ech_y)).transform_filter(survol),
        base.mark_point(filled=True, size=90, stroke="white", strokeWidth=1.5, opacity=1).encode(
            alt.X("fin:T"), alt.Y("niveau_fin:Q", scale=ech_y),
            color=alt.Color("issue:N", scale=ech_c, legend=None)).transform_filter(survol),
        # survol : ligne verticale + infobulle unique
        base.mark_rule(color=G.NIVEAU, strokeWidth=1.2).encode(
            x, opacity=alt.condition(survol, alt.value(0.9), alt.value(0)),
            tooltip=infobulle).add_params(survol),
        base.mark_point(filled=True, size=170, stroke="white", strokeWidth=2, opacity=1).encode(
            x, y, color=alt.Color("issue:N", scale=ech_c, legend=None)).transform_filter(survol),
    ]
    return alt.layer(*couches).properties(title=G._titre(titre, sous_titre), height=440)


def _graphe_repartition(categories, histo, mc, dates, titre, sous_titre):
    """
    Barres groupees : part des emissions reelles et des scenarios Monte Carlo
    dans chaque issue. L'infobulle donne les deux valeurs et leur cumul.
    """
    histo, mc = np.asarray(histo, float), np.asarray(mc, float)
    cum_h, cum_m = np.cumsum(histo), np.cumsum(mc)
    lignes = []
    for i, cat in enumerate(categories):
        commun = {"cat": cat, "ordre": i, "date": dates[i], "h": _p(histo[i]), "m": _p(mc[i]),
                  "ch": _p(min(cum_h[i], 1.0), 0), "cm": _p(min(cum_m[i], 1.0), 0)}
        for source, val in zip(SOURCES, (histo[i], mc[i])):
            lignes.append(dict(commun, source=source, v=val * 100,
                               txt=f"{val * 100:.0f}" if val * 100 >= 1.5 else ""))
    df = pd.DataFrame(lignes)
    haut = max(float(df["v"].max()), 1.0) * 1.18
    x = alt.X("cat:N", sort=list(categories), title=None,
              axis=alt.Axis(labelAngle=-40 if len(categories) > 6 else 0, labelLimit=160))
    xo = alt.XOffset("source:N", sort=SOURCES)
    y = alt.Y("v:Q", title="Part des émissions (%)", scale=alt.Scale(domain=[0, haut], nice=False))
    survol = alt.selection_point(name="survol", on="pointerover", fields=["cat"],
                                 clear="pointerout")
    infobulle = [alt.Tooltip("cat:N", title="Issue"), alt.Tooltip("date:N", title="Date"),
                 alt.Tooltip("h:N", title=SOURCES[0]), alt.Tooltip("m:N", title=SOURCES[1]),
                 alt.Tooltip("ch:N", title="Termine à cette date ou avant (historique)"),
                 alt.Tooltip("cm:N", title="Termine à cette date ou avant (Monte Carlo)")]
    base = alt.Chart(df)
    return alt.layer(
        base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x, xo, y,
            color=alt.Color("source:N", scale=alt.Scale(domain=SOURCES,
                                                        range=[G.PRODUIT, G.ACCENT]),
                            legend=G._legende()),
            opacity=alt.condition(survol, alt.value(1.0), alt.value(0.45)),
            tooltip=infobulle).add_params(survol),
        base.mark_text(baseline="bottom", dy=-3, fontSize=10.5, fontWeight="bold").encode(
            x, xo, y, text="txt:N"),
    ).properties(title=G._titre(titre, sous_titre), height=340)


# =====================================================================
# DISCOURS SALES
# =====================================================================

def _a_ne_pas_dire(sous_jacent, n, p_capital):
    lignes = [
        f"« Vous avez {p_capital * 100:.0f} % de chances de récupérer votre capital » : un "
        f"backtest n'est pas une probabilité. Les {n} émissions se chevauchent ; elles ne "
        "racontent qu'une poignée de crises (2000-2003, 2008, 2011, 2020), et rien ne dit que "
        "la prochaine leur ressemblera.",
        "Choisir la date de départ qui arrange : démarrer en 2009 efface la bulle internet et "
        "Lehman. Essaie le curseur « Première émission ».",
        "Comparer au seul indice des prix : le client renonce aux dividendes, la comparaison "
        "honnête est l'indice dividendes réinvestis.",
        "Donner la moyenne sans le pire cas et sa date.",
        "Oublier l'émetteur : le backtest suppose que la banque a toujours payé. Les porteurs "
        "de produits Lehman ont récupéré environ 9 cents par dollar.",
    ]
    if sous_jacent == "decrement":
        lignes.append("Montrer le backtest de l'indice classique pour un produit sur indice a "
                      "décrément : les 50 points retires chaque année pèsent d'autant plus lourd "
                      "que l'indice baisse.")
    return lignes


def _discours(phrase, sous_jacent, n, p_capital):
    interdits = "\n".join(f"- {t}" for t in _a_ne_pas_dire(sous_jacent, n, p_capital))
    st.info(
        "**Ce qu'on peut dire au client.** " + phrase + "\n\n"
        "Le pire cas se cite avec sa date et son contexte : c'est lui qui donne la mesure du "
        "risque. On peut aussi dire combien de fois le produit a fait mieux que l'indice "
        "dividendes réinvestis, sur la même durée.\n\n"
        "**Ce qu'on ne dit pas.**\n" + interdits)


# =====================================================================
# ONGLET AUTOCALL
# =====================================================================

def _onglet_autocall(m, sj, debut):
    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturité maximale (années)", 1.0, 6.0, 3.0, 0.5,
                  help="Durée MAXIMALE : le produit est presque toujours rappelé avant. Seules "
                       "les émissions arrivées à leur terme avant octobre 2026 sont comptées.",
                  key="bt_ac_maturite", bind="query-params")
    freq = c1.selectbox("Fréquence d'observation", ["Trimestrielle", "Semestrielle", "Annuelle"],
                        help="Dates où l'on regarde l'indice : rappel, coupon. Même date du "
                             "mois que l'émission, premier jour de cotation suivant.",
                        key="bt_ac_frequence", bind="query-params")
    trigger = c2.slider("Trigger de rappel (%)", 80, 110, 100, 5,
                        help="Au-dessus à une date d'observation, le produit s'arrête : nominal "
                             "rembourse plus le coupon.", key="bt_ac_trigger", bind="query-params")
    bar_cp = c2.slider("Barrière de coupon (%)", 50, 100, 70, 5,
                       help="Le coupon n'est payé que si l'indice est au-dessus à la date "
                            "d'observation.", key="bt_ac_bar_coupon", bind="query-params")
    bar_cap = c3.slider("Barrière de capital (%)", 40, 90, 60, 5,
                        help="Observée uniquement A L'ÉCHÉANCE. En dessous, le client encaisse "
                             "toute la baisse.", key="bt_ac_bar_capital", bind="query-params")
    memoire = c3.checkbox("Effet mémoire", value=True,
                          help="Un coupon manque est rattrapé à la première date où l'indice "
                               "repasse au-dessus de la barrière de coupon.",
                          key="bt_ac_memoire", bind="query-params")
    k1, k2 = st.columns([2, 1])
    au_marche = k2.checkbox(
        "Coupon de la vue de marché actuelle",
        help="Remplace le coupon fixe par celui que le desk offrirait aujourd'hui avec la barre "
             "latérale (même calcul que la page Autocall). Sur l'indice à décrément, le coupon "
             "monte : on voit alors si ce supplément a payé les pertes en plus.",
        key="bt_ac_coupon_marche", bind="query-params")
    coupon = k1.slider("Coupon (% par an)", 0.0, 20.0, 8.0, 0.25, disabled=au_marche,
                       help="Coupon fixe sur tout l'historique, versé par observation au "
                            "prorata (8 % par an = 2 % par trimestre). En réalité le coupon "
                            "suit les taux et la vol du jour d'émission.",
                       key="bt_ac_coupon", bind="query-params")

    par_an = {"Trimestrielle": 4, "Semestrielle": 2, "Annuelle": 1}[freq]
    n_obs = max(int(T * par_an), 1)

    # Monte Carlo de la vue de marche : coupon du desk et repartition des issues
    ch, _ = _trajectoires(m, T, bar_cap / 100, sj)
    tau = m.defauts(len(ch), T)
    cpn_marche = M.coupon_autocall(ch, T, m.r, m.nominal, n_obs, trigger / 100, bar_cp / 100,
                                   bar_cap / 100, memoire, m.marge, tau, m.recovery) * par_an
    coupon_pa = round(cpn_marche if au_marche else coupon / 100, 6)
    if au_marche:
        k1.caption(f"Coupon de la vue de marché : **{_p(coupon_pa, 2)} par an** "
                   f"({_p(coupon_pa / par_an, 2)} par observation).")

    res = _autocall(T, par_an, trigger / 100, bar_cp / 100, bar_cap / 100, memoire, coupon_pa,
                    sj, debut)
    r = res.resume
    if r["n"] == 0:
        st.warning("Aucune émission n'est arrivée à son terme avant la fin de l'historique : "
                   "avance la première émission ou raccourcis la maturité.")
        return

    # --- graphe principal
    issues = np.where(res.perte, PERTE, np.where(res.rappele, RAPPEL, TERME))
    issue_txt = [
        (f"perte : {c * 100:.1f} % du nominal rendu le {_date(f)}" if p else
         f"rappelé à l'observation {k} sur {res.n_obs}, le {_date(f)}" if rp else
         f"échéance le {_date(f)}, nominal rendu")
        for p, rp, k, c, f in zip(res.perte, res.rappele, res.obs_fin, res.capital, res.fin)]
    details = [
        ("Émission", [_date(e) for e in res.emission]),
        ("Issue", issue_txt),
        ("Coupons touchés", [f"{nb:.0f} coupon(s), {c * 100:.1f} % du nominal"
                             for nb, c in zip(res.nb_coupons, res.coupons)]),
        ("Durée de vie", [f"{d:.2f} an" for d in res.duree]),
        ("Sous-jacent a la fin", [f"{(s - 1) * 100:+.1f} %" for s in res.perf_fin]),
        ("Rendement du produit", [_rendement(t) for t in res.tri]),
        ("Détention directe (div. réinvestis)", [_rendement(d) for d in res.direct]),
        ("Produit moins direct", [f"{(t - d) * 100:+.1f} pts par an"
                                  for t, d in zip(res.tri, res.direct)]),
        ("VSTOXX à l'émission", [f"{v:.1f}" if np.isfinite(v) else "-" for v in res.vstoxx]),
    ]
    i_pire = int(np.argmin(res.tri))
    serie = "net_return" if sj == "net_return" else "prix"
    graphe = _graphe_emissions(
        res, serie, issues, COULEURS_AC, details, bar_cap / 100,
        f"{_p(r['rappel_premiere_date'], 0)} des autocalls rappelés à la 1re date, "
        f"{_p(r['perte'])} en perte",
        f"Un point par émission mensuelle depuis {debut[:4]}, colore selon l'issue. Survole un "
        "point : trait bleu = vie du produit, tireté rouge = barrière de capital.",
        i_pire, f"pire cas\n{_p(res.capital[i_pire])} rendu")
    G.affiche(graphe)
    if sj == "decrement":
        st.caption("La courbe est l'indice des prix pour se repérer ; chaque émission est "
                   "déroulée sur son propre indice a décrément, relance à 1000 le jour de "
                   "l'émission.")

    # --- statistiques cles
    c = st.columns(4)
    c[0].metric("Rappelés à la 1re date", _p(r["rappel_premiere_date"]),
                help="Part des émissions remboursées dès la première observation. Le client "
                     "touche un seul coupon et doit réinvestir : bonne nouvelle pour le "
                     "capital, mauvaise pour le rendement espère.")
    c[1].metric("Pertes en capital", _p(r["perte"]),
                f"perte moyenne {_p(r['perte_moyenne_si_perte'], 0)}" if r["perte"] else None,
                delta_color="off", delta_arrow="off",
                help="Part des émissions qui ont rendu moins que le nominal, et la perte "
                     "moyenne quand il y a perte. Sous la barrière, la perte est toute la "
                     "baisse de l'indice, pas l'écart à la barrière.")
    c[2].metric("Pire cas", f"{_p(res.capital[i_pire])} rendu",
                f"émis le {_date(res.emission[i_pire])}", delta_color="off", delta_arrow="off",
                help="L'émission au plus mauvais rendement. C'est le chiffre à citer en premier "
                     "à un client, avec sa date.")
    ecart = r["tri_par_annee"] - r["direct_par_annee"]
    c[3].metric("Rendement par année investie", _p(r["tri_par_annee"]),
                f"{ecart * 100:+.1f} pt contre le direct ({_p(r['direct_par_annee'])})",
                help="Moyenne géométrique pondérée par la durée : un rappel après 3 mois pèse "
                     "3 mois. Direct = Euro Stoxx 50 dividendes nets réinvestis, sur exactement "
                     "la même période que chaque produit.")
    c = st.columns(4)
    c[0].metric("Durée de vie moyenne", f"{r['duree_moyenne']:.2f} ans",
                help=f"Sur {T:g} ans possibles. Le rappel coupe les coupons futurs.")
    c[1].metric("Rappel anticipe", _p(r["rappel_anticipe"]),
                help="Part des émissions rappelées avant l'échéance, à n'importe quelle date.")
    c[2].metric("Échéance sans perte", _p(r["echeance_sans_perte"]),
                help="Jamais rappelé, mais au-dessus de la barrière de capital à l'échéance.")
    c[3].metric("Bat la détention directe", _p(r["bat_direct"]),
                help="Part des émissions dont le rendement dépasse celui de l'indice dividendes "
                     "réinvestis sur la même période. Un autocall plafonne les hausses : il "
                     "perd souvent contre l'indice en marché haussier.")
    st.caption(
        f"{r['n']} émissions du {_date(r['premiere_emission'])} au "
        f"{_date(r['derniere_emission'])}, déroulées sur {NOMS_SOUS_JACENT[sj]}. Biais : seuls "
        f"les produits termines comptent ({r['n_en_cours']} encore en cours), et des émissions "
        "mensuelles qui se chevauchent ne sont pas des expériences indépendantes.")

    # --- backtest contre Monte Carlo
    st.subheader("Historique contre Monte Carlo")
    indices = M.idx_obs(ch.shape[1] - 1, n_obs)
    vivant = np.ones(len(ch), dtype=bool)
    mc = []
    for j in indices[:-1]:
        rappel = vivant & (ch[:, j] >= trigger / 100 * S0)
        mc.append(rappel.mean())
        vivant &= ~rappel
    ST = ch[:, -1]
    sous_cap = ST < bar_cap / 100 * S0
    mc += [(vivant & ~sous_cap).mean(), (vivant & sous_cap).mean()]
    histo = list(r["fin_par_observation"][:-1]) + [r["echeance_sans_perte"], r["perte"]]
    cats = [f"Obs {k + 1}" for k in range(n_obs - 1)] + ["Terme sans perte", "Terme avec perte"]
    dates = [f"{(k + 1) * T / n_obs:.2f} an" for k in range(n_obs - 1)] + [f"{T:.2f} ans"] * 2

    c = st.columns(3)
    c[0].metric("Coupon de la vue de marché", f"{_p(cpn_marche, 2)} p.a.",
                help="Ce que le desk offrirait aujourd'hui pour ce produit avec la barre "
                     "latérale (marge et spread émetteur compris).")
    c[1].metric("Rappel à la 1re date", _p(r["rappel_premiere_date"], 0),
                f"modèle : {_p(mc[0], 0)}" if n_obs > 1 else None, delta_color="off",
                delta_arrow="off", help="Historique, et ce que le Monte Carlo attendait.")
    c[2].metric("Perte en capital", _p(r["perte"]), f"modèle : {_p(mc[-1])}",
                delta_color="off", delta_arrow="off",
                help="Historique, et ce que le Monte Carlo attendait.")
    G.affiche(_graphe_repartition(
        cats, histo, mc, dates,
        f"Perte : {_p(r['perte'])} dans l'histoire, {_p(mc[-1])} pour le modèle",
        "Quand le produit se termine : part des émissions réelles contre part des scénarios "
        "simulés avec la vue de marché actuelle"))
    st.markdown(
        f"Le Monte Carlo price sous la probabilité risque-neutre : le sous-jacent y dérive au "
        f"{_derive(m, sj)}, avec {_vol_modele(m, bar_cap / 100)}, sans prime de risque ni "
        "changement de régime. L'histoire, elle, alterne marchés calmes et krachs. Un écart "
        "n'est pas une erreur du modèle : le modèle chiffre ce que coûte la couverture "
        "aujourd'hui, le backtest raconte une seule trajectoire passée, découpée en fenêtres.")

    pire = (f"la pire, émise le {_date(res.emission[i_pire])}{_contexte(res.emission[i_pire])}, "
            f"aurait rendu {_p(res.capital[i_pire])} du nominal")
    phrase = (f"« Si cet autocall avait été émis chaque mois depuis {debut[:4]} sur "
              f"{NOMS_SOUS_JACENT[sj]}, {_p(r['rappel_premiere_date'], 0)} des {r['n']} "
              f"émissions auraient été remboursées dès la première date avec un coupon de "
              f"{_p(coupon_pa / par_an, 2)} (soit {_p(coupon_pa, 2)} par an), et "
              f"{_p(r['perte'])} auraient perdu du capital ; "
              f"{pire}. »")
    _discours(phrase, sj, r["n"], 1 - r["perte"])


# =====================================================================
# ONGLET BRC
# =====================================================================

def _onglet_brc(m, sj, debut):
    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturité (années)", 0.25, 3.0, 1.0, 0.25,
                  help="Durée fixe : un BRC n'est jamais rappelé.",
                  key="bt_brc_maturite", bind="query-params")
    bar = c2.slider("Barrière (% du spot)", 50, 95, 70, 5,
                    help="Le niveau qui ARME le risque. Le toucher ne fait pas perdre : c'est le "
                         "strike qui décide à la fin.", key="bt_brc_barriere", bind="query-params")
    strike = c3.slider("Strike (% du spot)", 80, 110, 100, 5,
                       help="Si la barrière a été touchée et que l'indice finit sous le strike, "
                            "le client reçoit niveau final / strike.",
                       key="bt_brc_strike", bind="query-params")
    continu = st.radio("Observation de la barrière",
                       ["Continue (américaine)", "À l'échéance (européenne)"], horizontal=True,
                       help="Continue : chaque clôture compte (pas les plus bas en séance). À "
                            "l'échéance : seule la dernière clôture compte.",
                       key="bt_brc_observation", bind="query-params").startswith("Continue")
    k1, k2 = st.columns([2, 1])
    au_marche = k2.checkbox(
        "Coupon de la vue de marché actuelle",
        help="Remplace le coupon fixe par celui que le desk offrirait aujourd'hui avec la barre "
             "latérale (même calcul que la page BRC).",
        key="bt_brc_coupon_marche", bind="query-params")
    coupon = k1.slider("Coupon (% par an)", 0.0, 20.0, 7.0, 0.25, disabled=au_marche,
                       help="Coupon garanti, versé à l'échéance avec le capital, fixe sur tout "
                            "l'historique. En réalité il suit les taux et la vol du jour.",
                       key="bt_brc_coupon", bind="query-params")

    ch, sg = _trajectoires(m, T, bar / 100, sj)
    tau = m.defauts(len(ch), T)
    cpn_marche = M.coupon_brc(ch, T, m.r, m.nominal, bar / 100, strike / 100, continu, m.marge,
                              sg, tau, m.recovery)
    coupon_pa = round(cpn_marche if au_marche else coupon / 100, 6)
    if au_marche:
        k1.caption(f"Coupon de la vue de marché : **{_p(coupon_pa, 2)} par an**.")

    res = _brc(T, bar / 100, strike / 100, continu, coupon_pa, sj, debut)
    r = res.resume
    if r["n"] == 0:
        st.warning("Aucune émission n'est arrivée à son terme avant la fin de l'historique : "
                   "avance la première émission ou raccourcis la maturité.")
        return

    # --- graphe principal
    issues = np.where(res.perte, PERTE, np.where(res.touche, TOUCHEE, INTACTE))
    issue_txt = [
        (f"barrière touchée : {c * 100:.1f} % du nominal rendu + coupon" if p else
         "barrière touchée, fin au-dessus du strike : nominal + coupon" if t else
         "barrière intacte : nominal + coupon")
        for p, t, c in zip(res.perte, res.touche, res.capital)]
    details = [
        ("Émission", [_date(e) for e in res.emission]),
        ("Échéance", [_date(f) for f in res.fin]),
        ("Issue", issue_txt),
        ("Plus bas atteint" if continu else "Niveau à l'échéance",
         [f"{(b - 1) * 100:+.1f} %" for b in res.plus_bas]),
        ("Sous-jacent a l'échéance", [f"{(s - 1) * 100:+.1f} %" for s in res.perf_fin]),
        ("Rendement du produit", [_rendement(t) for t in res.tri]),
        ("Détention directe (div. réinvestis)", [_rendement(d) for d in res.direct]),
        ("Produit moins direct", [f"{(t - d) * 100:+.1f} pts par an"
                                  for t, d in zip(res.tri, res.direct)]),
        ("VSTOXX à l'émission", [f"{v:.1f}" if np.isfinite(v) else "-" for v in res.vstoxx]),
    ]
    i_pire = int(np.argmin(res.capital))
    serie = "net_return" if sj == "net_return" else "prix"
    G.affiche(_graphe_emissions(
        res, serie, issues, COULEURS_BRC, details, bar / 100,
        f"{_p(r['barriere_touchee'], 0)} des BRC ont vu la barrière touchée, "
        f"{_p(r['perte'])} en perte",
        f"Un point par émission mensuelle depuis {debut[:4]}, colore selon l'issue. Survole un "
        "point : trait bleu = vie du produit, tireté rouge = barrière.",
        i_pire, f"pire cas\n{_p(res.capital[i_pire])} rendu"))
    if sj == "decrement":
        st.caption("La courbe est l'indice des prix pour se repérer ; chaque émission est "
                   "déroulée sur son propre indice a décrément, relance à 1000 le jour de "
                   "l'émission.")

    # --- statistiques cles
    c = st.columns(4)
    c[0].metric("Barrière touchée", _p(r["barriere_touchee"]),
                help="Part des émissions dont l'indice à clôture au moins une fois sous la "
                     "barrière (ou à fini sous la barrière, en observation à l'échéance).")
    c[1].metric("Pertes en capital", _p(r["perte"]),
                f"perte moyenne {_p(r['perte_moyenne_si_perte'], 0)}" if r["perte"] else None,
                delta_color="off", delta_arrow="off",
                help="Barrière touchée ET fin sous le strike. Le coupon est versé quand même.")
    c[2].metric("Pire cas", f"{_p(res.capital[i_pire])} rendu",
                f"émis le {_date(res.emission[i_pire])}", delta_color="off", delta_arrow="off",
                help="Capital rendu par la pire émission, hors coupon. C'est le chiffre à citer "
                     "en premier à un client, avec sa date.")
    ecart = r["tri_par_annee"] - r["direct_par_annee"]
    c[3].metric("Rendement par année investie", _p(r["tri_par_annee"]),
                f"{ecart * 100:+.1f} pt contre le direct ({_p(r['direct_par_annee'])})",
                help="Moyenne géométrique pondérée par la durée, coupon compris. Direct = Euro "
                     "Stoxx 50 dividendes nets réinvestis, sur la même période.")
    c = st.columns(4)
    c[0].metric("Coupon backteste", f"{_p(coupon_pa, 2)} p.a.",
                help="Le même coupon pour toutes les émissions.")
    c[1].metric("Rendement médian", _p(r["tri_median"]),
                f"direct : {_p(r['direct_median'])}", delta_color="off", delta_arrow="off",
                help="L'émission du milieu : la moitié fait mieux, la moitié moins bien.")
    c[2].metric("Bat la détention directe", _p(r["bat_direct"]),
                help="Part des émissions dont le rendement dépasse celui de l'indice dividendes "
                     "réinvestis. Le BRC plafonne au coupon : il perd contre l'indice dès que "
                     "celui-ci monte plus que le coupon.")
    c[3].metric("Émissions", f"{r['n']}", f"{r['n_en_cours']} en cours, non comptées",
                delta_color="off", delta_arrow="off")
    st.caption(
        f"{r['n']} émissions du {_date(r['premiere_emission'])} au "
        f"{_date(r['derniere_emission'])}, déroulées sur {NOMS_SOUS_JACENT[sj]}, barrière "
        f"observée sur les clôtures. Biais : seuls les produits termines comptent, et des "
        "émissions mensuelles qui se chevauchent ne sont pas des expériences indépendantes.")

    # --- backtest contre Monte Carlo
    st.subheader("Historique contre Monte Carlo")
    ST = ch[:, -1].astype(np.float64)
    p_no = (M.proba_non_touche(ch, bar / 100 * S0, sg, T) if continu
            else (ST > bar / 100 * S0).astype(np.float64))
    sous_k = ST < strike / 100 * S0
    mc = [p_no.mean(), ((1 - p_no) * ~sous_k).mean(), ((1 - p_no) * sous_k).mean()]
    histo = [1 - r["barriere_touchee"], r["barriere_touchee"] - r["perte"], r["perte"]]
    c = st.columns(3)
    c[0].metric("Coupon de la vue de marché", f"{_p(cpn_marche, 2)} p.a.",
                help="Ce que le desk offrirait aujourd'hui pour ce BRC avec la barre latérale.")
    c[1].metric("Barrière touchée", _p(r["barriere_touchee"]), f"modèle : {_p(1 - mc[0])}",
                delta_color="off", delta_arrow="off",
                help="Historique, et ce que le Monte Carlo attendait (pont brownien).")
    c[2].metric("Perte en capital", _p(r["perte"]), f"modèle : {_p(mc[2])}",
                delta_color="off", delta_arrow="off",
                help="Historique, et ce que le Monte Carlo attendait.")
    G.affiche(_graphe_repartition(
        [INTACTE, TOUCHEE, PERTE], histo, mc, [f"{T:.2f} an"] * 3,
        f"Perte : {_p(r['perte'])} dans l'histoire, {_p(mc[2])} pour le modèle",
        "Issue à l'échéance : part des émissions réelles contre part des scénarios simulés "
        "avec la vue de marché actuelle"))
    st.markdown(
        f"Le Monte Carlo price sous la probabilité risque-neutre : le sous-jacent y dérive au "
        f"{_derive(m, sj)}, avec {_vol_modele(m, bar / 100)}, sans prime de risque ni "
        "changement de régime. L'histoire, elle, alterne marchés calmes et krachs. Un écart "
        "n'est pas une erreur du modèle : le modèle chiffre ce que coûte la couverture "
        "aujourd'hui, le backtest raconte une seule trajectoire passée, découpée en fenêtres.")

    phrase = (f"« Si ce BRC avait été émis chaque mois depuis {debut[:4]} sur "
              f"{NOMS_SOUS_JACENT[sj]}, {_p(1 - r['perte'], 0)} des {r['n']} émissions auraient "
              f"rendu le nominal plus le coupon de {_p(coupon_pa, 2)} par an, et {_p(r['perte'])} "
              f"auraient perdu du capital ; la pire, émise le {_date(res.emission[i_pire])}"
              f"{_contexte(res.emission[i_pire])}, aurait rendu {_p(res.capital[i_pire])} du "
              f"nominal plus le coupon. »")
    _discours(phrase, sj, r["n"], 1 - r["perte"])


# =====================================================================
# PAGE
# =====================================================================

def _change_onglet():
    """Les onglets n'ont pas de bind : on recopie l'onglet ouvert dans le lien."""
    code = next(k for k, v in ONGLETS.items() if v == st.session_state["bt_onglet"])
    st.query_params["bt_onglet"] = code


def affiche(m):
    st.title("Backtest historique", help="**Ce que c'est.** Le même produit lance chaque mois depuis 2000 et déroule sur les vraies\nclôtures de l'Euro Stoxx 50 : quand à-t-il été rappelé, combien de coupons, quelle perte,\net combien aurait rapporte l'indice sur la même période.\n\n**Comment c'est construit.** Clôtures quotidiennes STOXX de 1986 à octobre 2026, dividendes\nnets réinvestis pour le net return et l'indice à décrément. Dates d'observation : même date\ndu mois que l'émission. Barrière continue = chaque clôture. Seuls les produits termines\ncomptent.\n\n**Pourquoi un client le demande.** C'est la première question en rendez-vous : « et si\nj'avais achète ce produit par le passe ? ». Le Monte Carlo dit ce que coûte le produit\naujourd'hui ; le backtest dit ce qui serait arrive.\n\n**Quoi regarder.** Le pire cas et sa date, la part de pertes, le rendement par année\ninvestie contre la détention directe, et l'écart entre l'histoire et le modèle.")
    st.caption("Et si on avait vendu ce produit chaque mois depuis 2000 ?")

    c1, c2 = st.columns([2, 1])
    sj = SOUS_JACENTS[c1.radio(
        "Sous-jacent", list(SOUS_JACENTS), horizontal=True,
        help="Indice des prix : le SX5E classique, dividendes exclus. Net return : dividendes "
             "nets réinvestis. Indice a décrément : dividendes réinvestis moins 50 points par "
             "an (base 1000 le jour de l'émission, soit 5 % la première année), le sous-jacent "
             "de la plupart des autocalls vendus en France : coupon plus élevé, mais l'indice "
             "décroche quand les dividendes réels sont sous le prélèvement.",
        key="bt_sous_jacent", bind="query-params")]
    annee = c2.slider("Première émission", 1987, 2020, 2000, 1,
                      help="Changer l'année de départ change le verdict : démarrer en 2009 "
                           "efface la bulle internet et la crise de 2008. C'est exactement ce "
                           "qu'un Sales ne doit pas faire pour embellir un backtest.",
                      key="bt_debut", bind="query-params")
    debut = f"{annee}-01-01"

    defaut = ONGLETS.get(st.query_params.get("bt_onglet"), ONGLETS["autocall"])
    onglets = st.tabs(list(ONGLETS.values()), default=defaut, key="bt_onglet",
                      on_change=_change_onglet)
    with onglets[0]:
        if onglets[0].open:
            _onglet_autocall(m, sj, debut)
    with onglets[1]:
        if onglets[1].open:
            _onglet_brc(m, sj, debut)
