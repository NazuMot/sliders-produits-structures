"""
Sliders Produits Structures - Tom Uzan
Lancer avec :  streamlit run app.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sliders Produits Structures", layout="wide")

NOMINAL = 1000.0
S0 = 100.0


# ----------------------------------------------------------------------
# BOITE A OUTILS
# ----------------------------------------------------------------------

def norm_cdf(x):
    """Fonction de repartition de la loi normale centree reduite."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, r, sigma, q=0.0):
    """
    Prix Black-Scholes d'un call europeen.
    q = taux de dividende. Il reduit le prix du call : les dividendes
    ne vont pas au detenteur de l'option, donc l'option vaut moins.
    """
    if T <= 0 or sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


@st.cache_data(show_spinner=False)
def simulate_paths(S0, T, r, sigma, q, n_paths, n_steps, seed=42):
    """
    Simule des trajectoires du sous-jacent (mouvement brownien geometrique).
    Rendement moyen = r - q : on price en univers risque-neutre, pas avec
    une prevision de marche. C'est la base de tout pricing d'option.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diff = sigma * math.sqrt(dt)
    shocks = rng.standard_normal((n_paths, n_steps))
    log_paths = np.cumsum(drift + diff * shocks, axis=1)
    paths = S0 * np.exp(log_paths)
    return np.hstack([np.full((n_paths, 1), S0), paths])


# ----------------------------------------------------------------------
# NAVIGATION
# ----------------------------------------------------------------------

produit = st.sidebar.radio(
    "Produit",
    ["1 - Tracker Certificate",
     "2 - Capital Protection Certificate",
     "3 - Barrier Reverse Convertible"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Nominal fixe a CHF 1'000. Sous-jacent initial = 100.")


# ======================================================================
# 1 - TRACKER
# ======================================================================

if produit.startswith("1"):
    st.title("Tracker Certificate")
    st.caption("Exposition lineaire au sous-jacent. Pas de protection, pas de coupon.")

    c1, c2 = st.columns(2)
    with c1:
        participation = st.slider("Participation (%)", 50, 150, 100, 5,
                                  help="100% = 1:1. Certains trackers ont un ratio different.")
    with c2:
        frais = st.slider("Frais de gestion annuels (%)", 0.0, 2.0, 0.0, 0.1,
                          help="Souvent finances par les dividendes non reverses.")

    spots = np.linspace(40, 160, 300)
    perf = spots / S0 - 1
    payoff = NOMINAL * (1 + participation / 100 * perf) * (1 - frais / 100)
    direct = NOMINAL * (1 + perf)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spots, payoff, lw=2.5, color="#C8102E", label="Tracker")
    ax.plot(spots, direct, lw=1.2, color="black", ls="--", label="Detention directe")
    ax.axhline(NOMINAL, color="grey", lw=0.8)
    ax.axvline(S0, color="grey", lw=0.8)
    ax.set_xlabel("Niveau du sous-jacent a l'echeance")
    ax.set_ylabel("Remboursement (CHF)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Scenarios")
    rows = []
    for p in [-40, -20, 0, 20, 40]:
        val = NOMINAL * (1 + participation / 100 * p / 100) * (1 - frais / 100)
        rows.append({
            "Sous-jacent": f"{p:+d}%",
            "Tracker": f"{val:,.0f}",
            "Detention directe": f"{NOMINAL * (1 + p / 100):,.0f}",
            "Ecart": f"{val - NOMINAL * (1 + p / 100):+,.0f}",
        })
    st.table(rows)

    st.info(
        "**Ce que tu vends** : de l'acces, pas du rendement. Un panier thematique, "
        "un marche ferme, un indice custom - en une seule ligne de portefeuille.\n\n"
        "**Le vrai risque a expliquer au client** : risque emetteur. "
        "Le client est creancier de la banque, il ne detient pas les actions. "
        "C'est la difference majeure avec un ETF."
    )


# ======================================================================
# 2 - CAPITAL PROTECTION
# ======================================================================

elif produit.startswith("2"):
    st.title("Capital Protection Certificate")
    st.caption("Capital garanti a l'echeance + participation partielle a la hausse.")

    c1, c2, c3 = st.columns(3)
    with c1:
        T = st.slider("Maturite (annees)", 1, 10, 5, 1)
        protection = st.slider("Niveau de protection (%)", 80, 100, 100, 5,
                               help="90% = le client accepte de perdre 10% max, "
                                    "ce qui libere du budget pour plus de participation.")
    with c2:
        r = st.slider("Taux sans risque (%)", 0.0, 6.0, 3.0, 0.25) / 100
        sigma = st.slider("Volatilite implicite (%)", 10.0, 45.0, 20.0, 1.0) / 100
    with c3:
        q = st.slider("Taux de dividende (%)", 0.0, 5.0, 2.0, 0.25) / 100
        marge = st.slider("Marge banque (%)", 0.0, 3.0, 1.5, 0.25,
                          help="Prelevee sur le budget option, pas visible du client.")

    # --- Le coeur pedagogique : la participation n'est pas choisie, elle est calculee
    zc = protection / 100 * math.exp(-r * T)          # cout de la garantie (par unite de nominal)
    budget = 1.0 - zc - marge / 100                    # ce qu'il reste pour acheter le call
    call_unitaire = bs_call(S0, S0, T, r, sigma, q) / S0
    participation = max(budget / call_unitaire, 0.0) if call_unitaire > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cout de la garantie", f"{zc * NOMINAL:,.0f}")
    m2.metric("Budget option", f"{budget * NOMINAL:,.0f}")
    m3.metric("Prix du call ATM", f"{call_unitaire * NOMINAL:,.0f}")
    m4.metric("=> Participation", f"{participation * 100:,.0f}%")

    if participation < 0.35:
        st.error("Participation trop faible : produit invendable. "
                 "Baisse le niveau de protection ou attends que les taux remontent.")

    spots = np.linspace(40, 200, 400)
    perf = spots / S0 - 1
    payoff = NOMINAL * (protection / 100 + participation * np.maximum(perf, 0))
    direct = NOMINAL * (1 + perf)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spots, payoff, lw=2.5, color="#C8102E", label="Capital Protection")
    ax.plot(spots, direct, lw=1.2, color="black", ls="--", label="Detention directe")
    ax.axhline(NOMINAL, color="grey", lw=0.8)
    ax.axvline(S0, color="grey", lw=0.8)
    ax.set_xlabel("Niveau du sous-jacent a l'echeance")
    ax.set_ylabel("Remboursement (CHF)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Effet des taux, tout le reste egal")
    rows = []
    for taux in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
        z = protection / 100 * math.exp(-taux * T)
        b = 1.0 - z - marge / 100
        c = bs_call(S0, S0, T, taux, sigma, q) / S0
        p = max(b / c, 0.0) if c > 0 else 0.0
        rows.append({
            "Taux": f"{taux * 100:.1f}%",
            "Mis de cote": f"{z * NOMINAL:,.0f}",
            "Budget option": f"{b * NOMINAL:,.0f}",
            "Participation": f"{p * 100:,.0f}%",
        })
    st.table(rows)

    st.info(
        "**Le mecanisme** : la banque doit pouvoir rembourser le nominal a l'echeance. "
        "Elle place aujourd'hui juste ce qu'il faut pour y arriver. Ce qui reste achete un call.\n\n"
        "**Consequence commerciale** : taux hauts = produit facile a vendre. "
        "Taux bas = participation ridicule, le marche bascule vers les BRC et autocalls. "
        "C'est exactement ce qui s'est passe entre 2015 et 2021."
    )


# ======================================================================
# 3 - BARRIER REVERSE CONVERTIBLE
# ======================================================================

else:
    st.title("Barrier Reverse Convertible")
    st.caption("Coupon inconditionnel. Capital rembourse sauf si la barriere est franchie "
               "ET que le sous-jacent finit sous le strike.")

    c1, c2, c3 = st.columns(3)
    with c1:
        T = st.slider("Maturite (annees)", 0.25, 3.0, 1.0, 0.25)
        barriere = st.slider("Barriere (% du spot initial)", 50, 95, 75, 5)
    with c2:
        sigma = st.slider("Volatilite implicite (%)", 10.0, 70.0, 25.0, 1.0) / 100
        r = st.slider("Taux sans risque (%)", 0.0, 6.0, 3.0, 0.25) / 100
    with c3:
        q = st.slider("Taux de dividende (%)", 0.0, 6.0, 2.0, 0.25) / 100
        marge = st.slider("Marge banque (%)", 0.0, 3.0, 1.0, 0.25)

    observation = st.radio(
        "Observation de la barriere",
        ["Continue (americaine) - standard marche", "A l'echeance seulement (europeenne)"],
        horizontal=True,
    )
    continue_obs = observation.startswith("Continue")

    B = barriere / 100 * S0
    K = S0  # strike a 100%, convention standard

    n_paths, n_steps = 30000, max(int(252 * T), 50)
    paths = simulate_paths(S0, T, r, sigma, q, n_paths, n_steps)
    ST = paths[:, -1]
    touched = (paths.min(axis=1) <= B) if continue_obs else (ST <= B)

    # --- Coupon equitable : celui qui rend le produit exactement egal au nominal
    # Prix = nominal*e^-rT + coupon*e^-rT - valeur du put down-and-in = nominal
    shares = NOMINAL / S0
    payoff_dip = np.where(touched, np.maximum(K - ST, 0.0), 0.0) * shares
    dip = math.exp(-r * T) * payoff_dip.mean()
    coupon_pct = ((NOMINAL * (math.exp(r * T) - 1) + dip * math.exp(r * T)) / NOMINAL
                  - marge / 100 * T)
    coupon_pct = max(coupon_pct, 0.0)
    coupon_chf = coupon_pct * NOMINAL

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coupon equitable", f"{coupon_pct * 100:,.2f}%")
    m2.metric("Valeur du put vendu", f"{dip:,.0f} CHF")
    m3.metric("Proba de toucher", f"{touched.mean() * 100:,.1f}%")
    m4.metric("Proba de perte", f"{(touched & (ST < K)).mean() * 100:,.1f}%")

    # --- Payoff reel, trajectoire par trajectoire
    payoff = np.where(touched & (ST < K), NOMINAL * ST / K, NOMINAL) + coupon_chf
    direct = NOMINAL * ST / S0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    idx = np.random.default_rng(1).choice(n_paths, 3000, replace=False)
    ax.scatter(ST[idx][~touched[idx]], payoff[idx][~touched[idx]], s=4, alpha=0.4,
               color="#2E7D32", label="Barriere intacte")
    ax.scatter(ST[idx][touched[idx]], payoff[idx][touched[idx]], s=4, alpha=0.4,
               color="#C8102E", label="Barriere touchee")
    ax.plot(np.sort(ST), NOMINAL * np.sort(ST) / S0, color="black", ls="--", lw=1,
            label="Detention directe")
    ax.axvline(B, color="grey", ls=":", lw=1.5)
    ax.text(B, ax.get_ylim()[1] * 0.95, " barriere", fontsize=8, color="grey")
    ax.set_xlabel("Sous-jacent a l'echeance")
    ax.set_ylabel("Remboursement (CHF)")
    ax.set_title("Payoff : deux points au meme spot, deux resultats")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i in range(40):
        col = "#C8102E" if touched[i] else "#2E7D32"
        ax.plot(np.linspace(0, T, n_steps + 1), paths[i], lw=0.7, alpha=0.6, color=col)
    ax.axhline(B, color="grey", ls=":", lw=1.5)
    ax.axhline(S0, color="black", lw=0.8)
    ax.set_xlabel("Temps (annees)")
    ax.set_ylabel("Sous-jacent")
    ax.set_title("Trajectoires : le chemin compte, pas seulement l'arrivee")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.subheader("Les 4 scenarios")
    scenarios = [
        ("Hausse forte, barriere intacte", 130, False),
        ("Lateral, barriere intacte", 102, False),
        ("Chute a 72 puis remontee a 103", 103, True),
        ("Baisse durable a 70", 70, True),
    ]
    rows = []
    for nom, st_val, touch in scenarios:
        pay = (NOMINAL * st_val / K if (touch and st_val < K) else NOMINAL) + coupon_chf
        rows.append({
            "Scenario": nom,
            "Spot final": f"{st_val}",
            "Barriere": "touchee" if touch else "intacte",
            "BRC": f"{pay:,.0f}",
            "Direct": f"{NOMINAL * st_val / S0:,.0f}",
            "Ecart": f"{pay - NOMINAL * st_val / S0:+,.0f}",
        })
    st.table(rows)

    st.subheader("D'ou vient le coupon : bouge la volatilite")
    rows = []
    for v in [0.15, 0.25, 0.35, 0.50, 0.65]:
        p = simulate_paths(S0, T, r, v, q, 20000, n_steps, seed=7)
        s_t = p[:, -1]
        t_ = (p.min(axis=1) <= B) if continue_obs else (s_t <= B)
        d = math.exp(-r * T) * (np.where(t_, np.maximum(K - s_t, 0.0), 0.0) * shares).mean()
        c = max((NOMINAL * (math.exp(r * T) - 1) + d * math.exp(r * T)) / NOMINAL
                - marge / 100 * T, 0.0)
        rows.append({
            "Volatilite": f"{v * 100:.0f}%",
            "Proba de toucher": f"{t_.mean() * 100:,.1f}%",
            "Valeur du put": f"{d:,.0f}",
            "Coupon": f"{c * 100:,.2f}%",
        })
    st.table(rows)

    st.info(
        "**Le coupon n'est pas un taux d'interet.** C'est le prix d'une option que le client "
        "vend a la banque : le droit de lui livrer les actions si elles s'effondrent.\n\n"
        "**Deux sources** : les taux (la banque garde le cash un an) + la prime du put. "
        "Un coupon eleve veut dire que le put vaut cher - donc que le sinistre est probable.\n\n"
        "**Ce que montre le graphe de gauche** : deux clients peuvent finir au meme niveau de "
        "sous-jacent et toucher des montants differents. Le chemin decide, pas l'arrivee seule."
    )
