"""
=======================================================================
SLIDERS PRODUITS STRUCTURES  -  v2
Tom Uzan - EDHEC BBA Finance
=======================================================================

  - 6 produits structures sur sous-jacent action / indice
  - les parametres commerciaux (participation, coupon) ne sont PAS
    choisis : ils sont CALCULES a partir du budget, comme sur un desk
  - moteur Monte Carlo commun avec variables antithetiques
  - grecques par bump & revalue a alea fixe
  - prise en compte approchee du skew de volatilite

Lancer avec :  streamlit run app.py
=======================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sliders Produits Structures", layout="wide")

ROUGE = "#C8102E"
VERT = "#2E7D32"
BLEU = "#1565C0"
GRIS = "#6E6E6E"
S0 = 100.0


# =====================================================================
# MOTEUR
# =====================================================================

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, r, sigma, q=0.0):
    """Call europeen Black-Scholes. q = taux de dividende."""
    if T <= 0 or sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def vol_au_strike(strike_pct, vol_atm, skew_pts):
    """
    Approximation praticien du skew.

    Constat de marche : sur actions, les puts bas coutent plus cher en vol
    que les options a la monnaie. Deux raisons : les actions tombent plus
    vite qu'elles ne montent, et les institutionnels achetent massivement
    des puts de protection sans vendeur naturel en face.

    skew_pts = points de vol gagnes pour 10% de baisse du strike.
    C'est une approximation, PAS un modele a volatilite locale.
    """
    return max(vol_atm + skew_pts / 100.0 * (1.0 - strike_pct) / 0.10, 0.01)


@st.cache_data(show_spinner=False, max_entries=80)
def simule(spot, T, r, q, sigma, n_paths, n_steps, seed=42):
    """
    Trajectoires en mouvement brownien geometrique.

    Deux points qui comptent :
      1. Le drift est (r - q), pas une prevision. On price en univers
         risque-neutre, sinon il y a arbitrage.
      2. Variables antithetiques : chaque tirage est double par son oppose.
         Reduit la variance sans cout supplementaire.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = max(n_paths // 2, 1)
    z = rng.standard_normal((half, n_steps))
    z = np.vstack([z, -z])
    incr = (r - q - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    chemins = spot * np.exp(np.cumsum(incr, axis=1))
    return np.hstack([np.full((chemins.shape[0], 1), spot), chemins])


def idx_obs(n_steps, n_obs):
    return [int(round(n_steps * (k + 1) / n_obs)) for k in range(n_obs)]


# =====================================================================
# PRICERS
# =====================================================================

def pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, coupon_pa, spot0):
    B = bar_pct * spot0
    K = strike_pct * spot0
    ST = chemins[:, -1]
    touche = (chemins.min(axis=1) <= B) if continu else (ST <= B)
    remb = np.where(touche & (ST < K), nominal * ST / K, nominal)
    return math.exp(-r * T) * (remb + coupon_pa * T * nominal).mean()


def coupon_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, marge_pa, spot0):
    """Payoff affine en coupon -> resolution directe."""
    base = pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, 0.0, spot0)
    unite = math.exp(-r * T) * nominal * T
    cible = nominal * (1.0 - marge_pa * T)
    return max((cible - base) / unite, 0.0) if unite > 0 else 0.0


def pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                memoire, coupon, spot0):
    """
    Autocall Phoenix a memoire.

    A chaque observation :
      spot >= trigger        -> remboursement anticipe du nominal
      spot >= barriere coupon -> coupon paye (+ les coupons manques si memoire)
    A l'echeance si jamais rappele :
      spot >= barriere capital -> nominal
      sinon                    -> nominal * spot / spot initial
    """
    n_paths, n_steps = chemins.shape[0], chemins.shape[1] - 1
    idx = idx_obs(n_steps, n_obs)
    dt_obs = T / n_obs
    vivant = np.ones(n_paths, dtype=bool)
    manques = np.zeros(n_paths)
    pv = np.zeros(n_paths)

    for k, i in enumerate(idx):
        t = (k + 1) * dt_obs
        disc = math.exp(-r * t)
        S = chemins[:, i]
        paye = vivant & (S >= bar_cp * spot0)
        n_cp = (manques + 1.0) if memoire else np.ones(n_paths)
        pv += np.where(paye, coupon * nominal * n_cp * disc, 0.0)
        manques = np.where(paye, 0.0, manques + 1.0)
        if k < len(idx) - 1:
            rappel = vivant & (S >= trigger * spot0)
            pv += np.where(rappel, nominal * disc, 0.0)
            vivant = vivant & ~rappel

    ST = chemins[:, -1]
    final = np.where(ST >= bar_cap * spot0, nominal, nominal * ST / spot0)
    pv += np.where(vivant, final * math.exp(-r * T), 0.0)
    return pv.mean()


def coupon_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, marge_pa, spot0):
    a = pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, 0.0, spot0)
    b = pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, 0.01, spot0)
    pente = (b - a) / 0.01
    cible = nominal * (1.0 - marge_pa * T)
    return max((cible - a) / pente, 0.0) if pente > 0 else 0.0


def stats_autocall(chemins, T, n_obs, trigger, bar_cap, spot0):
    n_paths, n_steps = chemins.shape[0], chemins.shape[1] - 1
    idx = idx_obs(n_steps, n_obs)
    dt_obs = T / n_obs
    vivant = np.ones(n_paths, dtype=bool)
    duree = np.full(n_paths, T)
    for k, i in enumerate(idx[:-1]):
        rappel = vivant & (chemins[:, i] >= trigger * spot0)
        duree = np.where(rappel, (k + 1) * dt_obs, duree)
        vivant = vivant & ~rappel
    ST = chemins[:, -1]
    perte = vivant & (ST < bar_cap * spot0)
    return duree.mean(), 1.0 - vivant.mean(), perte.mean()


# =====================================================================
# PARAMETRES DE MARCHE PARTAGES
# =====================================================================

st.sidebar.title("Parametres de marche")
st.sidebar.caption("Une seule vue de marche alimente tous les produits, "
                   "comme sur un desk.")

r = st.sidebar.slider("Taux sans risque (%)", 0.0, 7.0, 3.0, 0.25) / 100
q = st.sidebar.slider("Taux de dividende (%)", 0.0, 7.0, 2.5, 0.25) / 100
vol_atm = st.sidebar.slider("Volatilite implicite ATM (%)", 8.0, 70.0, 22.0, 1.0) / 100
skew = st.sidebar.slider("Skew (pts de vol / -10% de strike)", 0.0, 6.0, 2.0, 0.5,
                         help="Mets 0 pour voir disparaitre l'effet du skew "
                              "sur les coupons.")
marge = st.sidebar.slider("Marge banque (% par an)", 0.0, 2.5, 0.8, 0.1) / 100

st.sidebar.markdown("---")
nominal = float(st.sidebar.select_slider("Nominal (CHF)",
                                         [1000, 10000, 100000, 1000000], 1000))
n_paths = st.sidebar.select_slider("Trajectoires Monte Carlo",
                                   [10000, 30000, 60000], 30000)

st.sidebar.markdown("---")
produit = st.sidebar.radio(
    "Produit",
    ["Comparateur",
     "1 - Tracker Certificate",
     "2 - Capital Protection",
     "3 - Barrier Reverse Convertible",
     "4 - Autocall Phoenix",
     "5 - Bonus Certificate",
     "6 - Twin-Win"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Sous-jacent initial fixe a 100.")


def cadre(ax, titre_x="Sous-jacent a l'echeance"):
    ax.axhline(nominal, color=GRIS, lw=0.7)
    ax.axvline(S0, color=GRIS, lw=0.7)
    ax.set_xlabel(titre_x)
    ax.set_ylabel("Remboursement (CHF)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.28)


# =====================================================================
# COMPARATEUR
# =====================================================================

if produit == "Comparateur":
    st.title("Comparateur")
    st.caption("Meme sous-jacent, meme maturite, meme vue de marche. "
               "Ce que chaque structure donne, et ce qu'elle prend.")

    T = st.slider("Maturite (annees)", 1.0, 5.0, 2.0, 0.5)
    n_steps = max(int(252 * T), 60)

    ch_atm = simule(S0, T, r, q, vol_atm, n_paths, n_steps)
    ch_brc = simule(S0, T, r, q, vol_au_strike(0.70, vol_atm, skew), n_paths, n_steps)
    ch_ac = simule(S0, T, r, q, vol_au_strike(0.65, vol_atm, skew), n_paths, n_steps)

    zc = math.exp(-r * T)
    budget = 1.0 - zc - marge * T
    call_u = bs_call(S0, S0, T, r, vol_au_strike(1.0, vol_atm, skew), q) / S0
    part = max(budget / call_u, 0.0) if call_u > 0 else 0.0

    cpn_brc = coupon_brc(ch_brc, T, r, nominal, 0.70, 1.0, True, marge, S0)
    n_obs = max(int(T * 4), 1)
    cpn_ac = coupon_autocall(ch_ac, T, r, nominal, n_obs, 1.0, 0.70, 0.65,
                             True, marge, S0)
    duree, p_rappel, p_perte = stats_autocall(ch_ac, T, n_obs, 1.0, 0.65, S0)

    st.subheader("Ce que le desk peut offrir aujourd'hui")
    st.table([
        {"Produit": "Tracker", "Offre": "Participation 100%",
         "Protection": "aucune", "Plafond hausse": "aucun", "Vue client": "haussier franc"},
        {"Produit": "Capital Protection", "Offre": f"Participation {part * 100:,.0f}%",
         "Protection": "capital garanti", "Plafond hausse": "participation reduite",
         "Vue client": "haussier prudent"},
        {"Produit": "Barrier Reverse Conv.", "Offre": f"Coupon {cpn_brc * 100:,.2f}% p.a.",
         "Protection": "barriere 70%", "Plafond hausse": f"{cpn_brc * 100:,.2f}% p.a.",
         "Vue client": "neutre"},
        {"Produit": "Autocall Phoenix", "Offre": f"Coupon {cpn_ac * 100:,.2f}% / trimestre",
         "Protection": "barriere 65% a l'echeance", "Plafond hausse": "coupons cumules",
         "Vue client": "neutre a legerement haussier"},
    ])

    c = st.columns(3)
    c[0].metric("Proba de hausse du sous-jacent",
                f"{(ch_atm[:, -1] > S0).mean() * 100:,.1f}%",
                help="Sous mesure risque-neutre. Ce n'est pas une prevision.")
    c[1].metric("Duree de vie moyenne autocall", f"{duree:,.2f} ans")
    c[2].metric("Proba de perte autocall", f"{p_perte * 100:,.1f}%")

    st.subheader("Profils de remboursement a l'echeance")
    spots = np.linspace(30, 190, 400)
    perf = spots / S0 - 1
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(spots, nominal * (1 + perf), color="black", ls="--", lw=1.2, label="Direct")
    ax.plot(spots, nominal * (1 + part * np.maximum(perf, 0)), lw=2.2, color=BLEU,
            label=f"Capital Protection ({part * 100:,.0f}%)")
    ax.plot(spots, np.where(spots < 70, nominal * spots / S0, nominal)
            + cpn_brc * nominal * T, lw=2.2, color=ROUGE,
            label="BRC (cas barriere touchee)")
    cadre(ax)
    st.pyplot(fig)

    st.info(
        "**La regle qui structure tout le metier** : protection, participation, coupon. "
        "Jamais trois sur trois. Le budget est fixe : il vient des taux et de ce que le "
        "client accepte de vendre comme optionalite.\n\n"
        "**Ta question au client** : quelle est ta vue, et qu'es-tu pret a abandonner "
        "pour l'exprimer ?"
    )


# =====================================================================
# 1 - TRACKER
# =====================================================================

elif produit.startswith("1"):
    st.title("Tracker Certificate")
    st.caption("Exposition lineaire. Produit d'acces, pas produit de rendement.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 1.0, 10.0, 3.0, 0.5)
    participation = c2.slider("Participation (%)", 50, 150, 100, 5)
    frais_pa = c3.slider("Frais annuels (%)", 0.0, 2.0, 0.6, 0.1)

    spots = np.linspace(30, 180, 400)
    perf = spots / S0 - 1
    payoff = nominal * (1 + participation / 100 * perf) * (1 - frais_pa / 100 * T)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spots, nominal * spots / S0, color="black", ls="--", lw=1.1, label="Direct")
    ax.plot(spots, payoff, lw=2.6, color=ROUGE, label="Tracker")
    cadre(ax)
    st.pyplot(fig)

    st.subheader("Cout reel de la detention indirecte")
    st.table([
        {"Sous-jacent": f"{p:+d}%",
         "Tracker": f"{nominal * (1 + participation / 100 * p / 100) * (1 - frais_pa / 100 * T):,.0f}",
         "Direct": f"{nominal * (1 + p / 100):,.0f}",
         "Ecart": f"{nominal * (1 + participation / 100 * p / 100) * (1 - frais_pa / 100 * T) - nominal * (1 + p / 100):+,.0f}"}
        for p in [-40, -20, 0, 20, 40]])

    st.info(
        f"**Le dividende est le nerf du produit.** Le sous-jacent verse {q * 100:,.1f}% par an "
        f"que le client ne touche pas : sur {T:,.1f} ans cela represente environ "
        f"{(math.exp(q * T) - 1) * 100:,.1f}% de rendement abandonne. C'est ce qui finance "
        "la structure et la marge.\n\n"
        "**Le risque a nommer en clientele** : le client est creancier de la banque, il ne "
        "detient pas les actions. Difference majeure avec un ETF, et le premier point sur "
        "lequel un institutionnel te challengera."
    )


# =====================================================================
# 2 - CAPITAL PROTECTION
# =====================================================================

elif produit.startswith("2"):
    st.title("Capital Protection Certificate")
    st.caption("La participation n'est pas un choix commercial. C'est un reste de budget.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 1.0, 10.0, 5.0, 0.5)
    protection = c2.slider("Niveau de protection (%)", 80, 100, 100, 5)
    cap = c3.slider("Cap sur la hausse (%)", 110, 300, 300, 10,
                    help="300 = pas de cap en pratique. Capper libere du budget "
                         "et remonte la participation.")

    zc = protection / 100 * math.exp(-r * T)
    budget = 1.0 - zc - marge * T
    vol_call = vol_au_strike(1.0, vol_atm, skew)
    prix_call = bs_call(S0, S0, T, r, vol_call, q) / S0
    prix_cap = bs_call(S0, cap / 100 * S0, T, r,
                       vol_au_strike(cap / 100, vol_atm, skew), q) / S0
    cout = max(prix_call - prix_cap, 1e-9)
    part = max(budget / cout, 0.0)

    m = st.columns(4)
    m[0].metric("Cout de la garantie", f"{zc * nominal:,.0f}")
    m[1].metric("Budget option", f"{budget * nominal:,.0f}")
    m[2].metric("Cout du call spread", f"{cout * nominal:,.0f}")
    m[3].metric("Participation offerte", f"{part * 100:,.0f}%")

    if part < 0.35:
        st.error("Participation sous 35% : invendable. Baisse la protection, allonge la "
                 "maturite, ou mets un cap.")
    elif part > 1.0:
        st.success("Participation au-dessus de 100% : configuration de taux favorable. "
                   "C'est la fenetre commerciale a exploiter.")

    spots = np.linspace(30, 220, 500)
    perf = np.minimum(spots / S0 - 1, cap / 100 - 1)
    payoff = nominal * (protection / 100 + part * np.maximum(perf, 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spots, nominal * spots / S0, color="black", ls="--", lw=1.1, label="Direct")
    ax.plot(spots, payoff, lw=2.6, color=ROUGE, label="Capital Protection")
    cadre(ax)
    st.pyplot(fig)

    st.subheader("Sensibilite aux taux, tout le reste constant")
    st.table([
        {"Taux": f"{t * 100:.0f}%",
         "Mis de cote": f"{protection / 100 * math.exp(-t * T) * nominal:,.0f}",
         "Budget option": f"{(1 - protection / 100 * math.exp(-t * T) - marge * T) * nominal:,.0f}",
         "Participation": f"{max((1 - protection / 100 * math.exp(-t * T) - marge * T) / max((bs_call(S0, S0, T, t, vol_call, q) - bs_call(S0, cap / 100 * S0, T, t, vol_au_strike(cap / 100, vol_atm, skew), q)) / S0, 1e-9), 0) * 100:,.0f}%"}
        for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]])

    st.info(
        "**Le mecanisme** : la banque doit rembourser le nominal a l'echeance. Elle place "
        "aujourd'hui juste ce qu'il faut pour y arriver. Ce qui reste achete un call.\n\n"
        "**A savoir dire en entretien** : taux hauts = capital garanti vendable. Taux a zero "
        "= participation ridicule, le marche bascule vers BRC et autocalls. C'est exactement "
        "le mouvement observe entre 2015 et 2021, puis l'inverse depuis.\n\n"
        "**Le skew joue en faveur du client ici** : le call est a la monnaie, donc price avec "
        "une vol plus basse que les puts bas. Mets le skew a 0 dans la barre laterale, la "
        "participation bouge a peine. Sur un BRC le coupon s'effondrerait."
    )


# =====================================================================
# 3 - BRC
# =====================================================================

elif produit.startswith("3"):
    st.title("Barrier Reverse Convertible")
    st.caption("Le client ne recoit pas un interet. Il encaisse une prime d'assurance.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 0.25, 3.0, 1.0, 0.25)
    bar = c2.slider("Barriere (% du spot)", 50, 95, 70, 5)
    strike = c3.slider("Strike (% du spot)", 80, 110, 100, 5)

    obs = st.radio("Observation de la barriere",
                   ["Continue (americaine)", "A l'echeance (europeenne)"],
                   horizontal=True)
    continu = obs.startswith("Continue")

    n_steps = max(int(252 * T), 60)
    vol_eff = vol_au_strike(bar / 100, vol_atm, skew)
    ch = simule(S0, T, r, q, vol_eff, n_paths, n_steps)
    cpn = coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu, marge, S0)

    ST = ch[:, -1]
    touche = (ch.min(axis=1) <= bar / 100 * S0) if continu else (ST <= bar / 100 * S0)
    perte = touche & (ST < strike / 100 * S0)

    m = st.columns(4)
    m[0].metric("Coupon equitable", f"{cpn * 100:,.2f}% p.a.")
    m[1].metric("Vol utilisee", f"{vol_eff * 100:,.1f}%",
                delta=f"{(vol_eff - vol_atm) * 100:+,.1f} pts vs ATM")
    m[2].metric("Proba de toucher", f"{touche.mean() * 100:,.1f}%")
    m[3].metric("Proba de perte en capital", f"{perte.mean() * 100:,.1f}%")

    # --- Grecques : bump & revalue, meme graine pour les deux evaluations
    h = 0.01
    pv_up = pv_brc(simule(S0 * (1 + h), T, r, q, vol_eff, n_paths, n_steps),
                   T, r, nominal, bar / 100, strike / 100, continu, cpn, S0)
    pv_dn = pv_brc(simule(S0 * (1 - h), T, r, q, vol_eff, n_paths, n_steps),
                   T, r, nominal, bar / 100, strike / 100, continu, cpn, S0)
    delta = (pv_up - pv_dn) / 2.0
    pv_vu = pv_brc(simule(S0, T, r, q, vol_eff + 0.01, n_paths, n_steps),
                   T, r, nominal, bar / 100, strike / 100, continu, cpn, S0)
    pv_vd = pv_brc(simule(S0, T, r, q, max(vol_eff - 0.01, 0.01), n_paths, n_steps),
                   T, r, nominal, bar / 100, strike / 100, continu, cpn, S0)
    vega = (pv_vu - pv_vd) / 2.0

    g = st.columns(2)
    g[0].metric("Delta (pour +1% de spot)", f"{delta:+,.1f} CHF")
    g[1].metric("Vega (pour +1 pt de vol)", f"{vega:+,.1f} CHF")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ech = np.random.default_rng(1).choice(len(ST), min(3000, len(ST)), replace=False)
    pay = np.where(perte, nominal * ST / (strike / 100 * S0), nominal) + cpn * nominal * T
    ax.scatter(ST[ech][~touche[ech]], pay[ech][~touche[ech]], s=4, alpha=0.4,
               color=VERT, label="Barriere intacte")
    ax.scatter(ST[ech][touche[ech]], pay[ech][touche[ech]], s=4, alpha=0.4,
               color=ROUGE, label="Barriere touchee")
    ax.plot(np.sort(ST), nominal * np.sort(ST) / S0, color="black", ls="--", lw=1,
            label="Direct")
    ax.axvline(bar / 100 * S0, color=GRIS, ls=":", lw=1.6)
    ax.set_xlabel("Sous-jacent a l'echeance")
    ax.set_ylabel("Remboursement (CHF)")
    ax.set_title("Meme spot final, resultats differents")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.28)

    ax = axes[1]
    tt = np.linspace(0, T, n_steps + 1)
    for i in range(45):
        ax.plot(tt, ch[i], lw=0.7, alpha=0.65, color=ROUGE if touche[i] else VERT)
    ax.axhline(bar / 100 * S0, color=GRIS, ls=":", lw=1.6)
    ax.axhline(S0, color="black", lw=0.8)
    ax.set_xlabel("Temps (annees)")
    ax.set_ylabel("Sous-jacent")
    ax.set_title("Le chemin compte, pas seulement l'arrivee")
    ax.grid(alpha=0.28)
    st.pyplot(fig)

    st.subheader("D'ou vient le coupon")
    lignes = []
    for v in [0.12, 0.20, 0.30, 0.45, 0.60]:
        ve = vol_au_strike(bar / 100, v, skew)
        c_ = simule(S0, T, r, q, ve, 20000, n_steps, seed=7)
        s_ = c_[:, -1]
        t_ = (c_.min(axis=1) <= bar / 100 * S0) if continu else (s_ <= bar / 100 * S0)
        cp = coupon_brc(c_, T, r, nominal, bar / 100, strike / 100, continu, marge, S0)
        lignes.append({"Vol ATM": f"{v * 100:.0f}%",
                       "Vol au niveau barriere": f"{ve * 100:.1f}%",
                       "Proba de toucher": f"{t_.mean() * 100:,.1f}%",
                       "Coupon": f"{cp * 100:,.2f}% p.a."})
    st.table(lignes)

    st.info(
        "**Le coupon a deux sources** : les taux (la banque garde le cash) et la prime du "
        "put que le client vend. Coupon eleve = sinistre probable. Jamais bonne affaire.\n\n"
        "**Le delta** dit combien d'actions le trader doit detenir pour etre couvert. Il "
        "bouge tous les jours, et pres de la barriere il bascule violemment : c'est le gamma "
        "qui explose. D'ou le prix degrade sur les barrieres proches du spot, et le refus "
        "pur et simple sur les sous-jacents illiquides.\n\n"
        "**Ta fenetre commerciale se lit sur la vol implicite**, pas sur la direction du "
        "marche. Vol haute = coupons attractifs = moment d'appeler tes clients."
    )


# =====================================================================
# 4 - AUTOCALL
# =====================================================================

elif produit.startswith("4"):
    st.title("Autocall Phoenix")
    st.caption("Le produit phare des desks. Coupon conditionnel, rappel anticipe, "
               "barriere de capital a l'echeance.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite maximale (annees)", 1.0, 6.0, 3.0, 0.5)
    freq = c1.selectbox("Frequence d'observation",
                        ["Trimestrielle", "Semestrielle", "Annuelle"])
    trigger = c2.slider("Trigger de rappel (%)", 80, 110, 100, 5)
    bar_cp = c2.slider("Barriere de coupon (%)", 50, 100, 70, 5)
    bar_cap = c3.slider("Barriere de capital (%)", 40, 90, 60, 5)
    memoire = c3.checkbox("Effet memoire", value=True,
                          help="Les coupons manques sont rattrapes au premier "
                               "paiement suivant.")

    par_an = {"Trimestrielle": 4, "Semestrielle": 2, "Annuelle": 1}[freq]
    n_obs = max(int(T * par_an), 1)
    n_steps = max(int(252 * T), 80)

    vol_eff = vol_au_strike(bar_cap / 100, vol_atm, skew)
    ch = simule(S0, T, r, q, vol_eff, n_paths, n_steps)
    cpn = coupon_autocall(ch, T, r, nominal, n_obs, trigger / 100, bar_cp / 100,
                          bar_cap / 100, memoire, marge, S0)
    duree, p_rappel, p_perte = stats_autocall(ch, T, n_obs, trigger / 100,
                                              bar_cap / 100, S0)

    m = st.columns(4)
    m[0].metric("Coupon par observation", f"{cpn * 100:,.2f}%")
    m[1].metric("Equivalent annuel", f"{cpn * par_an * 100:,.2f}% p.a.")
    m[2].metric("Duree de vie moyenne", f"{duree:,.2f} ans")
    m[3].metric("Proba de perte en capital", f"{p_perte * 100:,.1f}%")

    st.subheader("Trajectoires et calendrier de sortie")
    indices = idx_obs(n_steps, n_obs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    tt = np.linspace(0, T, n_steps + 1)
    for i in range(50):
        chem = ch[i]
        rappel_a = None
        for j in indices[:-1]:
            if chem[j] >= trigger / 100 * S0:
                rappel_a = j
                break
        if rappel_a is not None:
            ax.plot(tt[:rappel_a + 1], chem[:rappel_a + 1], lw=0.8, alpha=0.7, color=VERT)
            ax.scatter([tt[rappel_a]], [chem[rappel_a]], s=20, color=VERT, zorder=5)
        else:
            col = ROUGE if chem[-1] < bar_cap / 100 * S0 else GRIS
            ax.plot(tt, chem, lw=0.8, alpha=0.7, color=col)
    ax.axhline(trigger / 100 * S0, color=VERT, ls="--", lw=1.2)
    ax.axhline(bar_cp / 100 * S0, color=BLEU, ls=":", lw=1.2)
    ax.axhline(bar_cap / 100 * S0, color=ROUGE, ls=":", lw=1.4)
    ax.text(0.02, trigger / 100 * S0 + 2, "trigger de rappel", fontsize=7, color=VERT)
    ax.text(0.02, bar_cp / 100 * S0 + 2, "barriere coupon", fontsize=7, color=BLEU)
    ax.text(0.02, bar_cap / 100 * S0 - 6, "barriere capital", fontsize=7, color=ROUGE)
    ax.set_xlabel("Temps (annees)")
    ax.set_ylabel("Sous-jacent")
    ax.set_title("Vert = rappele tot. Rouge = perte en capital.")
    ax.grid(alpha=0.28)

    ax = axes[1]
    probas, vivant = [], np.ones(len(ch), dtype=bool)
    for j in indices[:-1]:
        rappel = vivant & (ch[:, j] >= trigger / 100 * S0)
        probas.append(rappel.mean())
        vivant = vivant & ~rappel
    probas.append(vivant.mean())
    labels = [f"Obs {k + 1}" for k in range(len(probas) - 1)] + ["Echeance"]
    ax.bar(labels, np.array(probas) * 100,
           color=[VERT] * (len(probas) - 1) + [GRIS])
    ax.set_ylabel("Probabilite (%)")
    ax.set_title("Quand le produit se termine")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(alpha=0.28, axis="y")
    st.pyplot(fig)

    st.subheader("Effet de la barriere de capital")
    lignes = []
    for b in [40, 50, 60, 70, 80]:
        ve = vol_au_strike(b / 100, vol_atm, skew)
        c_ = simule(S0, T, r, q, ve, 20000, n_steps, seed=11)
        cp = coupon_autocall(c_, T, r, nominal, n_obs, trigger / 100, bar_cp / 100,
                             b / 100, memoire, marge, S0)
        _, _, pp = stats_autocall(c_, T, n_obs, trigger / 100, b / 100, S0)
        lignes.append({"Barriere capital": f"{b}%",
                       "Coupon / obs": f"{cp * 100:,.2f}%",
                       "Equivalent annuel": f"{cp * par_an * 100:,.2f}%",
                       "Proba de perte": f"{pp * 100:,.1f}%"})
    st.table(lignes)

    st.info(
        "**Pourquoi ce produit domine le marche** : le client touche des coupons meme quand "
        f"le sous-jacent baisse un peu, et recupere son cash tot si ca monte. Duree de vie "
        f"moyenne ici : {duree:,.2f} ans sur {T:,.1f} ans possibles.\n\n"
        "**Le piege commercial a connaitre** : le rappel anticipe coupe les coupons futurs. "
        "Un client qui compare le coupon affiche a un rendement obligataire se trompe : il "
        "ne le touchera peut-etre qu'une ou deux fois avant d'etre rembourse.\n\n"
        "**Ce qu'il manque encore ici** : la plupart des autocalls vendus sont sur panier "
        "worst-of, ou le plus mauvais des trois sous-jacents decide de tout. Le coupon monte "
        "fortement, le risque aussi, et la correlation devient un parametre de pricing "
        "central. C'est la prochaine brique a ajouter."
    )


# =====================================================================
# 5 - BONUS CERTIFICATE
# =====================================================================

elif produit.startswith("5"):
    st.title("Bonus Certificate")
    st.caption("Un niveau bonus garanti si la barriere tient, et la hausse reste illimitee.")

    c1, c2 = st.columns(2)
    T = c1.slider("Maturite (annees)", 0.5, 4.0, 2.0, 0.25)
    bar = c2.slider("Barriere (% du spot)", 50, 90, 70, 5)

    n_steps = max(int(252 * T), 60)
    vol_eff = vol_au_strike(bar / 100, vol_atm, skew)
    ch = simule(S0, T, r, q, vol_eff, n_paths, n_steps)
    ST = ch[:, -1]
    touche = ch.min(axis=1) <= bar / 100 * S0

    def pv_bonus(niveau):
        pay = np.where(touche, nominal * ST / S0,
                       np.maximum(nominal * ST / S0, nominal * niveau))
        return math.exp(-r * T) * pay.mean()

    lo, hi, cible = 1.0, 2.5, nominal * (1.0 - marge * T)
    for _ in range(45):
        mid = (lo + hi) / 2
        if pv_bonus(mid) > cible:
            hi = mid
        else:
            lo = mid
    bonus = lo

    m = st.columns(3)
    m[0].metric("Niveau bonus offert", f"{bonus * 100:,.1f}%")
    m[1].metric("Rendement si marche lateral", f"{(bonus - 1) * 100:,.1f}%")
    m[2].metric("Proba de toucher la barriere", f"{touche.mean() * 100:,.1f}%")

    spots = np.linspace(30, 200, 500)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spots, np.maximum(nominal * spots / S0, nominal * bonus), lw=2.6,
            color=VERT, label="Barriere intacte")
    ax.plot(spots, nominal * spots / S0, lw=2.2, color=ROUGE, label="Barriere touchee")
    ax.axhline(nominal * bonus, color=GRIS, lw=0.8)
    ax.axvline(bar / 100 * S0, color=GRIS, ls=":", lw=1.6)
    cadre(ax)
    st.pyplot(fig)

    st.info(
        "**Decomposition** : detention du sous-jacent sans dividendes, plus achat d'un put "
        "down-and-out de strike egal au niveau bonus. Les dividendes abandonnes financent "
        "le put.\n\n"
        "**Ce qui le distingue du BRC** : pas de plafond a la hausse. Si le sous-jacent "
        "explose, le client suit. Il paie ca par un bonus plus modeste qu'un coupon de BRC.\n\n"
        "**A ne jamais laisser passer en clientele** : la protection est conditionnelle et "
        "disparait entierement si la barriere casse. Ce n'est pas une garantie."
    )


# =====================================================================
# 6 - TWIN-WIN
# =====================================================================

else:
    st.title("Twin-Win")
    st.caption("Gagne a la hausse ET a la baisse moderee, tant que la barriere tient.")

    c1, c2 = st.columns(2)
    T = c1.slider("Maturite (annees)", 0.5, 4.0, 2.0, 0.25)
    bar = c2.slider("Barriere (% du spot)", 50, 90, 65, 5)

    n_steps = max(int(252 * T), 60)
    vol_eff = vol_au_strike(bar / 100, vol_atm, skew)
    ch = simule(S0, T, r, q, vol_eff, n_paths, n_steps)
    ST = ch[:, -1]
    touche = ch.min(axis=1) <= bar / 100 * S0

    def pv_twin(pb):
        perf = ST / S0 - 1
        pay = np.where(touche, nominal * ST / S0,
                       np.where(perf >= 0, nominal * (1 + perf),
                                nominal * (1 + pb * (-perf))))
        return math.exp(-r * T) * pay.mean()

    lo, hi, cible = 0.0, 3.0, nominal * (1.0 - marge * T)
    for _ in range(45):
        mid = (lo + hi) / 2
        if pv_twin(mid) > cible:
            hi = mid
        else:
            lo = mid
    part_bas = lo

    m = st.columns(3)
    m[0].metric("Participation a la baisse", f"{part_bas * 100:,.0f}%")
    m[1].metric("Participation a la hausse", "100%")
    m[2].metric("Proba de toucher", f"{touche.mean() * 100:,.1f}%")

    spots = np.linspace(30, 200, 500)
    perf = spots / S0 - 1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spots, np.where(perf >= 0, nominal * (1 + perf),
                            nominal * (1 + part_bas * (-perf))),
            lw=2.6, color=VERT, label="Barriere intacte")
    ax.plot(spots, nominal * spots / S0, lw=2.2, color=ROUGE, label="Barriere touchee")
    ax.axvline(bar / 100 * S0, color=GRIS, ls=":", lw=1.6)
    cadre(ax)
    st.pyplot(fig)

    st.info(
        "**Le produit qui impressionne les clients, et qu'il faut savoir demonter.** Une "
        "baisse de 20% rapporte de l'argent, tant que la barriere n'a pas casse.\n\n"
        "**Decomposition** : tracker plus achat d'un put down-and-out. Le put transforme la "
        "baisse en gain et s'evapore si la barriere casse.\n\n"
        "**Le point a ne jamais laisser passer** : entre la barriere et le niveau initial, "
        "le client gagne. Sous la barriere, il perd d'un coup toute cette zone de gain. "
        "La discontinuite est brutale : c'est le profil le plus dangereux a mal expliquer."
    )


# =====================================================================
# PIED DE PAGE
# =====================================================================

st.markdown("---")
with st.expander("Limites du modele - a lire avant de montrer cette appli"):
    st.markdown("""
**1. Volatilite constante par produit.**
On simule en Black-Scholes avec une vol unique, choisie au niveau de strike pertinent.
C'est une approximation praticien. Un vrai desk interpole dans une surface de vol calibree
au marche, voire utilise un modele a volatilite locale ou stochastique. Les prix ici sont
dans le bon ordre de grandeur, pas au niveau marche.

**2. Pas de sauts.**
Le mouvement brownien geometrique ne genere pas de gap de -20% en une seance. Or c'est
exactement ce qui fait mal sur un produit a barriere. Les pertes en queue de distribution
sont sous-estimees.

**3. Pas de risque emetteur.**
Un produit structure est une dette de la banque. En 2008, les detenteurs de produits Lehman
ont tout perdu alors meme que leurs barrieres tenaient. Ce risque est absent du pricing.

**4. Mono-sous-jacent.**
La majorite des autocalls vendus sont sur panier worst-of. La correlation devient alors un
parametre de pricing central, et le risque reel est bien superieur a ce que suggere un
sous-jacent unique.

**5. Pas de frais de transaction, pas de bid-ask, pas de cout de hedge.**
La marge est modelisee comme un prelevement forfaitaire, ce qui est une simplification.
""")

st.caption("Tom Uzan - EDHEC BBA Finance - outil pedagogique, aucune valeur d'offre.")
