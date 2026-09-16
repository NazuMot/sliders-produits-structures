"""
=======================================================================
SLIDERS PRODUITS STRUCTURES  -  v3
Tom Uzan - EDHEC BBA Finance
=======================================================================

Moteur :
  - simulation hebdomadaire en float32 (memoire divisee par ~40 vs v2)
  - correction par pont brownien pour les barrieres continues :
    plus rapide ET plus precis qu'une discretisation quotidienne
  - variables antithetiques
  - analyses de sensibilite derriere une case a cocher, pour que les
    curseurs restent instantanes

Lancer avec :  streamlit run app.py
=======================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sliders Produits Structures", layout="wide")

ROUGE, VERT, BLEU, GRIS = "#C8102E", "#2E7D32", "#1565C0", "#6E6E6E"
S0 = 100.0
PAS_PAR_AN = 52          # pas hebdomadaires : le pont brownien corrige le reste
MAX_CACHE = 6            # peu d'entrees en cache = peu de RAM


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

    Les puts bas coutent plus cher en vol que les options a la monnaie :
    les actions tombent plus vite qu'elles ne montent, et les
    institutionnels achetent des puts sans vendeur naturel en face.

    skew_pts = points de vol gagnes pour 10% de baisse du strike.
    Approximation, pas un modele a volatilite locale.
    """
    return max(vol_atm + skew_pts / 100.0 * (1.0 - strike_pct) / 0.10, 0.01)


def nb_pas(T):
    return max(int(round(PAS_PAR_AN * T)), 12)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def simule(spot, T, r, q, sigma, n_paths, seed=42):
    """
    Trajectoires en mouvement brownien geometrique, pas hebdomadaires.

      - drift (r - q) : univers risque-neutre, pas une prevision
      - variables antithetiques : chaque tirage double par son oppose
      - float32 : moitie moins de memoire, precision largement suffisante
    """
    n_steps = nb_pas(T)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = max(n_paths // 2, 1)
    z = rng.standard_normal((half, n_steps), dtype=np.float32)
    z = np.vstack([z, -z])
    incr = np.float32((r - q - 0.5 * sigma ** 2) * dt) + np.float32(sigma * math.sqrt(dt)) * z
    ch = spot * np.exp(np.cumsum(incr, axis=1, dtype=np.float32))
    return np.hstack([np.full((ch.shape[0], 1), spot, dtype=np.float32), ch])


def proba_non_touche(chemins, B, sigma, T):
    """
    Correction par pont brownien.

    Une simulation par pas discrets rate les franchissements qui se
    produisent ENTRE deux observations : elle sous-estime donc la
    probabilite de toucher, et surestime le prix du produit.

    Entre deux points S1 et S2 tous deux au-dessus de B, la probabilite
    que le minimum du pont brownien soit passe sous B vaut :

        p = exp( -2 * ln(S1/B) * ln(S2/B) / (sigma^2 * dt) )

    On multiplie les (1 - p) sur tous les pas : on obtient la probabilite
    exacte de ne jamais toucher, sans simuler jour par jour.

    Resultat : plus rapide ET plus juste qu'une discretisation quotidienne.
    """
    n_steps = chemins.shape[1] - 1
    dt = T / n_steps
    s1, s2 = chemins[:, :-1], chemins[:, 1:]
    x1 = np.log(np.maximum(s1, 1e-9) / B)
    x2 = np.log(np.maximum(s2, 1e-9) / B)
    with np.errstate(over="ignore", invalid="ignore"):
        p = np.exp(-2.0 * x1 * x2 / (sigma ** 2 * dt))
    p = np.where((x1 <= 0) | (x2 <= 0), 1.0, p)
    return np.clip(np.prod(1.0 - p, axis=1), 0.0, 1.0)


def idx_obs(n_steps, n_obs):
    return [int(round(n_steps * (k + 1) / n_obs)) for k in range(n_obs)]


# =====================================================================
# PRICERS
# =====================================================================

def pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, coupon_pa,
           sigma, spot0=S0):
    B, K = bar_pct * spot0, strike_pct * spot0
    ST = chemins[:, -1].astype(np.float64)
    if continu:
        p_no = proba_non_touche(chemins, B, sigma, T)
    else:
        p_no = (ST > B).astype(np.float64)
    bas = nominal * ST / K
    remb = np.where(ST >= K, nominal, p_no * nominal + (1.0 - p_no) * bas)
    return math.exp(-r * T) * (remb + coupon_pa * T * nominal).mean()


def coupon_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, marge_pa,
               sigma, spot0=S0):
    """Payoff affine en coupon -> resolution directe, sans iteration."""
    base = pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, 0.0,
                  sigma, spot0)
    unite = math.exp(-r * T) * nominal * T
    return max((nominal * (1.0 - marge_pa * T) - base) / unite, 0.0) if unite > 0 else 0.0


def pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                memoire, coupon, spot0=S0):
    """
    Autocall Phoenix a memoire.

    A chaque observation :
      spot >= trigger         -> remboursement anticipe du nominal
      spot >= barriere coupon -> coupon paye (+ coupons manques si memoire)
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
        disc = math.exp(-r * (k + 1) * dt_obs)
        S = chemins[:, i].astype(np.float64)
        paye = vivant & (S >= bar_cp * spot0)
        n_cp = (manques + 1.0) if memoire else np.ones(n_paths)
        pv += np.where(paye, coupon * nominal * n_cp * disc, 0.0)
        manques = np.where(paye, 0.0, manques + 1.0)
        if k < len(idx) - 1:
            rappel = vivant & (S >= trigger * spot0)
            pv += np.where(rappel, nominal * disc, 0.0)
            vivant &= ~rappel

    ST = chemins[:, -1].astype(np.float64)
    final = np.where(ST >= bar_cap * spot0, nominal, nominal * ST / spot0)
    pv += np.where(vivant, final * math.exp(-r * T), 0.0)
    return pv.mean()


def coupon_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, marge_pa, spot0=S0):
    a = pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, 0.0, spot0)
    b = pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, 0.01, spot0)
    pente = (b - a) / 0.01
    return max((nominal * (1.0 - marge_pa * T) - a) / pente, 0.0) if pente > 0 else 0.0


def stats_autocall(chemins, T, n_obs, trigger, bar_cap, spot0=S0):
    n_paths, n_steps = chemins.shape[0], chemins.shape[1] - 1
    idx = idx_obs(n_steps, n_obs)
    dt_obs = T / n_obs
    vivant = np.ones(n_paths, dtype=bool)
    duree = np.full(n_paths, T)
    for k, i in enumerate(idx[:-1]):
        rappel = vivant & (chemins[:, i] >= trigger * spot0)
        duree = np.where(rappel, (k + 1) * dt_obs, duree)
        vivant &= ~rappel
    ST = chemins[:, -1]
    return duree.mean(), 1.0 - vivant.mean(), (vivant & (ST < bar_cap * spot0)).mean()


# --- fonctions lourdes mises en cache pour ne pas ralentir les curseurs

@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def table_sensi_brc(T, r, q, skew, marge, bar, strike, continu, nominal, n_paths):
    lignes = []
    for v in [0.12, 0.20, 0.30, 0.45, 0.60]:
        ve = vol_au_strike(bar, v, skew)
        ch = simule(S0, T, r, q, ve, min(n_paths, 20000), seed=7)
        p_no = (proba_non_touche(ch, bar * S0, ve, T) if continu
                else (ch[:, -1] > bar * S0).astype(np.float64))
        cp = coupon_brc(ch, T, r, nominal, bar, strike, continu, marge, ve)
        lignes.append({"Vol ATM": f"{v * 100:.0f}%",
                       "Vol au niveau barriere": f"{ve * 100:.1f}%",
                       "Proba de toucher": f"{(1 - p_no.mean()) * 100:,.1f}%",
                       "Coupon": f"{cp * 100:,.2f}% p.a."})
    return lignes


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def table_sensi_autocall(T, r, q, vol_atm, skew, marge, n_obs, trigger, bar_cp,
                         memoire, nominal, n_paths, par_an):
    lignes = []
    for b in [0.40, 0.50, 0.60, 0.70, 0.80]:
        ve = vol_au_strike(b, vol_atm, skew)
        ch = simule(S0, T, r, q, ve, min(n_paths, 20000), seed=11)
        cp = coupon_autocall(ch, T, r, nominal, n_obs, trigger, bar_cp, b,
                             memoire, marge)
        _, _, pp = stats_autocall(ch, T, n_obs, trigger, b)
        lignes.append({"Barriere capital": f"{b * 100:.0f}%",
                       "Coupon / obs": f"{cp * 100:,.2f}%",
                       "Equivalent annuel": f"{cp * par_an * 100:,.2f}%",
                       "Proba de perte": f"{pp * 100:,.1f}%"})
    return lignes


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def grecques_brc(T, r, q, vol_eff, bar, strike, continu, cpn, nominal, n_paths):
    h = 0.01
    up = pv_brc(simule(S0 * (1 + h), T, r, q, vol_eff, n_paths), T, r, nominal,
                bar, strike, continu, cpn, vol_eff)
    dn = pv_brc(simule(S0 * (1 - h), T, r, q, vol_eff, n_paths), T, r, nominal,
                bar, strike, continu, cpn, vol_eff)
    vu = pv_brc(simule(S0, T, r, q, vol_eff + h, n_paths), T, r, nominal,
                bar, strike, continu, cpn, vol_eff + h)
    vd = pv_brc(simule(S0, T, r, q, max(vol_eff - h, 0.01), n_paths), T, r, nominal,
                bar, strike, continu, cpn, max(vol_eff - h, 0.01))
    return (up - dn) / 2.0, (vu - vd) / 2.0


# =====================================================================
# PARAMETRES DE MARCHE
# =====================================================================

st.sidebar.title("Parametres de marche")
st.sidebar.caption("Une seule vue de marche alimente tous les produits, "
                   "comme sur un desk.")

r = st.sidebar.slider(
    "Taux sans risque (%)", 0.0, 7.0, 3.0, 0.25,
    help="Le taux auquel la banque place ou emprunte du cash sans risque. C'est le "
         "carburant du budget option : pour garantir 1 000 CHF dans 5 ans a 3%, la "
         "banque ne met que 861 aujourd'hui, et les 139 restants achetent des options. "
         "A 0,5% elle doit mettre 975 et il ne reste que 25.") / 100

q = st.sidebar.slider(
    "Taux de dividende (%)", 0.0, 7.0, 2.5, 0.25,
    help="Ce que verse le sous-jacent chaque annee. Le client d'un produit structure ne "
         "le touche PAS : il reste a la banque et finance la structure. C'est la source "
         "de financement la plus invisible et la plus importante.") / 100

vol_atm = st.sidebar.slider(
    "Volatilite implicite ATM (%)", 8.0, 70.0, 22.0, 1.0,
    help="Ce n'est PAS une prevision. C'est le prix d'une option exprime dans une autre "
         "unite. ATM = strike au niveau du spot. Vol haute = coupons eleves la ou le "
         "client vend de l'optionalite, mais participation degradee la ou il en "
         "achete.") / 100

skew = st.sidebar.slider(
    "Skew (pts de vol / -10% de strike)", 0.0, 6.0, 2.0, 0.5,
    help="Les puts bas coutent plus cher en vol que les options a la monnaie : les "
         "actions tombent plus vite qu'elles ne montent, et les institutionnels "
         "achetent des puts sans vendeur naturel en face. Mets 0 pour voir le coupon "
         "du BRC s'effondrer.")

marge = st.sidebar.slider(
    "Marge banque (% par an)", 0.0, 2.5, 0.8, 0.1,
    help="Prelevee sur le budget option, invisible dans le payoff. Elle sort "
         "directement du coupon ou de la participation offerte.") / 100

st.sidebar.markdown("---")

nominal = float(st.sidebar.select_slider(
    "Nominal (CHF)", [1000, 10000, 100000, 1000000], 1000,
    help="Ne change aucun pourcentage, seulement l'echelle des montants affiches."))

n_paths = st.sidebar.select_slider(
    "Trajectoires Monte Carlo", [5000, 10000, 20000, 40000], 10000,
    help="Nombre de scenarios simules. Plus il y en a, plus le resultat est stable, "
         "mais plus c'est lent. 10 000 suffit largement ici grace au pont brownien "
         "et aux variables antithetiques.")

st.sidebar.markdown("---")
produit = st.sidebar.radio(
    "Produit",
    ["Comparateur", "Glossaire",
     "1 - Tracker Certificate",
     "2 - Capital Protection",
     "3 - Barrier Reverse Convertible",
     "4 - Autocall Phoenix",
     "5 - Bonus Certificate",
     "6 - Twin-Win"])

st.sidebar.markdown("---")
st.sidebar.caption("Sous-jacent initial fixe a 100.")


def cadre(ax):
    ax.axhline(nominal, color=GRIS, lw=0.7)
    ax.axvline(S0, color=GRIS, lw=0.7)
    ax.set_xlabel("Sous-jacent a l'echeance")
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

    T = st.slider("Maturite (annees)", 1.0, 5.0, 2.0, 0.5,
                  help="Maturite commune imposee aux quatre structures pour que la "
                       "comparaison ait un sens.")

    v_brc = vol_au_strike(0.70, vol_atm, skew)
    v_ac = vol_au_strike(0.65, vol_atm, skew)
    ch_brc = simule(S0, T, r, q, v_brc, n_paths)
    ch_ac = simule(S0, T, r, q, v_ac, n_paths)

    zc = math.exp(-r * T)
    budget = 1.0 - zc - marge * T
    call_u = bs_call(S0, S0, T, r, vol_au_strike(1.0, vol_atm, skew), q) / S0
    part = max(budget / call_u, 0.0) if call_u > 0 else 0.0

    cpn_brc = coupon_brc(ch_brc, T, r, nominal, 0.70, 1.0, True, marge, v_brc)
    n_obs = max(int(T * 4), 1)
    cpn_ac = coupon_autocall(ch_ac, T, r, nominal, n_obs, 1.0, 0.70, 0.65, True, marge)
    duree, _, p_perte = stats_autocall(ch_ac, T, n_obs, 1.0, 0.65)

    st.subheader("Ce que le desk peut offrir aujourd'hui")
    st.table([
        {"Produit": "Tracker", "Offre": "Participation 100%", "Protection": "aucune",
         "Plafond hausse": "aucun", "Vue client": "haussier franc"},
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
                f"{(ch_ac[:, -1] > S0).mean() * 100:,.1f}%",
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
    plt.close(fig)

    st.info(
        "**La regle qui structure tout le metier** : protection, participation, coupon. "
        "Jamais trois sur trois. Le budget est fixe : il vient des taux, des dividendes, "
        "et de ce que le client accepte de vendre comme optionalite.\n\n"
        "**Ta question au client** : quelle est ta vue, et qu'es-tu pret a abandonner "
        "pour l'exprimer ?")


# =====================================================================
# GLOSSAIRE
# =====================================================================

elif produit == "Glossaire":
    st.title("Glossaire")
    st.caption("Tout le vocabulaire de l'appli, et ce que chaque terme implique "
               "concretement pour un Sales.")

    o = st.tabs(["Parametres de marche", "Mecaniques produit", "Les grecques",
                 "Volatilite", "Jargon de desk"])

    with o[0]:
        st.markdown("""
### Taux sans risque (r)
Le taux auquel la banque place ou emprunte du cash sans risque.

**Pourquoi ca compte** : c'est le carburant du budget option. Pour garantir 1 000 CHF
dans 5 ans a 3%, la banque ne met que 861 aujourd'hui. Les 139 restants achetent des
options. A 0,5%, elle doit mettre 975 et il ne reste que 25.

**Consequence commerciale** : taux hauts = capital garanti vendable. Taux bas = le
marche bascule vers BRC et autocalls.

### Taux de dividende (q)
Ce que verse le sous-jacent chaque annee. Le detenteur d'un produit structure ne le
touche pas. Sur un tracker 3 ans avec 2,5% de dividende, le client abandonne environ
7,8% de rendement sans le voir. C'est la source de financement la plus invisible.

### Marge banque
Prelevee sur le budget option. Elle sort directement du coupon ou de la participation
offerte. Reglementairement divulguee (PRIIP, guidelines SSPA sur les couts), mais elle
ne se lit jamais dans le payoff.

### Monte Carlo
Pricing par simulation : on genere des milliers de trajectoires possibles, on calcule
le payoff dans chacune, on fait la moyenne actualisee. Indispensable des qu'un produit
depend du **chemin** et pas seulement du prix final.

### Univers risque-neutre
Dans les simulations, le sous-jacent monte en moyenne au taux (r - q), pas selon une
prevision de marche. **Ce n'est pas un pari directionnel** : c'est la seule hypothese
qui interdit l'arbitrage.

### Pont brownien
La correction utilisee ici pour les barrieres. Une simulation par pas discrets rate les
franchissements qui se produisent entre deux observations, donc elle sous-estime la
probabilite de toucher. Le pont brownien calcule analytiquement cette probabilite entre
deux points. Resultat : plus rapide **et** plus juste qu'une simulation quotidienne.
""")

    with o[1]:
        st.markdown("""
### Strike
Le niveau de reference qui decide du remboursement. Standard : 100% du spot initial.

### Barriere
Un niveau qui declenche un changement de regime. Trois choses a savoir :

- **Toucher une barriere ne fait pas perdre.** Ca arme le risque. C'est le strike qui
  decide a la fin. Une action qui descend a 72, touche la barriere a 75, puis remonte
  a 103 : le client touche tout son capital plus le coupon.
- **Observation continue (americaine)** : la barriere compte a tout moment, meme une
  meche intraday. Standard marche.
- **Observation europeenne** : seul le niveau a l'echeance compte. Bien plus favorable
  au client, donc coupon plus faible.

### Knock-in / Knock-out
- **Knock-in** : l'option s'active quand la barriere est franchie. Le put d'un BRC.
- **Knock-out** : l'option disparait quand la barriere est franchie. Le put d'un Bonus
  Certificate ou d'un Twin-Win.

### Participation
Le ratio de suivi du sous-jacent. 70% : le sous-jacent fait +20%, le client touche +14%.
**Elle n'est jamais choisie, elle est calculee** : budget option divise par prix du call.

### Coupon
**Ce n'est pas un interet.** C'est le prix d'une option que le client vend a la banque.
Deux sources : les taux, et la prime du put vendu.

Coupon eleve = sinistre probable. Quand un client dit "celui-la paie 15%, c'est le
meilleur", la bonne reponse est : "c'est le plus risque - sur quel sous-jacent, et a
quelle barriere ?"

### Coupon conditionnel vs inconditionnel
- **Inconditionnel** (BRC) : paye quoi qu'il arrive.
- **Conditionnel** (Phoenix) : paye seulement si le sous-jacent est au-dessus de la
  barriere de coupon a la date d'observation.

### Effet memoire
Les coupons manques sont mis de cote et rattrapes au premier paiement suivant. Tres bon
argument de vente : un trou passager ne coute rien si le sous-jacent remonte.

### Trigger de rappel
Si le sous-jacent est au-dessus a une date d'observation, le produit s'arrete.
**Le piege** : le rappel coupe tous les coupons futurs. Un client qui compare un coupon
de 8% p.a. a un rendement obligataire se trompe.

### Livraison physique
Si la barriere casse et que le sous-jacent finit sous le strike, le client recoit des
actions au lieu de son cash. Il devient actionnaire malgre lui.

### Worst-of
Panier ou le plus mauvais des sous-jacents decide de tout. Coupon bien plus eleve,
risque bien plus eleve, corrrelation comme parametre de pricing central. C'est la
version dominante des autocalls reellement vendus.

### Risque emetteur
**Un produit structure est une dette de la banque**, pas une detention d'actions. En
2008, les porteurs de produits Lehman ont tout perdu alors que leurs barrieres tenaient.
Premier point sur lequel un institutionnel te challengera.
""")

    with o[2]:
        st.markdown("""
Les grecques mesurent la sensibilite du prix a un parametre. Un Sales ne les calcule
pas, il doit savoir ce qu'elles impliquent sur le prix qu'il obtient de son desk.

| Grecque | Ce que c'est | Ce que ca change pour toi |
|---|---|---|
| **Delta** | Sensibilite au prix du sous-jacent | Le nombre d'actions que le trader doit detenir pour etre couvert. Il l'ajuste tous les jours : c'est le delta-hedging. |
| **Gamma** | Vitesse a laquelle le delta change | Explose pres d'une barriere : volumes enormes a trader en peu de temps (pin risk). C'est LA raison du prix degrade sur les barrieres proches du spot. |
| **Vega** | Sensibilite a la volatilite implicite | Ta grecque. Vol haute au pricing = coupon eleve. Ta fenetre commerciale. |
| **Theta** | Erosion du prix avec le temps | Le vendeur d'option gagne du theta chaque jour. Un client qui achete un BRC est structurellement long theta. |
| **Rho** | Sensibilite aux taux | Determinante sur le capital garanti, secondaire sur les produits a coupon. |

### Bump and revalue
La methode utilisee ici : on decale un parametre de peu, on reprice, on regarde l'ecart.

**Le detail technique qui compte** : il faut reutiliser exactement les memes tirages
aleatoires pour les deux evaluations (common random numbers). Sinon le bruit Monte
Carlo domine completement la difference et la grecque est inexploitable.
""")

    with o[3]:
        st.markdown("""
### Volatilite implicite
**Ce n'est pas une prevision.** C'est le prix d'une option exprime dans une autre unite.
Une option vaut 55 CHF, ou elle vaut 22% de vol : meme information. On utilise la vol
parce que ca permet de comparer des options sur des sous-jacents differents.

### ATM / ITM / OTM
At the money (strike au spot) / in the money (valeur intrinseque) / out of the money.

### Skew
Les puts bas coutent plus cher **en vol** que les options a la monnaie. Exemple typique
sur indice a 1 an : put 70% a 26%, ATM a 18%, call 110% a 16%.

**Deux causes, les deux valides** :
1. Les actions ne baissent pas comme elles montent. Un krach de -20% en une semaine
   existe ; une hausse de +20% en une semaine, non. Queue gauche epaisse.
2. Les institutionnels achetent massivement des puts de protection, sans vendeur
   naturel en face.

**Ce que ca change** : le skew fait monter le coupon d'un BRC (le client vend un put bas,
donc cher) et ameliore la participation d'un capital garanti (la banque achete un call
ATM, donc moins cher). Il profite au client dans les deux cas, pour des raisons opposees.

### Terme structure
La vol varie aussi avec la maturite. En regime normal, la vol longue est au-dessus de la
courte. En stress, ca s'inverse : la vol 1 mois explose alors que la vol 2 ans bouge peu,
parce que le marche sait que la panique retombera.

*Pour toi* : en periode de stress, les BRC courts offrent des coupons exceptionnels.
C'est le moment d'appeler tes clients.

### Surface de volatilite
Skew (par strike) + terme structure (par maturite) = une surface a deux dimensions.
C'est l'ecran du trader et la matiere premiere du structureur.

### Vol implicite vs vol realisee
- **Implicite** : ce que le marche fait payer aujourd'hui pour l'avenir.
- **Realisee** : ce qui s'est effectivement passe, mesure apres coup.

L'implicite est structurellement au-dessus de la realisee, de 2 a 4 points sur les
indices. Cet ecart s'appelle la **variance risk premium**.

**C'est la raison d'etre economique des BRC et des autocalls.** Le client qui les achete
capture systematiquement cette prime. Ni hasard ni anomalie : c'est la remuneration de
celui qui accepte d'etre expose aux krachs. Il gagne neuf annees sur dix, et perd
beaucoup la dixieme.
""")

    with o[4]:
        st.markdown("""
### Qui fait quoi

| Role | Ce qu'il fait |
|---|---|
| **Sales (toi)** | Interface client. Tu traduis une vue de marche en parametres, tu fais pricer en interne, tu ramenes le prix. **Tu ne portes aucun risque.** |
| **Structureur** | Assemble le produit, calcule ce que la banque peut offrir, redige le termsheet. |
| **Trader** | Porte le risque et le couvre. Quand un client achete un BRC, la banque se retrouve acheteuse d'un put qu'elle doit gerer pendant toute la duree. |

### Les deux ventes qu'il ne faut pas confondre
**La vente commerciale** : toi vers le client. Il te donne 1 000 CHF, tu lui livres le
produit. La seule qui apparait dans ton metier au quotidien.

**La vente economique, cachee dans le produit** : le client vers ta banque. Il lui vend
une protection contre la baisse, et le coupon est le prix qu'elle lui paie pour ca.

Une seule contrepartie en face de toi. Les 1 000 vont dans un sens, le coupon revient
dans l'autre.

### Vocabulaire courant

| Terme | Definition |
|---|---|
| Sous-jacent | L'actif dont depend le produit : action, indice, panier. |
| Spot | Le prix actuel du sous-jacent. |
| Termsheet | Le document contractuel : sous-jacent, dates, barrieres, coupon, emetteur. Ce que le client signe. |
| Autocall | Tout produit avec rappel anticipe automatique. |
| Phoenix | Autocall dont les coupons sont conditionnels a une barriere, generalement avec effet memoire. La structure la plus vendue. |
| Reverse convertible | Produit ou le client vend de l'optionalite contre un coupon. Le "reverse" vient de la : c'est l'emetteur qui a le droit de livrer les actions. |
| Call spread | Achat d'un call + vente d'un call de strike plus haut. Moins cher qu'un call seul mais plafonne le gain. Permet de capper un capital garanti pour remonter la participation. |
| Delta-hedging | L'activite quotidienne du trader : ajuster sa position en actions pour rester neutre a la direction. |
| Pin risk | Le risque de couverture quand le sous-jacent stagne autour d'une barriere. Le delta oscille violemment. |
| SSPA | Swiss Structured Products Association. Reference du marche suisse, premier marche mondial. Publie la Swiss Derivative Map. |
| PRIIP / KID | Reglementation europeenne imposant un document d'information standardise pour tout produit structure vendu au retail. |
""")


# =====================================================================
# 1 - TRACKER
# =====================================================================

elif produit.startswith("1"):
    st.title("Tracker Certificate")
    st.caption("Exposition lineaire. Produit d'acces, pas produit de rendement.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 1.0, 10.0, 3.0, 0.5,
                  help="Joue peu sur le payoff, mais beaucoup sur le cumul des frais "
                       "et des dividendes abandonnes.")
    participation = c2.slider("Participation (%)", 50, 150, 100, 5,
                              help="100% = 1:1. Au-dessus de 100% on parle "
                                   "d'outperformance certificate : la banque finance "
                                   "le levier en capturant les dividendes.")
    frais_pa = c3.slider("Frais annuels (%)", 0.0, 2.0, 0.6, 0.1,
                         help="Souvent invisibles pour le client car finances par les "
                              "dividendes non reverses. C'est le vrai cout du produit.")

    spots = np.linspace(30, 180, 400)
    perf = spots / S0 - 1
    payoff = nominal * (1 + participation / 100 * perf) * (1 - frais_pa / 100 * T)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spots, nominal * spots / S0, color="black", ls="--", lw=1.1, label="Direct")
    ax.plot(spots, payoff, lw=2.6, color=ROUGE, label="Tracker")
    cadre(ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Cout reel de la detention indirecte")
    st.table([{"Sous-jacent": f"{p:+d}%",
               "Tracker": f"{nominal * (1 + participation / 100 * p / 100) * (1 - frais_pa / 100 * T):,.0f}",
               "Direct": f"{nominal * (1 + p / 100):,.0f}",
               "Ecart": f"{nominal * (1 + participation / 100 * p / 100) * (1 - frais_pa / 100 * T) - nominal * (1 + p / 100):+,.0f}"}
              for p in [-40, -20, 0, 20, 40]])

    st.info(
        f"**Le dividende est le nerf du produit.** Le sous-jacent verse {q * 100:,.1f}% "
        f"par an que le client ne touche pas : sur {T:,.1f} ans cela represente environ "
        f"{(math.exp(q * T) - 1) * 100:,.1f}% de rendement abandonne. C'est ce qui "
        "finance la structure et la marge.\n\n"
        "**Le risque a nommer en clientele** : le client est creancier de la banque, il "
        "ne detient pas les actions. Difference majeure avec un ETF, et le premier point "
        "sur lequel un institutionnel te challengera.")


# =====================================================================
# 2 - CAPITAL PROTECTION
# =====================================================================

elif produit.startswith("2"):
    st.title("Capital Protection Certificate")
    st.caption("La participation n'est pas un choix commercial. C'est un reste de budget.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 1.0, 10.0, 5.0, 0.5,
                  help="Parametre CLE ici. Plus c'est long, moins la banque doit mettre "
                       "de cote aujourd'hui, donc plus le budget option est gros. "
                       "Premier levier pour remonter une participation trop faible.")
    protection = c2.slider("Niveau de protection (%)", 80, 100, 100, 5,
                           help="90% = le client accepte de perdre 10% maximum. Chaque "
                                "point abandonne libere du budget et remonte la "
                                "participation. L'arbitrage central a poser au client.")
    cap = c3.slider("Cap sur la hausse (%)", 110, 300, 300, 10,
                    help="300 = pas de cap en pratique. Capper vend un call de strike "
                         "haut, ce qui libere du budget et remonte la participation.")

    zc = protection / 100 * math.exp(-r * T)
    budget = 1.0 - zc - marge * T
    vol_call = vol_au_strike(1.0, vol_atm, skew)
    cout = max((bs_call(S0, S0, T, r, vol_call, q)
                - bs_call(S0, cap / 100 * S0, T, r,
                          vol_au_strike(cap / 100, vol_atm, skew), q)) / S0, 1e-9)
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spots, nominal * spots / S0, color="black", ls="--", lw=1.1, label="Direct")
    ax.plot(spots, nominal * (protection / 100 + part * np.maximum(perf, 0)),
            lw=2.6, color=ROUGE, label="Capital Protection")
    cadre(ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sensibilite aux taux, tout le reste constant")
    lignes = []
    for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        z = protection / 100 * math.exp(-t * T)
        b = 1 - z - marge * T
        c_ = max((bs_call(S0, S0, T, t, vol_call, q)
                  - bs_call(S0, cap / 100 * S0, T, t,
                            vol_au_strike(cap / 100, vol_atm, skew), q)) / S0, 1e-9)
        lignes.append({"Taux": f"{t * 100:.0f}%",
                       "Mis de cote": f"{z * nominal:,.0f}",
                       "Budget option": f"{b * nominal:,.0f}",
                       "Participation": f"{max(b / c_, 0) * 100:,.0f}%"})
    st.table(lignes)

    st.info(
        "**Le mecanisme** : la banque doit rembourser le nominal a l'echeance. Elle place "
        "aujourd'hui juste ce qu'il faut pour y arriver. Ce qui reste achete un call.\n\n"
        "**A savoir dire en entretien** : taux hauts = capital garanti vendable. Taux a "
        "zero = participation ridicule, le marche bascule vers BRC et autocalls. C'est le "
        "mouvement observe entre 2015 et 2021, puis l'inverse depuis.\n\n"
        "**Le skew joue en faveur du client ici** : le call est a la monnaie, donc price "
        "avec une vol plus basse que les puts bas. Mets le skew a 0, la participation "
        "bouge a peine. Sur un BRC le coupon s'effondrerait.")


# =====================================================================
# 3 - BRC
# =====================================================================

elif produit.startswith("3"):
    st.title("Barrier Reverse Convertible")
    st.caption("Le client ne recoit pas un interet. Il encaisse une prime d'assurance.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 0.25, 3.0, 1.0, 0.25,
                  help="Plus c'est long, plus le sous-jacent a de temps pour toucher la "
                       "barriere. Les BRC courts se vendent en periode de stress, quand "
                       "la vol courte explose.")
    bar = c2.slider("Barriere (% du spot)", 50, 95, 70, 5,
                    help="Le niveau qui ARME le risque de perte. Le toucher ne fait pas "
                         "perdre : c'est le strike qui decide a la fin. Barriere proche "
                         "du spot = coupon plus eleve et risque plus eleve.")
    strike = c3.slider("Strike (% du spot)", 80, 110, 100, 5,
                       help="Niveau de reference si la barriere a ete touchee. Standard "
                            "100%. Un strike abaisse reduit la perte mais fait baisser "
                            "le coupon.")

    continu = st.radio("Observation de la barriere",
                       ["Continue (americaine)", "A l'echeance (europeenne)"],
                       horizontal=True).startswith("Continue")

    vol_eff = vol_au_strike(bar / 100, vol_atm, skew)
    ch = simule(S0, T, r, q, vol_eff, n_paths)
    cpn = coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu, marge, vol_eff)

    ST = ch[:, -1].astype(np.float64)
    p_no = (proba_non_touche(ch, bar / 100 * S0, vol_eff, T) if continu
            else (ST > bar / 100 * S0).astype(np.float64))
    p_touche = 1.0 - p_no.mean()
    p_perte = ((1.0 - p_no) * (ST < strike / 100 * S0)).mean()

    m = st.columns(4)
    m[0].metric("Coupon equitable", f"{cpn * 100:,.2f}% p.a.")
    m[1].metric("Vol utilisee", f"{vol_eff * 100:,.1f}%",
                delta=f"{(vol_eff - vol_atm) * 100:+,.1f} pts vs ATM")
    m[2].metric("Proba de toucher", f"{p_touche * 100:,.1f}%")
    m[3].metric("Proba de perte en capital", f"{p_perte * 100:,.1f}%")

    d, vg = grecques_brc(T, r, q, vol_eff, bar / 100, strike / 100, continu, cpn,
                         nominal, min(n_paths, 20000))
    g = st.columns(2)
    g[0].metric("Delta (pour +1% de spot)", f"{d:+,.1f} CHF")
    g[1].metric("Vega (pour +1 pt de vol)", f"{vg:+,.1f} CHF")

    # tirage d'un indicateur de franchissement pour l'affichage uniquement
    rng = np.random.default_rng(3)
    touche = rng.random(len(p_no)) > p_no

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ech = rng.choice(len(ST), min(2500, len(ST)), replace=False)
    pay = np.where(touche & (ST < strike / 100 * S0),
                   nominal * ST / (strike / 100 * S0), nominal) + cpn * nominal * T
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
    tt = np.linspace(0, T, ch.shape[1])
    for i in range(45):
        ax.plot(tt, ch[i], lw=0.7, alpha=0.65, color=ROUGE if touche[i] else VERT)
    ax.axhline(bar / 100 * S0, color=GRIS, ls=":", lw=1.6)
    ax.axhline(S0, color="black", lw=0.8)
    ax.set_xlabel("Temps (annees)")
    ax.set_ylabel("Sous-jacent")
    ax.set_title("Le chemin compte, pas seulement l'arrivee")
    ax.grid(alpha=0.28)
    st.pyplot(fig)
    plt.close(fig)

    if st.checkbox("Afficher l'analyse de sensibilite a la volatilite"):
        st.table(table_sensi_brc(T, r, q, skew, marge, bar / 100, strike / 100,
                                 continu, nominal, n_paths))

    st.info(
        "**Le coupon a deux sources** : les taux (la banque garde le cash) et la prime du "
        "put que le client vend. Coupon eleve = sinistre probable. Jamais bonne affaire.\n\n"
        "**Le delta** dit combien d'actions le trader doit detenir pour etre couvert. Il "
        "bouge tous les jours, et pres de la barriere il bascule violemment : c'est le "
        "gamma qui explose. D'ou le prix degrade sur les barrieres proches du spot, et le "
        "refus pur et simple sur les sous-jacents illiquides.\n\n"
        "**Ta fenetre commerciale se lit sur la vol implicite**, pas sur la direction du "
        "marche.")


# =====================================================================
# 4 - AUTOCALL
# =====================================================================

elif produit.startswith("4"):
    st.title("Autocall Phoenix")
    st.caption("Le produit phare des desks. Coupon conditionnel, rappel anticipe, "
               "barriere de capital a l'echeance.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite maximale (annees)", 1.0, 6.0, 3.0, 0.5,
                  help="Duree MAXIMALE : le produit est presque toujours rappele bien "
                       "avant. Regarde la duree de vie moyenne calculee en dessous.")
    freq = c1.selectbox("Frequence d'observation",
                        ["Trimestrielle", "Semestrielle", "Annuelle"],
                        help="Les dates ou on regarde le sous-jacent pour decider du "
                             "rappel et du coupon. Trimestriel = plus d'occasions "
                             "d'etre rappele tot.")
    trigger = c2.slider("Trigger de rappel (%)", 80, 110, 100, 5,
                        help="Si le sous-jacent est au-dessus a une observation, le "
                             "produit s'arrete et le client est rembourse. Trigger bas "
                             "= rappel frequent = duree courte = coupon plus faible.")
    bar_cp = c2.slider("Barriere de coupon (%)", 50, 100, 70, 5,
                       help="Le coupon n'est paye QUE si le sous-jacent est au-dessus a "
                            "la date d'observation. C'est ce qui distingue un Phoenix "
                            "d'un BRC : ici le coupon est conditionnel.")
    bar_cap = c3.slider("Barriere de capital (%)", 40, 90, 60, 5,
                        help="Observee uniquement A L'ECHEANCE. Si le sous-jacent finit "
                             "en dessous, le client encaisse toute la baisse. C'est le "
                             "vrai risque du produit et le moteur principal du coupon.")
    memoire = c3.checkbox("Effet memoire", value=True,
                          help="Les coupons manques sont rattrapes au premier paiement "
                               "suivant. Tres bon argument de vente.")

    par_an = {"Trimestrielle": 4, "Semestrielle": 2, "Annuelle": 1}[freq]
    n_obs = max(int(T * par_an), 1)

    vol_eff = vol_au_strike(bar_cap / 100, vol_atm, skew)
    ch = simule(S0, T, r, q, vol_eff, n_paths)
    cpn = coupon_autocall(ch, T, r, nominal, n_obs, trigger / 100, bar_cp / 100,
                          bar_cap / 100, memoire, marge)
    duree, p_rappel, p_perte = stats_autocall(ch, T, n_obs, trigger / 100, bar_cap / 100)

    m = st.columns(4)
    m[0].metric("Coupon par observation", f"{cpn * 100:,.2f}%")
    m[1].metric("Equivalent annuel", f"{cpn * par_an * 100:,.2f}% p.a.")
    m[2].metric("Duree de vie moyenne", f"{duree:,.2f} ans")
    m[3].metric("Proba de perte en capital", f"{p_perte * 100:,.1f}%")

    st.subheader("Trajectoires et calendrier de sortie")
    indices = idx_obs(ch.shape[1] - 1, n_obs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    tt = np.linspace(0, T, ch.shape[1])
    for i in range(50):
        chem = ch[i]
        rappel_a = next((j for j in indices[:-1] if chem[j] >= trigger / 100 * S0), None)
        if rappel_a is not None:
            ax.plot(tt[:rappel_a + 1], chem[:rappel_a + 1], lw=0.8, alpha=0.7, color=VERT)
            ax.scatter([tt[rappel_a]], [chem[rappel_a]], s=20, color=VERT, zorder=5)
        else:
            ax.plot(tt, chem, lw=0.8, alpha=0.7,
                    color=ROUGE if chem[-1] < bar_cap / 100 * S0 else GRIS)
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
        vivant &= ~rappel
    probas.append(vivant.mean())
    ax.bar([f"Obs {k + 1}" for k in range(len(probas) - 1)] + ["Echeance"],
           np.array(probas) * 100, color=[VERT] * (len(probas) - 1) + [GRIS])
    ax.set_ylabel("Probabilite (%)")
    ax.set_title("Quand le produit se termine")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(alpha=0.28, axis="y")
    st.pyplot(fig)
    plt.close(fig)

    if st.checkbox("Afficher l'analyse de sensibilite a la barriere de capital"):
        st.table(table_sensi_autocall(T, r, q, vol_atm, skew, marge, n_obs,
                                      trigger / 100, bar_cp / 100, memoire,
                                      nominal, n_paths, par_an))

    st.info(
        "**Pourquoi ce produit domine le marche** : le client touche des coupons meme "
        "quand le sous-jacent baisse un peu, et recupere son cash tot si ca monte. "
        f"Duree de vie moyenne ici : {duree:,.2f} ans sur {T:,.1f} ans possibles.\n\n"
        "**Le piege commercial** : le rappel anticipe coupe les coupons futurs. Un client "
        "qui compare le coupon affiche a un rendement obligataire se trompe : il ne le "
        "touchera peut-etre qu'une ou deux fois.\n\n"
        "**Ce qu'il manque encore ici** : la plupart des autocalls vendus sont sur panier "
        "worst-of, ou le plus mauvais des trois sous-jacents decide de tout. La "
        "correlation devient alors un parametre de pricing central.")


# =====================================================================
# 5 - BONUS CERTIFICATE
# =====================================================================

elif produit.startswith("5"):
    st.title("Bonus Certificate")
    st.caption("Un niveau bonus garanti si la barriere tient, et la hausse reste illimitee.")

    c1, c2 = st.columns(2)
    T = c1.slider("Maturite (annees)", 0.5, 4.0, 2.0, 0.25,
                  help="Plus long = plus de dividendes captures = bonus plus eleve "
                       "financable. Mais aussi plus de temps pour toucher la barriere.")
    bar = c2.slider("Barriere (% du spot)", 50, 90, 70, 5,
                    help="Observee en continu. Si elle casse, le bonus disparait "
                         "definitivement et le client se retrouve avec un simple tracker.")

    vol_eff = vol_au_strike(bar / 100, vol_atm, skew)
    ch = simule(S0, T, r, q, vol_eff, n_paths)
    ST = ch[:, -1].astype(np.float64)
    p_no = proba_non_touche(ch, bar / 100 * S0, vol_eff, T)

    def pv_bonus(niveau):
        haut = np.maximum(nominal * ST / S0, nominal * niveau)
        bas = nominal * ST / S0
        return math.exp(-r * T) * (p_no * haut + (1 - p_no) * bas).mean()

    lo, hi, cible = 1.0, 2.5, nominal * (1.0 - marge * T)
    for _ in range(40):
        mid = (lo + hi) / 2
        if pv_bonus(mid) > cible:
            hi = mid
        else:
            lo = mid
    bonus = lo

    m = st.columns(3)
    m[0].metric("Niveau bonus offert", f"{bonus * 100:,.1f}%")
    m[1].metric("Rendement si marche lateral", f"{(bonus - 1) * 100:,.1f}%")
    m[2].metric("Proba de toucher la barriere", f"{(1 - p_no.mean()) * 100:,.1f}%")

    spots = np.linspace(30, 200, 500)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spots, np.maximum(nominal * spots / S0, nominal * bonus), lw=2.6,
            color=VERT, label="Barriere intacte")
    ax.plot(spots, nominal * spots / S0, lw=2.2, color=ROUGE, label="Barriere touchee")
    ax.axhline(nominal * bonus, color=GRIS, lw=0.8)
    ax.axvline(bar / 100 * S0, color=GRIS, ls=":", lw=1.6)
    cadre(ax)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "**Decomposition** : detention du sous-jacent sans dividendes, plus achat d'un put "
        "down-and-out de strike egal au niveau bonus. Les dividendes abandonnes financent "
        "le put.\n\n"
        "**Ce qui le distingue du BRC** : pas de plafond a la hausse. Si le sous-jacent "
        "explose, le client suit. Il paie ca par un bonus plus modeste qu'un coupon.\n\n"
        "**A ne jamais laisser passer** : la protection est conditionnelle et disparait "
        "entierement si la barriere casse. Ce n'est pas une garantie.")


# =====================================================================
# 6 - TWIN-WIN
# =====================================================================

else:
    st.title("Twin-Win")
    st.caption("Gagne a la hausse ET a la baisse moderee, tant que la barriere tient.")

    c1, c2 = st.columns(2)
    T = c1.slider("Maturite (annees)", 0.5, 4.0, 2.0, 0.25,
                  help="Le budget vient des dividendes abandonnes : plus la maturite est "
                       "longue, plus la participation a la baisse est elevee.")
    bar = c2.slider("Barriere (% du spot)", 50, 90, 65, 5,
                    help="Sous ce niveau, tout le mecanisme de gain a la baisse disparait "
                         "d'un coup. La discontinuite est brutale.")

    vol_eff = vol_au_strike(bar / 100, vol_atm, skew)
    ch = simule(S0, T, r, q, vol_eff, n_paths)
    ST = ch[:, -1].astype(np.float64)
    p_no = proba_non_touche(ch, bar / 100 * S0, vol_eff, T)
    perf_st = ST / S0 - 1

    def pv_twin(pb):
        haut = np.where(perf_st >= 0, nominal * (1 + perf_st),
                        nominal * (1 + pb * (-perf_st)))
        bas = nominal * ST / S0
        return math.exp(-r * T) * (p_no * haut + (1 - p_no) * bas).mean()

    lo, hi, cible = 0.0, 3.0, nominal * (1.0 - marge * T)
    for _ in range(40):
        mid = (lo + hi) / 2
        if pv_twin(mid) > cible:
            hi = mid
        else:
            lo = mid
    part_bas = lo

    m = st.columns(3)
    m[0].metric("Participation a la baisse", f"{part_bas * 100:,.0f}%")
    m[1].metric("Participation a la hausse", "100%")
    m[2].metric("Proba de toucher", f"{(1 - p_no.mean()) * 100:,.1f}%")

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
    plt.close(fig)

    st.info(
        "**Le produit qui impressionne les clients, et qu'il faut savoir demonter.** Une "
        "baisse de 20% rapporte de l'argent, tant que la barriere n'a pas casse.\n\n"
        "**Decomposition** : tracker plus achat d'un put down-and-out. Le put transforme "
        "la baisse en gain et s'evapore si la barriere casse.\n\n"
        "**Le point a ne jamais laisser passer** : entre la barriere et le niveau initial, "
        "le client gagne. Sous la barriere, il perd d'un coup toute cette zone de gain. "
        "La discontinuite est brutale : le profil le plus dangereux a mal expliquer.")


# =====================================================================
# PIED DE PAGE
# =====================================================================

st.markdown("---")
with st.expander("Limites du modele - a lire avant de montrer cette appli"):
    st.markdown("""
**1. Volatilite constante par produit.**
On simule en Black-Scholes avec une vol unique, choisie au niveau de strike pertinent.
Approximation praticien. Un vrai desk interpole dans une surface de vol calibree au
marche, voire utilise un modele a volatilite locale ou stochastique.

**2. Pas de sauts.**
Le mouvement brownien geometrique ne genere pas de gap de -20% en une seance. Or c'est
exactement ce qui fait mal sur un produit a barriere. Les pertes en queue de distribution
sont sous-estimees.

**3. Pas de risque emetteur.**
Un produit structure est une dette de la banque. En 2008, les detenteurs de produits
Lehman ont tout perdu alors meme que leurs barrieres tenaient.

**4. Mono-sous-jacent.**
La majorite des autocalls vendus sont sur panier worst-of. La correlation devient alors
un parametre de pricing central.

**5. Pas de frais de transaction, pas de bid-ask, pas de cout de hedge.**
La marge est modelisee comme un prelevement forfaitaire.

**Ce que le modele fait bien en revanche** : les barrieres continues sont traitees par
pont brownien, ce qui evite le biais classique des simulations discretes qui
sous-estiment systematiquement la probabilite de franchissement.
""")

st.caption("Tom Uzan - EDHEC BBA Finance - outil pedagogique, aucune valeur d'offre.")
