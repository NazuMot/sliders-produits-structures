"""
=======================================================================
SLIDERS PRODUITS STRUCTURES  -  v4
Tom Uzan - EDHEC BBA Finance
=======================================================================

Interface. Le pricing est dans moteur.py.

Lancer avec :  streamlit run app.py
=======================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import moteur as M
from moteur import S0

st.set_page_config(page_title="Sliders Produits Structures", layout="wide")

ROUGE, VERT, BLEU, GRIS, ORANGE = "#C8102E", "#2E7D32", "#1565C0", "#6E6E6E", "#E07B00"
MAX_CACHE = 6


# =====================================================================
# ENVELOPPES MISES EN CACHE
# =====================================================================

@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_bs(spot, T, r, q, sigma, n, seed=42):
    return M.simule_bs(spot, T, r, q, sigma, n, seed=seed)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_bates(spot, T, r, q, n, v0, theta, kappa, xi, rho, lam, mu_j, sig_j, seed=42):
    return M.simule_bates(spot, T, r, q, n, v0, theta, kappa, xi, rho,
                          lam, mu_j, sig_j, seed=seed)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_panier(n_actifs, corr, spot, T, r, q, sigma, n, seed=42):
    return M.simule_panier(n_actifs, corr, spot, T, r, q, sigma, n, seed=seed)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_defaut(n, T, spread_bps, recovery, seed=123):
    return M.tire_defaut(n, T, spread_bps, recovery, seed)


# =====================================================================
# BARRE LATERALE
# =====================================================================

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
    help="Carburant du budget option. Pour garantir 1 000 CHF dans 5 ans a 3%, la banque "
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
nominal = float(st.sidebar.select_slider("Nominal (CHF)",
                                         [1000, 10000, 100000, 1000000], 1000))
n_paths = st.sidebar.select_slider("Trajectoires Monte Carlo",
                                   [5000, 10000, 20000, 40000], 10000)

st.sidebar.markdown("---")
produit = st.sidebar.radio(
    "Page",
    ["Accueil", "Comparateur", "Laboratoire modele", "Cout de couverture", "Glossaire",
     "1 - Tracker Certificate", "2 - Capital Protection",
     "3 - Barrier Reverse Convertible", "4 - Autocall Phoenix",
     "5 - Bonus Certificate", "6 - Twin-Win"])

st.sidebar.markdown("---")
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


def cadre(ax):
    ax.axhline(nominal, color=GRIS, lw=0.7)
    ax.axvline(S0, color=GRIS, lw=0.7)
    ax.set_xlabel("Sous-jacent a l'echeance")
    ax.set_ylabel("Remboursement (CHF)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.28)


def bandeau_modele():
    if BATES:
        st.caption(f"Pricing sous Bates — vol de depart {vol_atm*100:.0f}%, "
                   f"correlation spot/vol {rho:+.2f}, {lam:.1f} saut(s) par an."
                   + (f" Spread emetteur {spread} bp." if spread else ""))
    else:
        st.caption(f"Pricing sous Black-Scholes — vol ATM {vol_atm*100:.0f}%, "
                   f"skew {skew:.1f} pt/10%."
                   + (f" Spread emetteur {spread} bp." if spread else ""))


# =====================================================================
# ACCUEIL
# =====================================================================

if produit == "Accueil":
    st.title("Sliders Produits Structures")
    st.markdown(
        "**Tom Uzan** — EDHEC BBA, Finance Major — objectif Sales en produits "
        "structures sur desk institutionnel.")

    st.markdown("---")
    st.markdown("""
### A quoi sert cet outil

Comprendre les produits structures en manipulant leurs parametres plutot qu'en lisant
des formules. **Rien n'est choisi a la main** : la participation d'un capital garanti et
le coupon d'un BRC ou d'un autocall sont calcules a partir du budget disponible, comme
le ferait un structureur. Quand tu bouges les taux, la volatilite ou la barriere, tu vois
ce que le desk peut reellement offrir ce jour-la.

### Par ou commencer

| Page | Ce qu'on y voit |
|---|---|
| **Comparateur** | Les quatre structures principales cote a cote, sous la meme vue de marche |
| **1 a 6** | Un produit par page, avec son profil de remboursement et ses trajectoires |
| **Laboratoire modele** | Ce que change le choix du modele de pricing : pres de 3 points de coupon |
| **Cout de couverture** | Pourquoi la marge n'est pas un simple prelevement |
| **Glossaire** | Tout le vocabulaire, avec ce que chaque terme implique pour un Sales |

Tous les parametres de marche sont dans la barre laterale a gauche et alimentent
l'ensemble des produits. Chaque curseur a une infobulle qui explique son effet.

### Ce qu'il y a sous le capot

- **Deux modeles** : Black-Scholes avec skew impose, et Bates (volatilite stochastique
  de Heston + sauts de Merton), qui genere le skew au lieu de le postuler.
- **Barrieres continues traitees par pont brownien**, ce qui evite le biais classique
  des simulations discretes.
- **Risque emetteur** : intensite de defaut deduite du spread de credit, avec taux de
  recouvrement.
- **Paniers worst-of** avec correlation, sur la page Autocall.
- **Couverture en delta** simulee avec rebalancement discret et frais de transaction.

### Ce que ce n'est pas

Les parametres sont regles au curseur, **pas calibres** sur des prix d'options cotees.
Un desk recalibre son modele chaque matin. Les niveaux affiches sont donc des ordres de
grandeur pedagogiques, pas des prix de marche. La liste complete des limites est en bas
de chaque page.
""")
    st.info("Outil pedagogique. Aucune valeur d'offre, aucune recommandation "
            "d'investissement.")


# =====================================================================
# COMPARATEUR
# =====================================================================

elif produit == "Comparateur":
    st.title("Comparateur")
    bandeau_modele()

    T = st.slider("Maturite (annees)", 1.0, 5.0, 2.0, 0.5)

    ch_brc, s_brc = trajectoires(T, 0.70)
    ch_ac, s_ac = trajectoires(T, 0.65, seed=43)
    tau_brc = defauts(len(ch_brc), T)
    tau_ac = defauts(len(ch_ac), T)

    zc = math.exp(-r * T)
    budget = 1.0 - zc - marge * T
    vol_call = vol_atm if BATES else M.vol_au_strike(1.0, vol_atm, skew)
    call_u = M.bs_call(S0, S0, T, r, vol_call, q) / S0
    part = max(budget / call_u, 0.0) if call_u > 0 else 0.0

    cpn_brc = M.coupon_brc(ch_brc, T, r, nominal, 0.70, 1.0, True, marge, s_brc,
                           tau_brc, recovery)
    n_obs = max(int(T * 4), 1)
    cpn_ac = M.coupon_autocall(ch_ac, T, r, nominal, n_obs, 1.0, 0.70, 0.65, True,
                               marge, tau_ac, recovery)
    duree, _, p_perte = M.stats_autocall(ch_ac, T, n_obs, 1.0, 0.65)

    st.subheader("Ce que le desk peut offrir aujourd'hui")
    st.table([
        {"Produit": "Tracker", "Offre": "Participation 100%", "Protection": "aucune",
         "Plafond hausse": "aucun", "Vue client": "haussier franc"},
        {"Produit": "Capital Protection", "Offre": f"Participation {part*100:,.0f}%",
         "Protection": "capital garanti", "Plafond hausse": "participation reduite",
         "Vue client": "haussier prudent"},
        {"Produit": "Barrier Reverse Conv.", "Offre": f"Coupon {cpn_brc*100:,.2f}% p.a.",
         "Protection": "barriere 70%", "Plafond hausse": f"{cpn_brc*100:,.2f}% p.a.",
         "Vue client": "neutre"},
        {"Produit": "Autocall Phoenix", "Offre": f"Coupon {cpn_ac*100:,.2f}% / trimestre",
         "Protection": "barriere 65% a l'echeance", "Plafond hausse": "coupons cumules",
         "Vue client": "neutre a legerement haussier"},
    ])

    c = st.columns(3)
    c[0].metric("Proba de hausse du sous-jacent",
                f"{(ch_ac[:, -1] > S0).mean()*100:,.1f}%",
                help="Sous mesure risque-neutre. Ce n'est pas une prevision.")
    c[1].metric("Duree de vie moyenne autocall", f"{duree:,.2f} ans")
    c[2].metric("Proba de perte autocall", f"{p_perte*100:,.1f}%")

    spots = np.linspace(30, 190, 400)
    perf = spots / S0 - 1
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(spots, nominal * (1 + perf), color="black", ls="--", lw=1.2, label="Direct")
    ax.plot(spots, nominal * (1 + part * np.maximum(perf, 0)), lw=2.2, color=BLEU,
            label=f"Capital Protection ({part*100:,.0f}%)")
    ax.plot(spots, np.where(spots < 70, nominal * spots / S0, nominal)
            + cpn_brc * nominal * T, lw=2.2, color=ROUGE,
            label="BRC (cas barriere touchee)")
    cadre(ax)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "**La regle qui structure tout le metier** : protection, participation, coupon. "
        "Jamais trois sur trois. Le budget est fixe : il vient des taux, des dividendes, "
        "du spread emetteur et de ce que le client accepte de vendre comme optionalite.\n\n"
        "**Ta question au client** : quelle est ta vue, et qu'es-tu pret a abandonner "
        "pour l'exprimer ?")


# =====================================================================
# LABORATOIRE MODELE
# =====================================================================

elif produit == "Laboratoire modele":
    st.title("Laboratoire modele")
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

    st.subheader("1. Le smile, genere et non postule")
    ks = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
    vols = M.vol_implicite_bates(ch_bt, T, r, q, ks)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot([k * 100 for k in ks], [v * 100 for v in vols], "o-", lw=2.2, color=ROUGE,
            label="Vol implicite generee par Bates")
    ax.axhline(vol_atm * 100, color=GRIS, ls="--", lw=1.2,
               label="Vol unique de Black-Scholes")
    ax.set_xlabel("Strike (% du spot)")
    ax.set_ylabel("Volatilite implicite (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.28)
    st.pyplot(fig)
    plt.close(fig)

    pente = (vols[0] - vols[3]) * 100
    st.markdown(
        f"Pente 70 % / 100 % : **{pente:+.1f} points de vol**. "
        "Black-Scholes donne une droite plate — il price le put a 70 % au meme prix "
        "relatif que l'option a la monnaie, ce que le marche ne fait jamais.")

    st.subheader("2. Les queues de distribution")
    st_bs = ch_bs[:, -1].astype(float)
    st_bt = ch_bt[:, -1].astype(float)
    c = st.columns(4)
    c[0].metric("P(S < 60) Black-Scholes", f"{(st_bs < 60).mean()*100:,.2f}%")
    c[1].metric("P(S < 60) Bates", f"{(st_bt < 60).mean()*100:,.2f}%")
    c[2].metric("1er centile BS", f"{np.percentile(st_bs, 1):,.1f}")
    c[3].metric("1er centile Bates", f"{np.percentile(st_bt, 1):,.1f}")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    bins = np.linspace(20, 200, 120)
    ax.hist(st_bs, bins=bins, alpha=0.55, color=GRIS, label="Black-Scholes", density=True)
    ax.hist(st_bt, bins=bins, alpha=0.55, color=ROUGE, label="Bates", density=True)
    ax.axvline(bar, color="black", ls=":", lw=1.6)
    ax.text(bar, ax.get_ylim()[1] * 0.9, " barriere", fontsize=8)
    ax.set_xlabel("Sous-jacent a l'echeance")
    ax.set_ylabel("Densite")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.28)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("3. Ce que ca change sur le prix")
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

    st.subheader("4. Effet du spread emetteur")
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


# =====================================================================
# COUT DE COUVERTURE
# =====================================================================

elif produit == "Cout de couverture":
    st.title("Cout de couverture")
    st.caption("Pourquoi la marge n'est pas un simple prelevement : le trader doit "
               "couvrir, et couvrir coute de l'argent.")

    st.markdown(
        "On prend un **Reverse Convertible sans barriere** : le client vend un put "
        "a la monnaie, la banque l'achete. Le trader delta-couvre cette position "
        "jusqu'a l'echeance. En theorie de Black-Scholes, avec une couverture continue "
        "et sans frais, le resultat est exactement nul. On enleve ces deux hypotheses.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 0.25, 2.0, 1.0, 0.25)
    freq_h = c2.select_slider("Rebalancement",
                              ["Mensuel", "Hebdomadaire", "2x par semaine", "Quotidien"],
                              "Hebdomadaire",
                              help="Plus on couvre souvent, plus l'erreur de couverture "
                                   "est faible, mais plus on paie de frais.")
    cout_bp = c3.slider("Cout de transaction (bp par transaction)", 0, 50, 10, 1,
                        help="Fourchette achat-vente plus impact de marche, en points "
                             "de base du montant echange.")

    par_an = {"Mensuel": 12, "Hebdomadaire": 52, "2x par semaine": 104, "Quotidien": 252}[freq_h]
    n_reb = max(int(round(par_an * T)), 4)

    vol_reelle = st.slider(
        "Volatilite realisee (%)", 5.0, 60.0, vol_atm * 100, 1.0,
        help="Ce qui se passe VRAIMENT. La banque a vendu le produit sur la base de la "
             "vol implicite. L'ecart entre les deux est son resultat principal.") / 100

    @st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
    def couverture(T, n_reb, vol_impl, vol_reel, cout_bp, r, q, nominal, n_sim=4000):
        """
        Simule la couverture en delta d'un put vendu au client.

        A chaque date : on recalcule le delta Black-Scholes avec la vol
        IMPLICITE (celle utilisee pour vendre), on ajuste la position en
        actions, on paie les frais, on capitalise le cash.

        Trois sources de resultat pour le trader :
          1. vol implicite > vol realisee : il a vendu cher, il gagne
          2. rebalancement discret : il rate une partie des mouvements,
             ca cree une erreur qui ne se compense pas
          3. les frais de transaction : cout pur, proportionnel au gamma
        """
        dt = T / n_reb
        rng = np.random.default_rng(7)
        z = rng.standard_normal((n_sim // 2, n_reb))
        z = np.vstack([z, -z])
        S = np.empty((z.shape[0], n_reb + 1))
        S[:, 0] = S0
        for t in range(n_reb):
            S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * vol_reel ** 2) * dt
                                           + vol_reel * math.sqrt(dt) * z[:, t])

        n_parts = nominal / S0

        def delta_put(s, tau):
            if tau <= 1e-9:
                return np.where(s < S0, -1.0, 0.0)
            d1 = (np.log(s / S0) + (r - q + 0.5 * vol_impl ** 2) * tau) / (vol_impl * math.sqrt(tau))
            return -math.exp(-q * tau) * (1.0 - M.norm_cdf_vec(d1))

        # la banque est ACHETEUSE du put -> delta du put achete = delta_put
        # pour etre neutre elle detient -delta actions
        prime = M.bs_call(S0, S0, T, r, vol_impl, q) - S0 * math.exp(-q * T) + S0 * math.exp(-r * T)
        cash = np.full(z.shape[0], -prime * n_parts)     # elle a paye la prime
        pos = np.zeros(z.shape[0])
        frais_tot = np.zeros(z.shape[0])

        for t in range(n_reb + 1):
            tau = T - t * dt
            cible = -delta_put(S[:, t], tau) * n_parts
            if t == n_reb:
                cible = np.zeros_like(cible)
            d = cible - pos
            frais = np.abs(d) * S[:, t] * cout_bp / 10000.0
            frais_tot += frais
            cash -= d * S[:, t] + frais
            pos = cible
            if t < n_reb:
                cash *= math.exp(r * dt)
                cash += pos * S[:, t] * (math.exp(q * dt) - 1.0)   # dividendes percus

        payoff_put = np.maximum(S0 - S[:, -1], 0.0) * n_parts      # elle encaisse le put
        pnl = cash + payoff_put
        return pnl, frais_tot, S

    pnl, frais, S = couverture(T, n_reb, vol_atm, vol_reelle, cout_bp, r, q, nominal)

    m = st.columns(4)
    m[0].metric("Resultat moyen du trader", f"{pnl.mean():+,.1f} CHF")
    m[1].metric("Ecart-type du resultat", f"{pnl.std():,.1f} CHF")
    m[2].metric("Frais de transaction moyens", f"{frais.mean():,.1f} CHF")
    m[3].metric("Pire cas sur 5%", f"{np.percentile(pnl, 5):+,.1f} CHF")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    ax.hist(pnl, bins=70, color=ROUGE, alpha=0.75)
    ax.axvline(0, color="black", lw=1.4)
    ax.axvline(pnl.mean(), color=VERT, lw=1.8, ls="--", label="Moyenne")
    ax.set_xlabel("Resultat de couverture (CHF)")
    ax.set_ylabel("Frequence")
    ax.set_title("Le resultat du trader n'est pas certain")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.28)

    ax = axes[1]
    freqs = [12, 26, 52, 104, 252]
    moy, ect = [], []
    for f in freqs:
        p, _, _ = couverture(T, max(int(f * T), 4), vol_atm, vol_reelle, cout_bp,
                             r, q, nominal, 3000)
        moy.append(p.mean())
        ect.append(p.std())
    ax.plot(freqs, moy, "o-", color=VERT, lw=2, label="Resultat moyen")
    ax.plot(freqs, ect, "s-", color=ORANGE, lw=2, label="Ecart-type (risque)")
    ax.set_xscale("log")
    ax.set_xlabel("Rebalancements par an")
    ax.set_ylabel("CHF")
    ax.set_title("L'arbitrage du trader")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.28)
    st.pyplot(fig)
    plt.close(fig)

    ecart = (vol_atm - vol_reelle) * 100
    st.info(
        f"**Lecture du resultat.** La banque a achete le put a {vol_atm*100:,.0f}% de vol "
        f"implicite, et le marche a realise {vol_reelle*100:,.0f}%. "
        + (f"Elle a donc paye trop cher de {-ecart:,.0f} points : elle perd sur cette jambe."
           if ecart < 0 else
           f"Elle a paye {ecart:,.0f} points de moins que ce que le marche a realise : "
           "elle perd, car elle est acheteuse de vol et la vol realisee est basse.")
        + "\n\n**L'arbitrage du graphe de droite** : couvrir plus souvent reduit le risque "
          "(courbe orange qui descend) mais coute plus cher en frais (courbe verte qui "
          "descend aussi). Il existe une frequence optimale, et elle depend du cout de "
          "transaction. C'est un vrai sujet de desk.\n\n"
        "**Ce que ca t'apprend en tant que Sales** : quand le desk degrade son prix sur un "
        "sous-jacent illiquide ou une barriere proche du spot, ce n'est pas de la mauvaise "
        "volonte. C'est ce cout-la, et il est mesurable.")

    st.warning(
        "**Limite assumee de cette page** : on couvre ici un put vanille, pas un put a "
        "barriere. Sur un vrai BRC, le gamma explose autour de la barriere et le cout de "
        "couverture est nettement superieur a ce qui est affiche ici. L'ordre de grandeur "
        "et le mecanisme sont justes ; le niveau est un plancher.")


# =====================================================================
# GLOSSAIRE
# =====================================================================

elif produit == "Glossaire":
    st.title("Glossaire")
    o = st.tabs(["Parametres de marche", "Mecaniques produit", "Les grecques",
                 "Volatilite", "Modeles", "Jargon de desk"])

    with o[0]:
        st.markdown("""
### Taux sans risque (r)
Carburant du budget option. Pour garantir 1 000 CHF dans 5 ans a 3%, la banque ne met
que 861 aujourd'hui. Les 139 restants achetent des options. A 0,5%, elle doit mettre 975
et il ne reste que 25.

**Consequence commerciale** : taux hauts = capital garanti vendable. Taux bas = le marche
bascule vers BRC et autocalls.

### Taux de dividende (q)
Le detenteur d'un produit structure ne le touche pas. Sur un tracker 3 ans avec 2,5% de
dividende, le client abandonne environ 7,8% de rendement sans le voir.

### Spread de credit emetteur
Ce que la banque paie au-dessus du taux sans risque pour se financer. **Il augmente le
coupon** : la banque se finance moins cher via le produit structure que sur le marche
obligataire, et reverse l'economie au client. Un coupon superieur a la concurrence peut
donc simplement signifier un emetteur moins bien note.

### Taux de recouvrement
Ce que le client recupere en cas de defaut. 40% est la convention senior non securise.
Les porteurs de produits Lehman ont recupere environ 9 cents par dollar.

### Marge banque
Prelevee sur le budget option, invisible dans le payoff. Divulguee reglementairement
(PRIIP, guidelines SSPA sur les couts).

### Monte Carlo
Pricing par simulation : des milliers de trajectoires, le payoff dans chacune, la moyenne
actualisee. Indispensable des qu'un produit depend du chemin.

### Univers risque-neutre
Le sous-jacent monte en moyenne au taux (r - q), pas selon une prevision. **Ce n'est pas
un pari directionnel** : c'est la seule hypothese qui interdit l'arbitrage.

### Pont brownien
Correction appliquee aux barrieres. Une simulation a pas discrets rate les franchissements
entre deux observations : elle sous-estime la probabilite de toucher. Le pont brownien la
calcule analytiquement entre deux points. Plus rapide **et** plus juste qu'une simulation
quotidienne.
""")

    with o[1]:
        st.markdown("""
### Strike
Niveau de reference qui decide du remboursement. Standard : 100% du spot initial.

### Barriere
- **Toucher une barriere ne fait pas perdre.** Ca arme le risque. C'est le strike qui
  decide a la fin.
- **Observation continue (americaine)** : la barriere compte a tout moment, meme une
  meche intraday. Standard marche.
- **Observation europeenne** : seul le niveau a l'echeance compte. Plus favorable au
  client, donc coupon plus faible.

### Knock-in / Knock-out
- **Knock-in** : l'option s'active au franchissement. Le put d'un BRC.
- **Knock-out** : l'option disparait au franchissement. Le put d'un Bonus ou d'un Twin-Win.

### Participation
Ratio de suivi. 70% : le sous-jacent fait +20%, le client touche +14%.
**Jamais choisie, toujours calculee** : budget option divise par prix du call.

### Coupon
**Ce n'est pas un interet.** C'est le prix d'une option que le client vend. Trois
sources : les taux, la prime du put vendu, et le spread emetteur.

Coupon eleve = sinistre probable, ou emetteur risque. Jamais bonne affaire.

### Coupon conditionnel vs inconditionnel
- **Inconditionnel** (BRC) : paye quoi qu'il arrive.
- **Conditionnel** (Phoenix) : paye seulement au-dessus de la barriere de coupon.

### Effet memoire
Les coupons manques sont rattrapes au premier paiement suivant.

### Trigger de rappel
Si le sous-jacent est au-dessus a une observation, le produit s'arrete.
**Le piege** : le rappel coupe tous les coupons futurs.

### Livraison physique
Si la barriere casse et que le sous-jacent finit sous le strike, le client recoit des
actions au lieu de son cash.

### Worst-of
Panier ou le plus mauvais des sous-jacents decide de tout. Le coupon affiche est
spectaculaire, et pour cause : il faut que TOUS tiennent, pas un seul.

**La correlation devient le parametre dominant** : correlation haute, les actifs bougent
ensemble et le worst-of ressemble a un actif unique, donc coupon faible. Correlation
basse, il y a presque toujours un trainard, donc coupon eleve et risque eleve.

### Risque emetteur
**Un produit structure est une dette de la banque**, pas une detention d'actions.
En 2008, les porteurs de produits Lehman ont tout perdu alors que leurs barrieres
tenaient. Premier point sur lequel un institutionnel te challengera.
""")

    with o[2]:
        st.markdown("""
| Grecque | Ce que c'est | Ce que ca change pour toi |
|---|---|---|
| **Delta** | Sensibilite au prix du sous-jacent | Le nombre d'actions que le trader detient pour etre couvert. Il l'ajuste tous les jours : delta-hedging. |
| **Gamma** | Vitesse a laquelle le delta change | Explose pres d'une barriere : volumes enormes a trader en peu de temps (pin risk). LA raison du prix degrade sur les barrieres proches du spot. |
| **Vega** | Sensibilite a la volatilite implicite | Ta grecque. Vol haute au pricing = coupon eleve. Ta fenetre commerciale. |
| **Theta** | Erosion du prix avec le temps | Le vendeur d'option gagne du theta chaque jour. Un client qui achete un BRC est structurellement long theta. |
| **Rho** | Sensibilite aux taux | Determinante sur le capital garanti, secondaire sur les produits a coupon. |

### Bump and revalue
On decale un parametre de peu, on reprice, on regarde l'ecart.

**Le detail qui compte** : reutiliser exactement les memes tirages aleatoires pour les
deux evaluations (common random numbers). Sinon le bruit Monte Carlo domine la
difference et la grecque est inexploitable.

### Cout de couverture
En theorie de Black-Scholes, une couverture continue et sans frais donne un resultat
exactement nul. Dans la vraie vie, deux fuites : le rebalancement est discret, et chaque
transaction coute. Vois la page Cout de couverture : l'arbitrage frequence / frais est un
vrai sujet de desk.
""")

    with o[3]:
        st.markdown("""
### Volatilite implicite
**Pas une prevision.** Le prix d'une option exprime dans une autre unite. Une option vaut
55 CHF, ou elle vaut 22% de vol : meme information.

### Skew
Les puts bas coutent plus cher **en vol** que les options a la monnaie. Typiquement sur
indice a 1 an : put 70% a 26%, ATM a 18%, call 110% a 16%.

**Deux causes** :
1. Les actions ne baissent pas comme elles montent. Un krach de -20% en une semaine
   existe ; une hausse de +20% en une semaine, non.
2. Les institutionnels achetent massivement des puts de protection, sans vendeur naturel.

**Ce que ca change** : le skew fait monter le coupon d'un BRC et ameliore la participation
d'un capital garanti. Il profite au client dans les deux cas, pour des raisons opposees.

### Terme structure
La vol varie avec la maturite. En regime normal, la vol longue est au-dessus de la courte.
En stress, ca s'inverse : la vol 1 mois explose alors que la vol 2 ans bouge peu.

*Pour toi* : en periode de stress, les BRC courts offrent des coupons exceptionnels.

### Surface de volatilite
Skew (par strike) + terme structure (par maturite) = une surface a deux dimensions.
L'ecran du trader, la matiere premiere du structureur.

### Vol implicite vs vol realisee
L'implicite est structurellement au-dessus de la realisee, de 2 a 4 points sur les
indices. Cet ecart s'appelle la **variance risk premium**.

**C'est la raison d'etre economique des BRC et des autocalls.** Le client capture cette
prime. Ni hasard ni anomalie : c'est la remuneration de celui qui accepte d'etre expose
aux krachs. Il gagne neuf annees sur dix, et perd beaucoup la dixieme.
""")

    with o[4]:
        st.markdown("""
### Black-Scholes
Une seule volatilite, constante, pour toutes les options d'un meme sous-jacent.

**Ce qu'il fait bien** : les payoffs, l'intuition, les ordres de grandeur.
**Ce qu'il rate** : le skew (il faut le plaquer a la main), les krachs, la dynamique de
la volatilite.

### Heston
La volatilite devient elle-meme un processus aleatoire, qui revient vers une moyenne.

    dv = kappa*(theta - v) dt + xi*sqrt(v) dW2

- **rho** : correlation entre le sous-jacent et sa volatilite. Negatif = la vol monte
  quand le marche baisse. **C'est ce qui cree le skew**, au lieu de le postuler.
- **xi** : vol de la vol. Donne de la convexite au smile.
- **kappa, theta** : retour a la moyenne. Cree la structure par terme.

### Merton (sauts)
On ajoute des sauts poissonniens au sous-jacent. Un gap de -15% en une seance devient
possible. Le brownien geometrique en est incapable, et c'est exactement ce qui fait mal
sur un produit a barriere.

### Bates
Heston + Merton. **Le standard de marche pour les exotiques actions.** C'est le modele
actif dans cette appli quand tu le selectionnes.

### Ce que les desks utilisent vraiment
Bates ou une variante, calibre chaque jour sur les prix d'options cotees. Pour certains
produits, un modele a volatilite locale (Dupire) qui reproduit exactement la surface
observee. Pour les worst-of, un modele multi-actifs avec correlation calibree.

Les parametres de cette appli sont regles a la main, pas calibres sur un marche reel.
C'est la difference honnete entre un outil pedagogique et un pricer de production.
""")

    with o[5]:
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

### Vocabulaire courant

| Terme | Definition |
|---|---|
| Sous-jacent | L'actif dont depend le produit : action, indice, panier. |
| Spot | Le prix actuel du sous-jacent. |
| Termsheet | Le document contractuel. Ce que le client signe. |
| Autocall | Tout produit avec rappel anticipe automatique. |
| Phoenix | Autocall a coupons conditionnels, generalement avec effet memoire. La structure la plus vendue. |
| Reverse convertible | Le client vend de l'optionalite contre un coupon. Le "reverse" : c'est l'emetteur qui a le droit de livrer les actions. |
| Call spread | Achat d'un call + vente d'un call plus haut. Permet de capper un capital garanti pour remonter la participation. |
| Delta-hedging | L'activite quotidienne du trader : rester neutre a la direction. |
| Pin risk | Risque de couverture quand le sous-jacent stagne autour d'une barriere. |
| Wrong-way risk | Le defaut de l'emetteur survient au pire moment, correle au marche. Ignore dans cette appli. |
| SSPA | Swiss Structured Products Association. Reference du marche suisse, premier marche mondial. |
| PRIIP / KID | Reglementation europeenne : document d'information standardise pour tout produit structure vendu au retail. |
""")


# =====================================================================
# 1 - TRACKER
# =====================================================================

elif produit.startswith("1"):
    st.title("Tracker Certificate")
    st.caption("Exposition lineaire. Produit d'acces, pas produit de rendement.")

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 1.0, 10.0, 3.0, 0.5)
    participation = c2.slider("Participation (%)", 50, 150, 100, 5,
                              help="100% = 1:1. Au-dessus, on parle d'outperformance "
                                   "certificate : la banque finance le levier en "
                                   "capturant les dividendes.")
    frais_pa = c3.slider("Frais annuels (%)", 0.0, 2.0, 0.6, 0.1)

    spots = np.linspace(30, 180, 400)
    perf = spots / S0 - 1
    payoff = nominal * (1 + participation / 100 * perf) * (1 - frais_pa / 100 * T)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spots, nominal * spots / S0, color="black", ls="--", lw=1.1, label="Direct")
    ax.plot(spots, payoff, lw=2.6, color=ROUGE, label="Tracker")
    cadre(ax)
    st.pyplot(fig)
    plt.close(fig)

    st.table([{"Sous-jacent": f"{p:+d}%",
               "Tracker": f"{nominal*(1+participation/100*p/100)*(1-frais_pa/100*T):,.0f}",
               "Direct": f"{nominal*(1+p/100):,.0f}",
               "Ecart": f"{nominal*(1+participation/100*p/100)*(1-frais_pa/100*T) - nominal*(1+p/100):+,.0f}"}
              for p in [-40, -20, 0, 20, 40]])

    st.info(
        f"**Le dividende est le nerf du produit.** Le sous-jacent verse {q*100:,.1f}% par "
        f"an que le client ne touche pas : sur {T:,.1f} ans cela represente environ "
        f"{(math.exp(q*T)-1)*100:,.1f}% de rendement abandonne.\n\n"
        "**Le risque a nommer en clientele** : le client est creancier de la banque. "
        "Difference majeure avec un ETF.")


# =====================================================================
# 2 - CAPITAL PROTECTION
# =====================================================================

elif produit.startswith("2"):
    st.title("Capital Protection Certificate")
    st.caption("La participation n'est pas un choix commercial. C'est un reste de budget.")
    bandeau_modele()

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 1.0, 10.0, 5.0, 0.5,
                  help="Parametre cle : plus c'est long, moins la banque doit mettre de "
                       "cote, donc plus le budget option est gros.")
    protection = c2.slider("Niveau de protection (%)", 80, 100, 100, 5,
                           help="90% = le client accepte de perdre 10% maximum. Chaque "
                                "point abandonne remonte la participation.")
    cap = c3.slider("Cap sur la hausse (%)", 110, 300, 300, 10,
                    help="Capper vend un call de strike haut, ce qui libere du budget.")

    # le spread emetteur reduit le cout de la garantie : la banque actualise
    # a son propre taux de financement
    r_fin = r + spread / 10000.0
    zc = protection / 100 * math.exp(-r_fin * T)
    budget = 1.0 - zc - marge * T
    v_call = vol_atm if BATES else M.vol_au_strike(1.0, vol_atm, skew)
    v_cap = vol_atm if BATES else M.vol_au_strike(cap / 100, vol_atm, skew)
    cout = max((M.bs_call(S0, S0, T, r, v_call, q)
                - M.bs_call(S0, cap / 100 * S0, T, r, v_cap, q)) / S0, 1e-9)
    part = max(budget / cout, 0.0)

    m = st.columns(4)
    m[0].metric("Cout de la garantie", f"{zc*nominal:,.0f}")
    m[1].metric("Budget option", f"{budget*nominal:,.0f}")
    m[2].metric("Cout du call spread", f"{cout*nominal:,.0f}")
    m[3].metric("Participation offerte", f"{part*100:,.0f}%")

    if part < 0.35:
        st.error("Participation sous 35% : invendable. Baisse la protection, allonge la "
                 "maturite, ou mets un cap.")
    elif part > 1.0:
        st.success("Participation au-dessus de 100% : configuration de taux favorable.")

    if spread > 0:
        st.caption(f"La garantie est actualisee au taux de financement de la banque "
                   f"({r*100:,.2f}% + {spread} bp), pas au taux sans risque. C'est ce qui "
                   "permet a un emetteur moins bien note d'offrir une meilleure "
                   "participation — au prix d'un risque de credit que le client porte.")

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
        z = protection / 100 * math.exp(-(t + spread / 10000.0) * T)
        b = 1 - z - marge * T
        c_ = max((M.bs_call(S0, S0, T, t, v_call, q)
                  - M.bs_call(S0, cap / 100 * S0, T, t, v_cap, q)) / S0, 1e-9)
        lignes.append({"Taux": f"{t*100:.0f}%", "Mis de cote": f"{z*nominal:,.0f}",
                       "Budget option": f"{b*nominal:,.0f}",
                       "Participation": f"{max(b/c_, 0)*100:,.0f}%"})
    st.table(lignes)

    st.info(
        "**Le mecanisme** : la banque doit rembourser le nominal a l'echeance. Elle place "
        "juste ce qu'il faut pour y arriver. Ce qui reste achete un call.\n\n"
        "**A savoir dire en entretien** : taux hauts = capital garanti vendable. Taux a "
        "zero = participation ridicule, le marche bascule vers BRC et autocalls. "
        "Mouvement observe entre 2015 et 2021, puis l'inverse depuis.")


# =====================================================================
# 3 - BRC
# =====================================================================

elif produit.startswith("3"):
    st.title("Barrier Reverse Convertible")
    st.caption("Le client ne recoit pas un interet. Il encaisse une prime d'assurance.")
    bandeau_modele()

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 0.25, 3.0, 1.0, 0.25)
    bar = c2.slider("Barriere (% du spot)", 50, 95, 70, 5,
                    help="Le niveau qui ARME le risque. Le toucher ne fait pas perdre : "
                         "c'est le strike qui decide a la fin.")
    strike = c3.slider("Strike (% du spot)", 80, 110, 100, 5)

    continu = st.radio("Observation de la barriere",
                       ["Continue (americaine)", "A l'echeance (europeenne)"],
                       horizontal=True).startswith("Continue")

    ch, sg = trajectoires(T, bar / 100)
    tau = defauts(len(ch), T)
    cpn = M.coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu, marge,
                       sg, tau, recovery)

    ST = ch[:, -1].astype(np.float64)
    p_no = (M.proba_non_touche(ch, bar / 100 * S0, sg, T) if continu
            else (ST > bar / 100 * S0).astype(np.float64))

    m = st.columns(4)
    m[0].metric("Coupon equitable", f"{cpn*100:,.2f}% p.a.")
    m[1].metric("Proba de toucher", f"{(1-p_no.mean())*100:,.1f}%")
    m[2].metric("Proba de perte en capital",
                f"{((1-p_no)*(ST < strike/100*S0)).mean()*100:,.1f}%")
    m[3].metric("Proba de defaut emetteur",
                f"{0.0 if tau is None else (tau <= T).mean()*100:,.1f}%")

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

    if st.checkbox("Decomposer le coupon"):
        base = M.coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu,
                            marge, sg, None, recovery)
        st.table([
            {"Composante": "Coupon hors risque emetteur", "Valeur": f"{base*100:,.2f}% p.a."},
            {"Composante": "Apport du spread emetteur", "Valeur": f"{(cpn-base)*100:+,.2f}% p.a."},
            {"Composante": "Coupon offert au client", "Valeur": f"{cpn*100:,.2f}% p.a."},
        ])
        st.caption("Le spread emetteur n'est pas un cadeau : c'est la remuneration d'un "
                   "risque de credit que le client porte sans toujours le savoir.")

    st.info(
        "**Le coupon a trois sources** : les taux, la prime du put que le client vend, et "
        "le spread de l'emetteur. Coupon eleve = sinistre probable, ou signature faible.\n\n"
        "**Pres de la barriere, le gamma explose** : le trader doit trader des volumes "
        "enormes en peu de temps. D'ou le prix degrade sur les barrieres proches du spot. "
        "La page Cout de couverture chiffre ce mecanisme.")


# =====================================================================
# 4 - AUTOCALL
# =====================================================================

elif produit.startswith("4"):
    st.title("Autocall Phoenix")
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
    ax.text(0.02, trigger / 100 * S0 + 2, "trigger", fontsize=7, color=VERT)
    ax.text(0.02, bar_cp / 100 * S0 + 2, "barriere coupon", fontsize=7, color=BLEU)
    ax.text(0.02, bar_cap / 100 * S0 - 6, "barriere capital", fontsize=7, color=ROUGE)
    ax.set_xlabel("Temps (annees)")
    ax.set_ylabel("Worst-of" if n_actifs > 1 else "Sous-jacent")
    ax.set_title("Vert = rappele tot. Rouge = perte en capital.")
    ax.grid(alpha=0.28)

    ax = axes[1]
    probas, vivant = [], np.ones(len(ch), dtype=bool)
    for j in indices[:-1]:
        rappel = vivant & (ch[:, j] >= trigger / 100 * S0)
        probas.append(rappel.mean())
        vivant = vivant & ~rappel
    probas.append(vivant.mean())
    ax.bar([f"Obs {k+1}" for k in range(len(probas) - 1)] + ["Echeance"],
           np.array(probas) * 100, color=[VERT] * (len(probas) - 1) + [GRIS])
    ax.set_ylabel("Probabilite (%)")
    ax.set_title("Quand le produit se termine")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(alpha=0.28, axis="y")
    st.pyplot(fig)
    plt.close(fig)

    if st.checkbox("Effet du nombre de sous-jacents et de la correlation"):
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


# =====================================================================
# 5 - BONUS CERTIFICATE
# =====================================================================

elif produit.startswith("5"):
    st.title("Bonus Certificate")
    st.caption("Un niveau bonus garanti si la barriere tient, hausse illimitee.")
    bandeau_modele()

    c1, c2 = st.columns(2)
    T = c1.slider("Maturite (annees)", 0.5, 4.0, 2.0, 0.25)
    bar = c2.slider("Barriere (% du spot)", 50, 90, 70, 5)

    ch, sg = trajectoires(T, bar / 100)
    ST = ch[:, -1].astype(np.float64)
    p_no = M.proba_non_touche(ch, bar / 100 * S0, sg, T)

    def pv_bonus(niveau):
        haut = np.maximum(nominal * ST / S0, nominal * niveau)
        bas = nominal * ST / S0
        return math.exp(-r * T) * (p_no * haut + (1 - p_no) * bas).mean()

    lo, hi, cible = 1.0, 2.5, nominal * (1.0 - marge * T)
    for _ in range(40):
        mid = (lo + hi) / 2
        hi, lo = (mid, lo) if pv_bonus(mid) > cible else (hi, mid)
    bonus = lo

    m = st.columns(3)
    m[0].metric("Niveau bonus offert", f"{bonus*100:,.1f}%")
    m[1].metric("Rendement si marche lateral", f"{(bonus-1)*100:,.1f}%")
    m[2].metric("Proba de toucher la barriere", f"{(1-p_no.mean())*100:,.1f}%")

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
        "down-and-out de strike egal au niveau bonus. Les dividendes financent le put.\n\n"
        "**Ce qui le distingue du BRC** : pas de plafond a la hausse.\n\n"
        "**A ne jamais laisser passer** : la protection est conditionnelle et disparait "
        "si la barriere casse. Ce n'est pas une garantie.")


# =====================================================================
# 6 - TWIN-WIN
# =====================================================================

else:
    st.title("Twin-Win")
    st.caption("Gagne a la hausse ET a la baisse moderee, tant que la barriere tient.")
    bandeau_modele()

    c1, c2 = st.columns(2)
    T = c1.slider("Maturite (annees)", 0.5, 4.0, 2.0, 0.25)
    bar = c2.slider("Barriere (% du spot)", 50, 90, 65, 5)

    ch, sg = trajectoires(T, bar / 100)
    ST = ch[:, -1].astype(np.float64)
    p_no = M.proba_non_touche(ch, bar / 100 * S0, sg, T)
    perf_st = ST / S0 - 1

    def pv_twin(pb):
        haut = np.where(perf_st >= 0, nominal * (1 + perf_st),
                        nominal * (1 + pb * (-perf_st)))
        bas = nominal * ST / S0
        return math.exp(-r * T) * (p_no * haut + (1 - p_no) * bas).mean()

    lo, hi, cible = 0.0, 3.0, nominal * (1.0 - marge * T)
    for _ in range(40):
        mid = (lo + hi) / 2
        hi, lo = (mid, lo) if pv_twin(mid) > cible else (hi, mid)
    part_bas = lo

    m = st.columns(3)
    m[0].metric("Participation a la baisse", f"{part_bas*100:,.0f}%")
    m[1].metric("Participation a la hausse", "100%")
    m[2].metric("Proba de toucher", f"{(1-p_no.mean())*100:,.1f}%")

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
        "**Decomposition** : tracker plus achat d'un put down-and-out. Le put transforme "
        "la baisse en gain et s'evapore si la barriere casse.\n\n"
        "**Le point a ne jamais laisser passer** : entre la barriere et le niveau initial, "
        "le client gagne. Sous la barriere, il perd d'un coup toute cette zone de gain. "
        "La discontinuite est brutale : le profil le plus dangereux a mal expliquer.")


# =====================================================================
# PIED DE PAGE
# =====================================================================

st.markdown("---")
with st.expander("Ce que ce modele fait, et ce qu'il ne fait pas"):
    st.markdown("""
### Traite correctement

**Volatilite stochastique et sauts.** Le modele de Bates genere le skew au lieu de le
postuler, via la correlation spot-vol, et produit de vrais gaps a la baisse. C'est le
standard de marche pour les exotiques actions. Compare les deux modeles dans le
Laboratoire : l'ecart sur le coupon d'un BRC depasse regulierement 2 points.

**Barrieres continues.** Correction par pont brownien, generalisee a une volatilite
variable. Evite le biais classique des simulations discretes, qui sous-estiment
systematiquement la probabilite de franchissement.

**Risque emetteur.** Intensite de defaut deduite du spread de credit, avec taux de
recouvrement. Le spread augmente le coupon, ce qui est contre-intuitif et essentiel a
comprendre en clientele.

**Paniers worst-of avec correlation.** Decomposition de Cholesky. La correlation devient
le parametre de pricing dominant, comme sur un vrai desk.

**Cout de couverture.** Simulation de la couverture en delta avec rebalancement discret
et frais de transaction. Montre que la marge n'est pas un simple prelevement.

### Limites qui restent, et elles sont reelles

**1. Parametres regles a la main, pas calibres.**
C'est la difference principale avec un pricer de production. Un desk recalibre Bates
chaque matin sur les prix d'options cotees. Ici tu choisis les parametres au curseur : le
modele est juste, l'etalonnage est arbitraire. Les niveaux sont donc indicatifs, pas des
prix de marche.

**2. Worst-of simule en Black-Scholes multi-actifs.**
Un Bates multi-dimensionnel demanderait de correler aussi les volatilites entre actifs.
Le mecanisme de correlation est juste, la dynamique de vol ne l'est pas.

**3. Correlation constante.**
En pratique elle monte fortement dans les krachs, exactement quand ca fait mal sur un
worst-of. Le modele sous-estime donc le risque des paniers en periode de stress.

**4. Pas de wrong-way risk.**
Le defaut de l'emetteur est simule independamment du sous-jacent. Dans la realite une
banque fait defaut au pire moment, quand les marches s'effondrent.

**5. Discretisation hebdomadaire de la variance sous Bates.**
Euler a troncature complete introduit un biais modeste sur le processus de variance. Un
schema de type QE (Andersen) serait plus propre pour des maturites longues.

**6. Couverture simulee sur un put vanille, pas sur un put a barriere.**
Sur un vrai BRC le gamma explose autour de la barriere. Le cout affiche est un plancher.
""")

st.caption("Tom Uzan - EDHEC BBA Finance - outil pedagogique, aucune valeur d'offre.")
