"""
=======================================================================
MOTEUR DE PRICING - module partage
Tom Uzan
=======================================================================

Contient les trois modeles et les pricers. Separe de l'interface pour
rester lisible et testable.

Modeles disponibles :
  - Black-Scholes avec skew applique au strike pertinent (rapide)
  - Bates : volatilite stochastique (Heston) + sauts (Merton)

Ajouts par rapport a une version jouet :
  - risque emetteur (intensite de defaut + taux de recouvrement)
  - paniers worst-of avec correlation
  - correction par pont brownien generalisee a une vol variable
=======================================================================
"""

import math
import numpy as np

S0 = 100.0


# ---------------------------------------------------------------------
# Black-Scholes analytique
# ---------------------------------------------------------------------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_cdf_vec(x):
    """
    Loi normale cumulee, vectorisee. Approximation rationnelle de
    Abramowitz-Stegun (precision ~1e-7), sans dependance a scipy.
    Utilisee pour calculer des deltas sur des milliers de trajectoires.
    """
    x = np.asarray(x, dtype=np.float64)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))))
    nd = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    res = 1.0 - nd * poly
    return np.where(x >= 0, res, 1.0 - res)


def bs_call(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def vol_au_strike(strike_pct, vol_atm, skew_pts):
    """
    Skew impose a la main, en Black-Scholes.
    skew_pts = points de vol gagnes pour 10% de baisse du strike.
    Sous Bates ce reglage est inutile : le skew est genere par le modele.
    """
    return max(vol_atm + skew_pts / 100.0 * (1.0 - strike_pct) / 0.10, 0.01)


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def nb_pas(T, pas_par_an=52):
    return max(int(round(pas_par_an * T)), 12)


def simule_bs(spot, T, r, q, sigma, n_paths, pas_par_an=52, seed=42):
    """
    Mouvement brownien geometrique, variables antithetiques, float32.
    Retourne (chemins, sigma) ; sigma scalaire car constante.
    """
    n_steps = nb_pas(T, pas_par_an)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = max(n_paths // 2, 1)
    z = rng.standard_normal((half, n_steps), dtype=np.float32)
    z = np.vstack([z, -z])
    incr = np.float32((r - q - 0.5 * sigma ** 2) * dt) + np.float32(sigma * math.sqrt(dt)) * z
    ch = spot * np.exp(np.cumsum(incr, axis=1, dtype=np.float32))
    return np.hstack([np.full((ch.shape[0], 1), spot, dtype=np.float32), ch]), sigma


def simule_bates(spot, T, r, q, n_paths, v0, theta, kappa, xi, rho,
                 lam, mu_j, sig_j, pas_par_an=52, seed=42):
    """
    Modele de Bates = Heston + sauts de Merton. Standard de marche pour
    les exotiques actions.

        dS/S = (r - q - lam*kbar) dt + sqrt(v) dW1 + (J-1) dN
        dv   = kappa*(theta - v) dt + xi*sqrt(v) dW2
        corr(dW1, dW2) = rho
        ln J ~ N(mu_j, sig_j^2)

    Ce que chaque brique apporte, et pourquoi c'est ce qui manquait :

      rho < 0   La vol monte quand le marche baisse. C'est CA qui cree le
                skew, au lieu de le plaquer a la main. Le modele reproduit
                un fait de marche au lieu de le postuler.

      xi        Vol de la vol. Donne de la convexite au smile : les
                options tres hors de la monnaie deviennent cheres dans
                les deux sens.

      kappa,    Retour a la moyenne. Cree la structure par terme : la vol
      theta     courte peut exploser sans contaminer la vol longue.

      lam,      Sauts. Un gap de -15% en une seance devient possible. Le
      mu_j      brownien geometrique en est incapable, et c'est exactement
                ce qui fait mal sur un produit a barriere.

    Discretisation Euler a troncature complete (Lord et al.) : la variance
    est bornee a zero dans le drift et la diffusion, ce qui evite les
    variances negatives sans biaiser fortement le prix.

    Retourne (chemins, sigma_pas) ou sigma_pas est la vol locale de chaque
    pas, necessaire au pont brownien.
    """
    n_steps = nb_pas(T, pas_par_an)
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    half = max(n_paths // 2, 1)
    n = 2 * half
    z1 = rng.standard_normal((half, n_steps), dtype=np.float32)
    z1 = np.vstack([z1, -z1])
    z2 = rng.standard_normal((half, n_steps), dtype=np.float32)
    z2 = np.vstack([z2, -z2])
    # correlation spot / vol
    w2 = np.float32(rho) * z1 + np.float32(math.sqrt(max(1 - rho ** 2, 0.0))) * z2

    # compensateur de saut : sans lui, le drift n'est plus risque-neutre
    kbar = math.exp(mu_j + 0.5 * sig_j ** 2) - 1.0
    n_jumps = rng.poisson(lam * dt, size=(n, n_steps))
    somme_j = np.where(
        n_jumps > 0,
        n_jumps * mu_j + np.sqrt(np.maximum(n_jumps, 0)) * sig_j
        * rng.standard_normal((n, n_steps)),
        0.0).astype(np.float32)

    logS = np.empty((n, n_steps + 1), dtype=np.float32)
    sig_pas = np.empty((n, n_steps), dtype=np.float32)
    logS[:, 0] = math.log(spot)
    v = np.full(n, v0, dtype=np.float32)
    sdt = np.float32(math.sqrt(dt))

    for t in range(n_steps):
        vp = np.maximum(v, 0.0)
        sv = np.sqrt(vp)
        sig_pas[:, t] = np.maximum(sv, 1e-3)
        logS[:, t + 1] = (logS[:, t]
                          + np.float32((r - q - lam * kbar) * dt) - 0.5 * vp * np.float32(dt)
                          + sv * sdt * z1[:, t]
                          + somme_j[:, t])
        v = v + np.float32(kappa * dt) * (np.float32(theta) - vp) + np.float32(xi) * sv * sdt * w2[:, t]

    return np.exp(logS), sig_pas


def simule_panier(n_actifs, correlation, spot, T, r, q, sigma, n_paths,
                  pas_par_an=52, seed=42):
    """
    Panier worst-of : n actifs correles, on suit le PLUS MAUVAIS a chaque
    instant.

    Pourquoi c'est central : la majorite des autocalls vendus sont des
    worst-of. Le coupon affiche est spectaculaire, et pour cause - il faut
    que les trois sous-jacents tiennent, pas un seul.

    La correlation devient le parametre de pricing dominant :
      correlation haute -> les actifs bougent ensemble -> le worst-of se
        comporte presque comme un actif unique -> coupon plus faible
      correlation basse -> il y a presque toujours un trainard -> le
        worst-of s'effondre -> coupon eleve, risque eleve

    Decomposition de Cholesky sur une matrice de correlation uniforme.
    """
    n_steps = nb_pas(T, pas_par_an)
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    corr = np.full((n_actifs, n_actifs), correlation)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr).astype(np.float32)

    half = max(n_paths // 2, 1)
    z = rng.standard_normal((n_actifs, half, n_steps), dtype=np.float32)
    z = np.concatenate([z, -z], axis=1)
    zc = np.einsum("ij,jkt->ikt", L, z)

    drift = np.float32((r - q - 0.5 * sigma ** 2) * dt)
    diff = np.float32(sigma * math.sqrt(dt))
    perf = np.exp(np.cumsum(drift + diff * zc, axis=2, dtype=np.float32))
    worst = perf.min(axis=0)                       # le plus mauvais a chaque date
    worst = spot * worst
    return np.hstack([np.full((worst.shape[0], 1), spot, dtype=np.float32), worst]), sigma


# ---------------------------------------------------------------------
# Barrieres : pont brownien
# ---------------------------------------------------------------------

def proba_non_touche(chemins, B, sigma, T):
    """
    Correction par pont brownien, generalisee a une volatilite variable.

    Une simulation a pas discrets rate les franchissements qui se
    produisent ENTRE deux observations : elle sous-estime la probabilite
    de toucher et surestime donc le prix.

    Entre deux points S1 et S2 tous deux au-dessus de B :
        p = exp( -2 * ln(S1/B) * ln(S2/B) / (sigma^2 * dt) )

    sigma peut etre un scalaire (Black-Scholes) ou un tableau
    (n_paths, n_steps) donnant la vol locale de chaque pas (Bates).
    """
    n_steps = chemins.shape[1] - 1
    dt = T / n_steps
    x1 = np.log(np.maximum(chemins[:, :-1], 1e-9) / B)
    x2 = np.log(np.maximum(chemins[:, 1:], 1e-9) / B)
    var = (sigma ** 2) * dt
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p = np.exp(-2.0 * x1 * x2 / var)
    p = np.where((x1 <= 0) | (x2 <= 0), 1.0, p)
    return np.clip(np.prod(1.0 - p, axis=1), 0.0, 1.0)


# ---------------------------------------------------------------------
# Risque emetteur
# ---------------------------------------------------------------------

def tire_defaut(n_paths, T, spread_bps, recovery, seed=123):
    """
    Defaut de l'emetteur modelise par une intensite constante.

    Relation credit standard :   lambda = spread / (1 - recovery)

    Un spread de 150 bp avec 40% de recouvrement donne une intensite de
    2,5% par an, soit environ 7% de probabilite de defaut sur 3 ans.

    Ce que ca change, et c'est contre-intuitif : le spread emetteur AUGMENTE
    le coupon. La banque se finance moins cher via le produit structure que
    sur le marche obligataire, et elle reverse cette economie au client.
    Un coupon superieur a la concurrence peut donc simplement signifier que
    l'emetteur est moins bien note. C'est une question a poser au desk.

    Hypothese simplificatrice : defaut independant du sous-jacent. En
    realite il y a du wrong-way risk, une banque fait defaut au pire moment.
    """
    if spread_bps <= 0:
        return np.full(n_paths, np.inf)
    lam = (spread_bps / 10000.0) / max(1.0 - recovery, 0.01)
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0 / lam, size=n_paths)


def pv_recouvrement(tau, T, r, nominal, recovery):
    """Valeur actuelle de ce que recupere le client en cas de defaut avant T."""
    defaut = tau <= T
    return np.where(defaut, recovery * nominal * np.exp(-r * np.minimum(tau, T)), 0.0)


# ---------------------------------------------------------------------
# Pricers
# ---------------------------------------------------------------------

def pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, coupon_pa,
           sigma, tau=None, recovery=0.4, spot0=S0):
    B, K = bar_pct * spot0, strike_pct * spot0
    ST = chemins[:, -1].astype(np.float64)
    if continu:
        p_no = proba_non_touche(chemins, B, sigma, T)
    else:
        p_no = (ST > B).astype(np.float64)
    bas = nominal * ST / K
    remb = np.where(ST >= K, nominal, p_no * nominal + (1.0 - p_no) * bas)
    flux = (remb + coupon_pa * T * nominal) * math.exp(-r * T)

    if tau is not None:
        vivant = tau > T
        flux = np.where(vivant, flux, pv_recouvrement(tau, T, r, nominal, recovery))
    return flux.mean()


def coupon_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, marge_pa,
               sigma, tau=None, recovery=0.4, spot0=S0):
    """Payoff affine en coupon -> resolution directe, sans iteration."""
    base = pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, 0.0,
                  sigma, tau, recovery, spot0)
    if tau is None:
        unite = math.exp(-r * T) * nominal * T
    else:
        unite = (math.exp(-r * T) * nominal * T * (tau > T)).mean()
    cible = nominal * (1.0 - marge_pa * T)
    return max((cible - base) / unite, 0.0) if unite > 1e-9 else 0.0


def idx_obs(n_steps, n_obs):
    return [int(round(n_steps * (k + 1) / n_obs)) for k in range(n_obs)]


def pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                memoire, coupon, tau=None, recovery=0.4, spot0=S0):
    """
    Autocall Phoenix a memoire, avec risque emetteur.

    A chaque observation :
      spot >= trigger         -> remboursement anticipe du nominal
      spot >= barriere coupon -> coupon paye (+ coupons manques si memoire)
    A l'echeance si jamais rappele :
      spot >= barriere capital -> nominal, sinon nominal * spot / spot initial

    Si l'emetteur fait defaut avant une date de flux, ce flux n'est jamais
    paye ; le client recupere le taux de recouvrement sur le nominal.
    """
    n_paths, n_steps = chemins.shape[0], chemins.shape[1] - 1
    idx = idx_obs(n_steps, n_obs)
    dt_obs = T / n_obs
    vivant = np.ones(n_paths, dtype=bool)
    manques = np.zeros(n_paths)
    pv = np.zeros(n_paths)
    inf = np.full(n_paths, np.inf) if tau is None else tau

    for k, i in enumerate(idx):
        t = (k + 1) * dt_obs
        disc = math.exp(-r * t)
        solvable = inf > t
        S = chemins[:, i].astype(np.float64)
        paye = vivant & solvable & (S >= bar_cp * spot0)
        n_cp = (manques + 1.0) if memoire else np.ones(n_paths)
        pv += np.where(paye, coupon * nominal * n_cp * disc, 0.0)
        manques = np.where(paye, 0.0, manques + 1.0)
        if k < len(idx) - 1:
            rappel = vivant & solvable & (S >= trigger * spot0)
            pv += np.where(rappel, nominal * disc, 0.0)
            vivant = vivant & ~rappel

    ST = chemins[:, -1].astype(np.float64)
    final = np.where(ST >= bar_cap * spot0, nominal, nominal * ST / spot0)
    pv += np.where(vivant & (inf > T), final * math.exp(-r * T), 0.0)

    if tau is not None:
        # defaut avant le rappel : recouvrement, et plus aucun flux ensuite
        pv = np.where(vivant & (tau <= T),
                      pv + pv_recouvrement(tau, T, r, nominal, recovery), pv)
    return pv.mean()


def coupon_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, marge_pa, tau=None, recovery=0.4, spot0=S0):
    a = pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, 0.0, tau, recovery, spot0)
    b = pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, 0.01, tau, recovery, spot0)
    pente = (b - a) / 0.01
    cible = nominal * (1.0 - marge_pa * T)
    return max((cible - a) / pente, 0.0) if pente > 1e-9 else 0.0


def stats_autocall(chemins, T, n_obs, trigger, bar_cap, spot0=S0):
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
    return duree.mean(), 1.0 - vivant.mean(), (vivant & (ST < bar_cap * spot0)).mean()


def vol_implicite_bates(chemins, T, r, q, strikes_pct, spot0=S0):
    """
    Extrait le smile genere par le modele : on price des vanilles par
    Monte Carlo, puis on inverse Black-Scholes pour retrouver la vol
    implicite de chaque strike.

    C'est la verification qui compte : si le modele ne produit pas de
    skew, ses parametres ne servent a rien.
    """
    ST = chemins[:, -1].astype(np.float64)
    out = []
    for kp in strikes_pct:
        K = kp * spot0
        if kp >= 1.0:
            prix = math.exp(-r * T) * np.maximum(ST - K, 0).mean()
            f = lambda s: bs_call(spot0, K, T, r, s, q) - prix
        else:
            prix = math.exp(-r * T) * np.maximum(K - ST, 0).mean()
            # parite call-put pour se ramener a un call
            call = prix + spot0 * math.exp(-q * T) - K * math.exp(-r * T)
            f = lambda s: bs_call(spot0, K, T, r, s, q) - call
        lo, hi = 0.01, 3.0
        if f(lo) * f(hi) > 0:
            out.append(np.nan)
            continue
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        out.append((lo + hi) / 2)
    return out
