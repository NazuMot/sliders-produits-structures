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

Produits et risques ajoutes ensuite (tous desactives par defaut : sans
parametre supplementaire, les resultats sont ceux d'avant) :
  - autocall a seuil de rappel degressif (step-down)
  - indice a decrement (dividende forfaitaire en points)
  - Discount Certificate (formule fermee et Monte Carlo)
  - correlation qui monte quand le marche baisse (worst-of)
  - wrong-way risk : defaut de l'emetteur correle au sous-jacent
  - grecques de l'autocall par rebond avec les memes tirages
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


def norm_inv_vec(p):
    """
    Inverse de la loi normale cumulee, vectorisee, sans scipy.

    Point de depart d'Acklam (precision ~1e-9), puis un pas de Halley sur
    norm_cdf_vec : l'aller-retour norm_cdf_vec(norm_inv_vec(p)) redonne p a
    la precision machine. C'est ce qui garantit qu'une correlation de
    wrong-way presque nulle redonne exactement les temps de defaut actuels.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0 - 1e-16)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)

    def queue(u):
        s = np.sqrt(-2.0 * np.log(u))
        return ((((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5])
                / ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1.0))

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        m = p - 0.5
        t = m * m
        x = (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * m \
            / (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1.0)
        x = np.where(p < 0.02425, queue(p), x)
        x = np.where(p > 1.0 - 0.02425, -queue(1.0 - p), x)
        # un pas de Halley pour coller a norm_cdf_vec
        e = norm_cdf_vec(x) - p
        u = e * math.sqrt(2.0 * math.pi) * np.exp(0.5 * x * x)
        x = np.where(np.isfinite(u), x - u / (1.0 + 0.5 * x * u), x)
    return x


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


def simule_bs(spot, T, r, q, sigma, n_paths, pas_par_an=52, seed=42, decrement=0.0):
    """
    Mouvement brownien geometrique, variables antithetiques, float32.
    Retourne (chemins, sigma) ; sigma scalaire car constante.

    decrement > 0 : indice a decrement, en points par an (meme unite que le
    spot : avec un spot a 100, 1 point = 1 % du niveau initial). Voir
    _decrement_bs. Passer alors q = 0 : l'indice reinvestit ses dividendes.
    """
    n_steps = nb_pas(T, pas_par_an)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = max(n_paths // 2, 1)
    z = rng.standard_normal((half, n_steps), dtype=np.float32)
    z = np.vstack([z, -z])
    incr = np.float32((r - q - 0.5 * sigma ** 2) * dt) + np.float32(sigma * math.sqrt(dt)) * z
    if decrement > 0:
        ch = _decrement_bs(incr, spot, decrement, dt)
    else:
        ch = spot * np.exp(np.cumsum(incr, axis=1, dtype=np.float32))
    return np.hstack([np.full((ch.shape[0], 1), spot, dtype=np.float32), ch]), sigma


PENTE_VOL_LOCALE = 1.9


def simule_lv(spot, T, r, q, vol_atm, skew_pts, n_paths, pas_par_an=52, seed=42,
              pente=PENTE_VOL_LOCALE, decrement=0.0):
    """
    Black-Scholes + skew en VOLATILITE LOCALE : la vol de chaque pas depend du
    niveau atteint par la trajectoire,
        sigma(S, t) = vol_atm + pente * skew * ln(F(t) / S) / 0.10
    (bornee entre 5 % et 150 %), F(t) etant le forward.

    Pourquoi : l'ancienne methode prenait UNE vol par produit, lue sur le skew
    au niveau de la barriere. Une barriere plus basse recevait donc une vol
    plus haute, et des que le skew etait sensible le coupon d'un autocall
    montait quand on baissait la barriere de capital : le sens s'inversait.
    Ici les memes trajectoires servent pour toutes les barrieres, le
    franchissement d'un niveau bas se fait avec la vol qui regne a ce niveau,
    et le coupon baisse bien quand la protection augmente.

    pente = 1,9 : la vol locale doit pencher environ deux fois plus que la vol
    implicite pour la reproduire (regle classique). Avec skew 2, les vols
    implicites des puts 80/90/100 % a 3 ans sortent a 25,9 / 23,9 / 22,0 %
    pour une cible de 26 / 24 / 22 %. Avec un skew tres fort (6), les ailes
    sortent moins pentues que la cible.

    decrement > 0 : indice a decrement (D points par an, q = 0), meme regle que
    simule_bs : S(k+1) = S(k) * rendement - D * dt, borne a 0.

    Retourne (chemins, sigma_pas) comme simule_bates : le pont brownien
    utilise la vol de chaque pas. skew_pts = 0 : exactement simule_bs.
    """
    if skew_pts == 0:
        return simule_bs(spot, T, r, q, vol_atm, n_paths, pas_par_an, seed, decrement=decrement)
    n_steps = nb_pas(T, pas_par_an)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = max(n_paths // 2, 1)
    z = rng.standard_normal((half, n_steps), dtype=np.float32)
    z = np.vstack([z, -z])
    a = np.float32(pente * skew_pts / 100.0 / 0.10)
    racine_dt = np.float32(math.sqrt(dt))
    ch = np.empty((z.shape[0], n_steps + 1), dtype=np.float32)
    sg = np.empty((z.shape[0], n_steps), dtype=np.float32)
    ch[:, 0] = spot
    niveau = np.full(z.shape[0], np.float32(spot), dtype=np.float32)
    for k in range(n_steps):
        fwd = (forward_decrement(spot, k * dt, r, decrement, q) if decrement > 0
               else spot * math.exp((r - q) * k * dt))
        x = np.log(np.maximum(niveau, np.float32(1e-6)))
        s = np.clip(np.float32(vol_atm) + a * (np.float32(math.log(max(fwd, 1e-6))) - x),
                    0.05, 1.5).astype(np.float32)
        sg[:, k] = s
        niveau = niveau * np.exp((np.float32(r - q) - np.float32(0.5) * s * s) * np.float32(dt)
                                 + s * racine_dt * z[:, k])
        if decrement > 0:
            niveau = np.maximum(niveau - np.float32(decrement * dt), np.float32(0.0))
        ch[:, k + 1] = niveau
    if decrement == 0:
        # recalage du forward date par date (bruit Monte Carlo) : le prix d'un
        # forward reste exact, comme avec les variables antithetiques en BS pur
        forward = spot * np.exp((r - q) * dt * np.arange(n_steps + 1))
        ch *= (forward / ch.mean(axis=0, dtype=np.float64)).astype(np.float32)
    return ch, sg


def _decrement_bs(incr, spot, decrement, dt):
    """
    Indice a decrement :   dS = (r S - D) dt + sigma S dW

    L'indice reinvestit tous ses dividendes (indice total return, drift r),
    puis on lui retire chaque annee un montant FIXE de D points. C'est la
    regle reelle de ces indices, appliquee ici a chaque pas :
        S(k+1) = S(k) * rendement(k+1) - D * dt
    ce qui se deroule en forme fermee, sans boucle :
        S(k) = Phi(k) * ( S0 - D dt * somme_{j<=k} 1 / Phi(j) )
    ou Phi est le brownien geometrique de drift r. L'indice est borne a 0
    (en pratique jamais atteint sur des horizons d'autocall).

    La vol en log reste sigma : le pont brownien s'applique tel quel.
    """
    phi = np.exp(np.cumsum(incr, axis=1, dtype=np.float32))
    retire = np.cumsum(np.float32(decrement * dt) / phi, axis=1, dtype=np.float32)
    return np.maximum(phi * (np.float32(spot) - retire), np.float32(0.0))


def simule_bates(spot, T, r, q, n_paths, v0, theta, kappa, xi, rho,
                 lam, mu_j, sig_j, pas_par_an=52, seed=42, decrement=0.0):
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

    decrement > 0 : indice a decrement en points par an (voir _decrement_bs),
    avec les memes tirages. Passer alors q = 0.
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

    if decrement > 0:
        return _bates_decrement(spot, n, n_steps, dt, r, q, v0, theta, kappa, xi,
                                lam, kbar, z1, w2, somme_j, decrement)

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


def _bates_decrement(spot, n, n_steps, dt, r, q, v0, theta, kappa, xi, lam, kbar,
                     z1, w2, somme_j, decrement):
    """Meme schema que simule_bates, en niveau, avec le decrement retire a chaque pas."""
    S = np.empty((n, n_steps + 1), dtype=np.float32)
    sig_pas = np.empty((n, n_steps), dtype=np.float32)
    S[:, 0] = spot
    v = np.full(n, v0, dtype=np.float32)
    sdt = np.float32(math.sqrt(dt))
    retire = np.float32(decrement * dt)
    for t in range(n_steps):
        vp = np.maximum(v, 0.0)
        sv = np.sqrt(vp)
        sig_pas[:, t] = np.maximum(sv, 1e-3)
        incr = (np.float32((r - q - lam * kbar) * dt) - 0.5 * vp * np.float32(dt)
                + sv * sdt * z1[:, t] + somme_j[:, t])
        S[:, t + 1] = np.maximum(S[:, t] * np.exp(incr) - retire, np.float32(0.0))
        v = v + np.float32(kappa * dt) * (np.float32(theta) - vp) + np.float32(xi) * sv * sdt * w2[:, t]
    return S, sig_pas


def simule_panier(n_actifs, correlation, spot, T, r, q, sigma, n_paths,
                  pas_par_an=52, seed=42, correlation_stress=None, seuil_stress=0.10):
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

    correlation_stress (desactivee si None) : la correlation passe a ce
    niveau tant que le marche (moyenne geometrique du panier) est a plus de
    seuil_stress sous son plus haut. Voir _panier_stress.
    """
    if correlation_stress is not None:
        worst, _ = _panier_stress(n_actifs, correlation, correlation_stress, seuil_stress,
                                  spot, T, r, q, sigma, n_paths, pas_par_an, seed)
        return worst, sigma

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


def _correlation_uniforme(n_actifs, c):
    m = np.full((n_actifs, n_actifs), c)
    np.fill_diagonal(m, 1.0)
    return np.linalg.cholesky(m).astype(np.float32)


def _panier_stress(n_actifs, correlation, correlation_stress, seuil_stress, spot, T, r,
                   q, sigma, n_paths, pas_par_an=52, seed=42, premier_pas=1.0):
    """
    Worst-of avec correlation de stress : deux regimes.

    Fait de marche : les correlations montent dans les ventes paniques, tout
    baisse en meme temps. On le reproduit simplement : a chaque pas, si le
    marche (moyenne geometrique des sous-jacents) est a plus de seuil_stress
    sous son plus haut, les chocs de la semaine sont tires avec la
    correlation de stress ; sinon avec la correlation normale.

    Le regime est decide AVANT le tirage de la semaine, sur le passe
    seulement : conditionnellement, les chocs restent gaussiens centres
    reduits, donc chaque sous-jacent garde exactement sa loi
    Black-Scholes (drift risque-neutre, vol sigma). Seule la dependance
    entre eux change.

    Memes tirages que simule_panier : avec correlation_stress = correlation,
    on retombe sur ses chemins. Boucle sur les pas, sans stocker les chocs
    correles : moins de memoire que la version constante.

    premier_pas < 1 raccourcit le premier pas (memes tirages) : c'est le
    meme produit vu un peu plus tard, ce qui sert au theta.

    Retourne (worst, diagnostic) ; diagnostic donne la part du temps passee
    en stress et la performance moyenne finale de chaque actif (controle
    de martingale).
    """
    n_steps = nb_pas(T, pas_par_an)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    L_n = _correlation_uniforme(n_actifs, correlation)
    L_s = _correlation_uniforme(n_actifs, correlation_stress)

    half = max(n_paths // 2, 1)
    n = 2 * half
    z = rng.standard_normal((n_actifs, half, n_steps), dtype=np.float32)

    drift = np.float32((r - q - 0.5 * sigma ** 2) * dt)
    diff = np.float32(sigma * math.sqrt(dt))
    limite = np.float32(math.log(max(1.0 - seuil_stress, 1e-6)))

    logp = np.zeros((n_actifs, n), dtype=np.float32)    # log-performance de chaque actif
    pic = np.zeros(n, dtype=np.float32)                 # plus haut du marche
    worst = np.empty((n, n_steps + 1), dtype=np.float32)
    worst[:, 0] = spot
    temps_stress = 0.0
    for t in range(n_steps):
        zt = np.concatenate([z[:, :, t], -z[:, :, t]], axis=1)
        stress = (logp.mean(axis=0) - pic) < limite
        zc = np.where(stress, np.einsum("ij,jk->ik", L_s, zt), np.einsum("ij,jk->ik", L_n, zt))
        if t == 0 and premier_pas != 1.0:
            dt0 = dt * premier_pas
            logp += (np.float32((r - q - 0.5 * sigma ** 2) * dt0)
                     + np.float32(sigma * math.sqrt(dt0)) * zc)
        else:
            logp += drift + diff * zc
        pic = np.maximum(pic, logp.mean(axis=0))
        worst[:, t + 1] = spot * np.exp(logp.min(axis=0))
        temps_stress += float(stress.mean())
    diagnostic = {"part_stress": temps_stress / n_steps,
                  "perf_finale": np.exp(logp.astype(np.float64)).mean(axis=1)}
    return worst, diagnostic


# ---------------------------------------------------------------------
# Indice a decrement : outils
# ---------------------------------------------------------------------

def decrement_base100(points, niveau_indice):
    """
    Convertit un decrement exprime en points d'indice en points pour une
    base 100, l'unite du moteur (sous-jacent initial a 100), soit aussi en
    % du niveau initial.
        50 points sur un indice a 5 000 -> 1   (1 % du niveau initial)
        50 points sur une base 1 000    -> 5   (5 %)
    """
    return 100.0 * points / niveau_indice


def forward_decrement(spot, T, r, decrement, q=0.0):
    """
    Forward d'un indice a decrement en points :
        F = S e^{(r-q)T} - D (e^{(r-q)T} - 1) / (r - q)
    Le decrement retire D points par an, chacun capitalise jusqu'a T.
    """
    g = r - q
    if abs(g) < 1e-12:
        return spot - decrement * T
    return spot * math.exp(g * T) - decrement * (math.exp(g * T) - 1.0) / g


def dividende_equivalent(niveau, decrement):
    """
    Poids du decrement en % du niveau courant de l'indice.

    C'est tout le sujet : un decrement de 1 point pese 1 % quand l'indice
    est a 100, mais 2 % quand il est tombe a 50. Un dividende classique,
    lui, baisse avec les cours. Le decrement en points accelere donc la
    chute dans les mauvais scenarios, exactement la ou se joue la perte en
    capital.
    """
    return decrement / np.maximum(niveau, 1e-9)


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

def tire_defaut(n_paths, T, spread_bps, recovery, seed=123, correlation=0.0, chemins=None):
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

    Par defaut, le defaut est independant du sous-jacent. En realite il y a
    du wrong-way risk : une banque fait defaut au pire moment.

    Wrong-way risk (correlation != 0, chemins fournis) : copule gaussienne
    entre le temps de defaut et la performance finale du sous-jacent.
      - chaque scenario recoit un score gaussien X d'apres le rang de son
        niveau final (le plus bas -> X tres negatif), quel que soit le
        modele (Black-Scholes, Bates, worst-of) ;
      - le tirage de defaut actuel est converti en gaussienne e, melange en
        Y = rho X + sqrt(1 - rho^2) e, puis reconverti en temps de defaut.
    La loi du temps de defaut ne change pas (meme probabilite de defaut,
    meme spread) : seuls les scenarios ou il tombe changent. rho > 0 : la
    banque fait defaut quand le sous-jacent s'effondre. rho = 0 redonne
    exactement les tirages actuels, et un rho minuscule en reste tout pres.

    chemins : matrice de trajectoires (on prend la derniere colonne) ou
    directement le vecteur des niveaux finaux.
    """
    if spread_bps <= 0:
        return np.full(n_paths, np.inf)
    lam = (spread_bps / 10000.0) / max(1.0 - recovery, 0.01)
    rng = np.random.default_rng(seed)
    tau = rng.exponential(1.0 / lam, size=n_paths)
    if correlation == 0 or chemins is None:
        return tau

    finaux = np.asarray(chemins)
    finaux = finaux[:, -1] if finaux.ndim == 2 else finaux
    if len(finaux) != n_paths:
        raise ValueError("chemins doit avoir une ligne par trajectoire")
    rangs = np.empty(n_paths)
    rangs[np.argsort(finaux, kind="stable")] = np.arange(n_paths)
    X = norm_inv_vec((rangs + 0.5) / n_paths)

    e = norm_inv_vec(-np.expm1(-lam * tau))        # P(defaut avant tau), en gaussienne
    rho = float(np.clip(correlation, -0.999, 0.999))
    Y = rho * X + math.sqrt(1.0 - rho ** 2) * e
    with np.errstate(divide="ignore"):
        return -np.log1p(-np.minimum(norm_cdf_vec(Y), 1.0)) / lam


def pv_recouvrement(tau, T, r, nominal, recovery):
    """Valeur actuelle de ce que recupere le client en cas de defaut avant T."""
    defaut = tau <= T
    return np.where(defaut, recovery * nominal * np.exp(-r * np.minimum(tau, T)), 0.0)


# ---------------------------------------------------------------------
# Pricers
# ---------------------------------------------------------------------

def pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, coupon_pa,
           sigma, tau=None, recovery=0.4, spot0=S0, creance="nominal"):
    """
    creance : ce sur quoi porte le recouvrement en cas de defaut.
      "nominal" (defaut) : le client recupere recovery x nominal.
      "valeur"           : recovery x ce que valait le produit dans ce
                           scenario. C'est la regle des EMTN (la creance est
                           la valeur de marche de la note), et la seule qui
                           ait un sens avec du wrong-way risk : apres un
                           krach, la note vaut deja peu, la creance aussi.
    """
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
        if creance == "valeur":
            flux = np.where(vivant, flux, recovery * flux)
        else:
            flux = np.where(vivant, flux, pv_recouvrement(tau, T, r, nominal, recovery))
    return flux.mean()


def coupon_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, marge_pa,
               sigma, tau=None, recovery=0.4, spot0=S0, creance="nominal"):
    """Payoff affine en coupon -> resolution directe, sans iteration."""
    base = pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, 0.0,
                  sigma, tau, recovery, spot0, creance)
    if tau is None:
        unite = math.exp(-r * T) * nominal * T
    elif creance == "valeur":
        unite = (math.exp(-r * T) * nominal * T * np.where(tau > T, 1.0, recovery)).mean()
    else:
        unite = (math.exp(-r * T) * nominal * T * (tau > T)).mean()
    cible = nominal * (1.0 - marge_pa * T)
    return max((cible - base) / unite, 0.0) if unite > 1e-9 else 0.0


def profil_brc(chemins, T, nominal, bar_pct, strike_pct, continu, coupon_pa, sigma,
               tau=None, recovery=0.4, spot0=S0, creance="nominal", niveau_es=0.05):
    """
    Distribution de ce que touche le client d'un BRC (non actualise).

    Un prix moyen ne dit pas tout : avec du wrong-way risk, ce qui change
    surtout c'est la forme des pertes. Le krach du sous-jacent et le defaut
    de l'emetteur arrivent ensemble ; le client perd deux fois au meme
    moment.

    Barriere continue : chaque trajectoire donne deux issues, barriere
    intacte (poids = probabilite de non-franchissement du pont brownien) ou
    touchee. Pas de bruit ajoute.

    Retourne :
      p_defaut          probabilite de defaut de l'emetteur avant T
      p_perte_capital   probabilite de recuperer moins que le nominal
                        (defaut compris, coupon exclu)
      p_double          probabilite de defaut ET de perte sur le sous-jacent
      esperance         rendement total moyen (coupon compris)
      es                perte moyenne dans les niveau_es pires scenarios
                        (expected shortfall, en % du nominal, positive)
    """
    B, K = bar_pct * spot0, strike_pct * spot0
    ST = chemins[:, -1].astype(np.float64)
    n = len(ST)
    if continu:
        p_no = proba_non_touche(chemins, B, sigma, T)
    else:
        p_no = (ST > B).astype(np.float64)
    touche = np.where(ST >= K, nominal, nominal * ST / K)
    remb = np.concatenate([np.full(n, nominal), touche])
    poids = np.concatenate([p_no, 1.0 - p_no]) / n
    defaut = np.zeros(2 * n, dtype=bool) if tau is None else np.concatenate([tau <= T] * 2)

    sain = remb + coupon_pa * T * nominal
    if creance == "valeur":
        recu = np.where(defaut, recovery * sain, sain)
    else:
        recu = np.where(defaut, recovery * nominal, sain)
    rendement = recu / nominal - 1.0
    perte_marche = remb < nominal - 1e-9

    ordre = np.argsort(rendement, kind="stable")
    w, x = poids[ordre], rendement[ordre]
    cumul = np.cumsum(w)
    k = int(np.searchsorted(cumul, niveau_es))
    reste = niveau_es - (cumul[k - 1] if k > 0 else 0.0)
    es = -(np.sum(w[:k] * x[:k]) + reste * x[min(k, len(x) - 1)]) / niveau_es

    return {"p_defaut": 0.0 if tau is None else float((tau <= T).mean()),
            "p_perte_capital": float(poids[defaut | perte_marche].sum()),
            "p_double": float(poids[defaut & perte_marche].sum()),
            "esperance": float(np.sum(poids * rendement)),
            "es": float(es)}


def idx_obs(n_steps, n_obs):
    return [int(round(n_steps * (k + 1) / n_obs)) for k in range(n_obs)]


def niveaux_rappel(trigger, n_obs):
    """
    Seuils de rappel, un par date d'observation.

    trigger scalaire : meme seuil a chaque date (autocall classique).
    trigger suite    : un seuil par date (step-down). Une suite de n_obs - 1
                       niveaux est acceptee : a la derniere date, c'est la
                       barriere de capital qui joue, pas le trigger.
    Un niveau infini (math.inf) interdit le rappel a cette date : periode
    de non-rappel en debut de vie.
    """
    if np.ndim(trigger) == 0:
        return [trigger] * n_obs
    niv = [float(x) for x in trigger]
    if len(niv) == n_obs - 1:
        niv.append(niv[-1])
    if len(niv) != n_obs:
        raise ValueError(f"{len(niv)} seuils de rappel pour {n_obs} dates d'observation")
    return niv


def trigger_degressif(n_obs, depart=1.0, pas=0.05, plancher=0.0, obs_par_palier=1):
    """
    Seuils d'un autocall step-down : le trigger baisse de `pas` tous les
    `obs_par_palier` observations, sans descendre sous `plancher`.
    Ex. observations trimestrielles, 100 % puis -5 % par an :
        trigger_degressif(n_obs, 1.0, 0.05, obs_par_palier=4)

    Pourquoi ca se vend : plus le temps passe, plus il devient facile
    d'etre rappele. Le client sort plus souvent avec son gain meme si le
    marche a un peu baisse, et va moins souvent jusqu'a l'echeance, la ou
    se joue la perte en capital. Le prix : un coupon plus faible.
    """
    return [max(depart - pas * (k // obs_par_palier), plancher) for k in range(n_obs)]


def _poids_au_dessus(S, niveau, lissage):
    """
    Indicatrice S >= niveau. Avec lissage > 0 (largeur en unite de spot),
    rampe lineaire centree sur le niveau, comme le call spread qu'un desk
    utilise pour couvrir une digitale : sans elle, une grecque par
    differences finies sur un payoff a sauts est inexploitable.
    """
    if lissage <= 0:
        return (S >= niveau).astype(np.float64)
    return np.clip((S - niveau) / lissage + 0.5, 0.0, 1.0)


def _flux_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap, memoire,
                   coupon, tau=None, recovery=0.4, spot0=S0, lissage=0.0, recul=0.0):
    """
    Valeur actuelle de l'autocall, trajectoire par trajectoire.

    lissage : largeur des rampes en % du spot initial (0 = payoff exact).
    recul   : rapproche toutes les dates de flux de `recul` annees (pour
              l'actualisation et le defaut). Sert au theta, avec des
              trajectoires dont le premier pas a ete raccourci d'autant.
    Les etats (vivant, coupon paye, rappel) sont des poids entre 0 et 1 ;
    sans lissage ils valent exactement 0 ou 1, comme des booleens.
    """
    n_paths, n_steps = chemins.shape[0], chemins.shape[1] - 1
    idx = idx_obs(n_steps, n_obs)
    dt_obs = T / n_obs
    seuils = niveaux_rappel(trigger, n_obs)
    largeur = lissage * spot0
    T_fin = T - recul
    vivant = np.ones(n_paths)
    manques = np.zeros(n_paths)
    pv = np.zeros(n_paths)
    inf = np.full(n_paths, np.inf) if tau is None else tau

    for k, i in enumerate(idx):
        t = (k + 1) * dt_obs - recul
        disc = math.exp(-r * t)
        solvable = inf > t
        S = chemins[:, i].astype(np.float64)
        paye = vivant * solvable * _poids_au_dessus(S, bar_cp * spot0, largeur)
        n_cp = (manques + 1.0) if memoire else np.ones(n_paths)
        pv += paye * (coupon * nominal * n_cp * disc)
        manques = (1.0 - paye) * (manques + 1.0)
        if k < len(idx) - 1:
            rappel = vivant * solvable * _poids_au_dessus(S, seuils[k] * spot0, largeur)
            pv += rappel * (nominal * disc)
            vivant = vivant - rappel

    ST = chemins[:, -1].astype(np.float64)
    au_dessus = _poids_au_dessus(ST, bar_cap * spot0, largeur)
    final = au_dessus * nominal + (1.0 - au_dessus) * (nominal * ST / spot0)
    pv += (vivant * (inf > T_fin)) * (final * math.exp(-r * T_fin))

    if tau is not None:
        # defaut avant le rappel : recouvrement, et plus aucun flux ensuite
        pv = pv + (vivant * (tau <= T_fin)) * pv_recouvrement(tau, T_fin, r, nominal, recovery)
    return pv


def pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                memoire, coupon, tau=None, recovery=0.4, spot0=S0):
    """
    Autocall Phoenix a memoire, avec risque emetteur.

    A chaque observation :
      spot >= trigger         -> remboursement anticipe du nominal
      spot >= barriere coupon -> coupon paye (+ coupons manques si memoire)
    A l'echeance si jamais rappele :
      spot >= barriere capital -> nominal, sinon nominal * spot / spot initial

    trigger : un niveau, ou une suite de niveaux (step-down), voir
    niveaux_rappel.

    Si l'emetteur fait defaut avant une date de flux, ce flux n'est jamais
    paye ; le client recupere le taux de recouvrement sur le nominal.
    """
    return _flux_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                          memoire, coupon, tau, recovery, spot0).mean()


def coupon_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, marge_pa, tau=None, recovery=0.4, spot0=S0, marge_sur="vie"):
    """
    Coupon par observation qui equilibre le produit, marge deduite.

    marge_sur : sur quelle duree la marge annuelle est prelevee.
      "vie" (defaut) : la duree de vie ATTENDUE du produit. C'est ce que fait
                       un desk : un autocall rappele en moyenne apres 9 mois ne
                       paie pas 3 ans de marge. Avant, avec un trigger bas, le
                       produit vivait ~3 mois mais payait la marge sur la
                       maturite maximale et le coupon tombait a zero.
      "maturite"     : la maturite maximale (ancien calcul).
    """
    duree = (stats_autocall(chemins, T, n_obs, trigger, bar_cap, spot0)[0]
             if marge_sur == "vie" else T)
    a = pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, 0.0, tau, recovery, spot0)
    b = pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, 0.01, tau, recovery, spot0)
    pente = (b - a) / 0.01
    cible = nominal * (1.0 - marge_pa * duree)
    return max((cible - a) / pente, 0.0) if pente > 1e-9 else 0.0


def stats_autocall(chemins, T, n_obs, trigger, bar_cap, spot0=S0):
    """Duree de vie moyenne, proba de rappel anticipe, proba de perte en capital."""
    n_paths, n_steps = chemins.shape[0], chemins.shape[1] - 1
    idx = idx_obs(n_steps, n_obs)
    dt_obs = T / n_obs
    seuils = niveaux_rappel(trigger, n_obs)
    vivant = np.ones(n_paths, dtype=bool)
    duree = np.full(n_paths, T)
    for k, i in enumerate(idx[:-1]):
        rappel = vivant & (chemins[:, i] >= seuils[k] * spot0)
        duree = np.where(rappel, (k + 1) * dt_obs, duree)
        vivant = vivant & ~rappel
    ST = chemins[:, -1]
    return duree.mean(), 1.0 - vivant.mean(), (vivant & (ST < bar_cap * spot0)).mean()


def compare_decrement(T, r, q, sigma, n_obs, trigger, bar_cp, bar_cap, memoire, marge_pa,
                      decrement, nominal=1000.0, n_paths=20000, seed=42, decote_div=0.0,
                      bates=None, pas_par_an=52, spot0=S0):
    """
    Autocall sur indice classique contre indice a decrement, a vue de marche
    egale (meme taux, meme vol ou memes parametres Bates, memes tirages).

    Trois colonnes, pour separer les deux effets :
      "classique"        indice de prix, dividende q. Le desk le price avec
                         q - decote_div : il est structurellement long
                         dividendes (il en perd si les dividendes baissent,
                         cf. 2020), donc il retient des dividendes prudents.
      "decrement_pct"    meme indice, dividendes reinvestis, decrement en %
                         du niveau courant = decrement / spot0. Equivaut a
                         un dividende certain : seul l'effet NIVEAU joue
                         (decrement contre dividende retenu) et plus aucun
                         risque de dividende pour la banque.
      "decrement_points" decrement fixe en points : meme poids au depart,
                         mais il pese plus quand l'indice baisse (effet
                         POINTS). Proba de perte plus forte, coupon aussi.

    decrement : points par an pour une base spot0 (1 = 1 % du niveau initial).
    bates     : None (Black-Scholes, vol sigma) ou dict v0, theta, kappa, xi,
                rho, lam, mu_j, sig_j.

    Retourne {colonne: {coupon, duree, p_rappel, p_perte, forward}} ; coupon
    par observation, forward = moyenne Monte Carlo du niveau final.
    """
    def chemins(q_, dec):
        if bates is None:
            return simule_bs(spot0, T, r, q_, sigma, n_paths, pas_par_an, seed, dec)[0]
        b = bates
        return simule_bates(spot0, T, r, q_, n_paths, b["v0"], b["theta"], b["kappa"], b["xi"],
                            b["rho"], b["lam"], b["mu_j"], b["sig_j"], pas_par_an, seed, dec)[0]

    out = {}
    for nom, q_, dec in [("classique", q - decote_div, 0.0),
                         ("decrement_pct", decrement / spot0, 0.0),
                         ("decrement_points", 0.0, decrement)]:
        ch = chemins(q_, dec)
        cpn = coupon_autocall(ch, T, r, nominal, n_obs, trigger, bar_cp, bar_cap, memoire,
                              marge_pa, spot0=spot0)
        duree, p_rappel, p_perte = stats_autocall(ch, T, n_obs, trigger, bar_cap, spot0)
        out[nom] = {"coupon": cpn, "duree": duree, "p_rappel": p_rappel, "p_perte": p_perte,
                    "forward": float(ch[:, -1].astype(np.float64).mean())}
        del ch
    return out


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


# ---------------------------------------------------------------------
# Options a barriere en formule fermee, et grecques
# ---------------------------------------------------------------------

def bs_put(S, K, T, r, sigma, q=0.0):
    if T <= 1e-9 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


# ---------------------------------------------------------------------
# Discount Certificate
# ---------------------------------------------------------------------

def _offre_discount(valeur, S, K, T, marge_pa):
    """Passe de la valeur equitable au prix client, a la decote et au rendement maximal."""
    prix = valeur / max(1.0 - marge_pa * T, 1e-9)
    return {"valeur": valeur,
            "prix": prix,
            "decote": 1.0 - prix / S,
            "rendement_max": K / prix - 1.0,
            "rendement_max_pa": (K / prix) ** (1.0 / T) - 1.0 if T > 0 else 0.0,
            "point_mort": prix}


def discount_certificate(S, K, T, r, sigma, q=0.0, marge_pa=0.0):
    """
    Discount Certificate en formule fermee (Black-Scholes).

    Ce que c'est : le client achete le sous-jacent moins cher que le
    marche (la decote), en echange d'un plafond : a l'echeance il recoit
    min(S_T, K), le cap etant aussi le strike.

    Construction : detenir le sous-jacent sans ses dividendes, et vendre
    un call de strike K.
        valeur = S e^{-qT} - call(K)
    Deux sources financent la decote : les dividendes abandonnes et la
    prime du call vendu. Plus la vol est haute, plus le call vaut cher,
    plus la decote est forte.

    Pourquoi un client l'achete : vue neutre a moderement haussiere. Il
    gagne si le marche monte un peu, stagne ou meme baisse moins que la
    decote ; il perd la hausse au-dela du cap.

    La marge banque se preleve comme ailleurs : valeur = prix x (1 - marge x T).
    Sous Black-Scholes avec skew, passer sigma = vol au strike K.

    Prix et point mort en unites de sous-jacent (S = 100 : en % du spot).
    """
    valeur = S * math.exp(-q * T) - bs_call(S, K, T, r, sigma, q)
    return _offre_discount(valeur, S, K, T, marge_pa)


def discount_mc(chemins, T, r, cap_pct, marge_pa=0.0, tau=None, recovery=0.4, spot0=S0):
    """
    Discount Certificate par Monte Carlo : meme sortie que
    discount_certificate, pour les trajectoires Bates (le skew et les sauts
    changent le prix du call vendu). Risque emetteur optionnel : en cas de
    defaut, le client recupere recovery x la valeur d'une part (spot0).
    """
    K = cap_pct * spot0
    ST = chemins[:, -1].astype(np.float64)
    flux = np.minimum(ST, K) * math.exp(-r * T)
    if tau is not None:
        flux = np.where(tau > T, flux, pv_recouvrement(tau, T, r, spot0, recovery))
    return _offre_discount(float(flux.mean()), spot0, K, T, marge_pa)


def put_down_and_in(S, K, H, T, r, sigma, q=0.0):
    """
    Put down-and-in en formule fermee (Reiner-Rubinstein).

    C'est l'option que le client d'un BRC vend a la banque : elle ne
    s'active que si le sous-jacent franchit la barriere H.

    On la veut en formule fermee et non par Monte Carlo, parce qu'on va
    en deriver les grecques par differences finies : avec du bruit de
    simulation, un gamma est inexploitable.
    """
    if T <= 1e-9:
        return max(K - S, 0.0) if S <= H else 0.0
    if S <= H:                      # barriere deja franchie : put vanille
        return bs_put(S, K, T, r, sigma, q)

    st_ = sigma * math.sqrt(T)
    mu = (r - q - 0.5 * sigma ** 2) / sigma ** 2
    phi, eta = -1.0, 1.0

    x1 = math.log(S / K) / st_ + (1 + mu) * st_
    x2 = math.log(S / H) / st_ + (1 + mu) * st_
    y1 = math.log(H ** 2 / (S * K)) / st_ + (1 + mu) * st_
    y2 = math.log(H / S) / st_ + (1 + mu) * st_

    dq, dr = math.exp(-q * T), math.exp(-r * T)
    A = phi * S * dq * norm_cdf(phi * x1) - phi * K * dr * norm_cdf(phi * (x1 - st_))
    B = phi * S * dq * norm_cdf(phi * x2) - phi * K * dr * norm_cdf(phi * (x2 - st_))
    C = (phi * S * dq * (H / S) ** (2 * (mu + 1)) * norm_cdf(eta * y1)
         - phi * K * dr * (H / S) ** (2 * mu) * norm_cdf(eta * (y1 - st_)))
    D = (phi * S * dq * (H / S) ** (2 * (mu + 1)) * norm_cdf(eta * y2)
         - phi * K * dr * (H / S) ** (2 * mu) * norm_cdf(eta * (y2 - st_)))

    # Combinaison Reiner-Rubinstein, validee contre Monte Carlo et contre
    # l'identite DI + DO = put vanille.
    #
    # Si la barriere est AU-DESSUS du strike, le cas se resout tout seul :
    # toute trajectoire finissant dans la monnaie a forcement franchi la
    # barriere en descendant. L'option vaut alors le put vanille.
    if K <= H:
        return bs_put(S, K, T, r, sigma, q)
    return max(B - C + D, 0.0)


def grecques(prix, S, T, sigma, r, **kw):
    """
    Les cinq grecques par differences finies centrees sur une formule
    fermee. Exactes a la precision numerique pres, et valables aussi bien
    pour un put vanille que pour un put a barriere.

    Conventions de marche, celles qu'un desk affiche :
      delta  variation pour +1 point de sous-jacent
      gamma  variation du delta pour +1 point de sous-jacent
      vega   variation pour +1 point de volatilite
      theta  variation pour un jour qui passe
      rho    variation pour +1 point de taux
    """
    h = max(S * 0.005, 0.01)
    p0 = prix(S=S, T=T, sigma=sigma, r=r, **kw)
    pu = prix(S=S + h, T=T, sigma=sigma, r=r, **kw)
    pd = prix(S=S - h, T=T, sigma=sigma, r=r, **kw)

    delta = (pu - pd) / (2 * h)
    gamma = (pu - 2 * p0 + pd) / (h ** 2)

    hv = 0.005
    vega = (prix(S=S, T=T, sigma=sigma + hv, r=r, **kw)
            - prix(S=S, T=T, sigma=max(sigma - hv, 1e-4), r=r, **kw)) / (2 * hv) / 100.0

    dt = 1.0 / 365.0
    theta = (prix(S=S, T=max(T - dt, 1e-9), sigma=sigma, r=r, **kw) - p0)

    hr = 0.0005
    rho = (prix(S=S, T=T, sigma=sigma, r=r + hr, **kw)
           - prix(S=S, T=T, sigma=sigma, r=max(r - hr, 0.0), **kw)) / (2 * hr) / 100.0

    return {"prix": p0, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho}


# ---------------------------------------------------------------------
# Grecques de l'autocall, du point de vue du desk
# ---------------------------------------------------------------------

def grecques_autocall(T, r, q, sigma, n_obs, trigger, bar_cp, bar_cap, memoire, coupon,
                      nominal=1000.0, n_actifs=1, correlation=0.0, correlation_stress=None,
                      seuil_stress=0.10, bates=None, decrement=0.0, n_paths=10000,
                      seed=42, pas_par_an=52, lissage=0.03, choc_spot=0.05, spot0=S0):
    """
    Grecques d'un autocall par rebond et reevaluation, avec les MEMES
    tirages a chaque rebond (common random numbers). Sans ca, l'ecart entre
    deux prix Monte Carlo serait noye dans le bruit de simulation.

    Point de vue du DESK, en face du client : le desk a vendu le produit,
    ses sensibilites sont l'oppose de celles du client. En EUR sur le
    nominal, hors risque emetteur.
      delta        pour +1 point de sous-jacent (spot 100 : +1 %), mesure
                   entre -choc_spot et +choc_spot (5 % par defaut)
      gamma        variation du delta pour +1 point, sur le meme intervalle
      vega         pour +1 point de vol (Bates : vol de depart et vol long
                   terme deplacees ensemble)
      correlation  pour +1 point de correlation (worst-of ; deplace aussi
                   la correlation de stress)
      dividende    pour +1 point de rendement du dividende. Nul sur un
                   indice a decrement : il reinvestit ses dividendes, la
                   banque n'a plus ce risque. C'est la raison d'etre de ces
                   indices.
      rho          pour +1 point de taux
      theta        pour un jour qui passe, spot inchange
    Plus :
      pv_client    valeur du produit pour le client (payoff exact)
      marge_desk   nominal - pv_client : ce que le desk garde au depart
      couverture   sous-jacent a acheter pour couvrir, en % du nominal

    Comment c'est fait, et pourquoi c'est rapide :
      - spot, dividende et taux : sur un modele homogene (tout sauf le
        decrement en points), deplacer le spot multiplie les trajectoires,
        deplacer r - q les multiplie par exp(h t). Aucune nouvelle
        simulation, et exactement les memes tirages.
      - vol et correlation : nouvelle simulation avec la meme graine.
      - theta : le meme produit un jour plus tard, spot inchange, c'est le
        meme produit dont le premier pas dure un jour de moins. On
        raccourcit donc le premier pas des memes trajectoires (le reste du
        chemin ne bouge pas) et on rapproche les dates d'actualisation d'un
        jour. Decaler toutes les dates d'un pas entier retirerait toute une
        semaine de diffusion a chaque observation : le theta serait noye
        dans le bruit.
      - les sauts du payoff (rappel, coupon, barriere) sont lisses par une
        rampe de largeur `lissage` x spot0 (3 points), comme le call spread
        d'un desk. Sans ca, le gamma d'un payoff digital par Monte Carlo est
        du bruit. Avec 3 points de rampe et un choc de 5 %, l'ecart-type du
        gamma d'une graine a l'autre est divise par 5 a 7 (etude sur 10
        graines), pour un biais de prix negligeable.
    Le worst-of est simule en Black-Scholes multi-actifs, comme dans l'appli.
    """
    if n_actifs > 1 and decrement > 0:
        raise ValueError("indice a decrement non gere en worst-of")
    n_steps = nb_pas(T, pas_par_an)
    dt = T / n_steps
    temps = np.arange(n_steps + 1, dtype=np.float32) * np.float32(dt)
    homogene = decrement <= 0

    def simule(spot=spot0, dvol=0.0, dcorr=0.0, r_=r, premier_pas=1.0):
        if n_actifs > 1:
            # sans stress, correlation_stress = correlation redonne exactement simule_panier
            cs = correlation if correlation_stress is None else correlation_stress
            return _panier_stress(n_actifs, correlation + dcorr, cs + dcorr, seuil_stress, spot,
                                  T, r_, q, sigma + dvol, n_paths, pas_par_an, seed,
                                  premier_pas)[0]
        if bates is not None:
            b = bates
            return simule_bates(spot, T, r_, q, n_paths, (math.sqrt(b["v0"]) + dvol) ** 2,
                                (math.sqrt(b["theta"]) + dvol) ** 2, b["kappa"], b["xi"],
                                b["rho"], b["lam"], b["mu_j"], b["sig_j"], pas_par_an, seed,
                                decrement)[0]
        return simule_bs(spot, T, r_, q, sigma + dvol, n_paths, pas_par_an, seed, decrement)[0]

    def deforme(ch, facteur=1.0, derive=0.0):
        """Memes tirages, spot multiplie par facteur, drift r - q deplace de derive."""
        f = np.float32(facteur) * np.exp(np.float32(derive) * temps)
        return ch * f.astype(np.float32)

    def pv(ch, r_=r, recul=0.0):
        return _flux_autocall(ch, T, r_, nominal, n_obs, trigger, bar_cp, bar_cap, memoire,
                              coupon, None, 0.4, spot0, lissage, recul).mean()

    ch0 = simule()
    pv_exact = pv_autocall(ch0, T, r, nominal, n_obs, trigger, bar_cp, bar_cap, memoire,
                           coupon, spot0=spot0)
    v0 = pv(ch0)

    # spot : +/- choc_spot
    h = choc_spot
    if homogene:
        v_up, v_dn = pv(deforme(ch0, 1.0 + h)), pv(deforme(ch0, 1.0 - h))
    else:
        v_up, v_dn = pv(simule(spot0 * (1.0 + h))), pv(simule(spot0 * (1.0 - h)))
    pas_spot = h * spot0
    delta = (v_up - v_dn) / (2.0 * pas_spot)
    gamma = (v_up - 2.0 * v0 + v_dn) / pas_spot ** 2

    # vol : +/- 1 point, meme graine
    vega = (pv(simule(dvol=0.01)) - pv(simule(dvol=-0.01))) / 2.0

    # correlation : +/- 5 points, borne pour garder une matrice valide
    sens_corr = 0.0
    if n_actifs > 1:
        bas = -1.0 / (n_actifs - 1) + 0.01
        cmax = max(correlation, correlation_stress if correlation_stress is not None else correlation)
        cmin = min(correlation, correlation_stress if correlation_stress is not None else correlation)
        c_up = min(0.05, 0.99 - cmax)
        c_dn = min(0.05, cmin - bas)
        if c_up + c_dn > 1e-6:
            sens_corr = ((pv(simule(dcorr=c_up)) - pv(simule(dcorr=-c_dn)))
                         / ((c_up + c_dn) / 0.01))

    # dividende : +/- 0,25 point, ramene a 1 point
    hq = 0.0025
    if homogene:
        sens_div = (pv(deforme(ch0, derive=-hq)) - pv(deforme(ch0, derive=hq))) / (2 * hq / 0.01)
    else:
        sens_div = 0.0          # indice a decrement : dividendes reinvestis

    # taux : +/- 0,25 point, ramene a 1 point (drift ET actualisation)
    hr = 0.0025
    if homogene:
        v_r_up, v_r_dn = pv(deforme(ch0, derive=hr), r + hr), pv(deforme(ch0, derive=-hr), r - hr)
    else:
        v_r_up, v_r_dn = pv(simule(r_=r + hr), r + hr), pv(simule(r_=r - hr), r - hr)
    sens_r = (v_r_up - v_r_dn) / (2 * hr / 0.01)

    # theta : un jour qui passe = premier pas raccourci d'un jour
    jour = 1.0 / 365.0
    f = max(1.0 - jour / dt, 0.0)
    if n_actifs > 1:
        ch_t = simule(premier_pas=f)
    else:
        # premier rendement x1 remplace par m f + sqrt(f) (x1 - m), m sa moyenne :
        # exact sous Black-Scholes, a la troncature des sauts pres sous Bates
        if bates is not None:
            kbar = math.exp(bates["mu_j"] + 0.5 * bates["sig_j"] ** 2) - 1.0
            m = (r - q - bates["lam"] * kbar - 0.5 * bates["v0"]) * dt
        else:
            m = (r - q - 0.5 * sigma ** 2) * dt
        m -= decrement * dt / spot0
        x1 = np.log(np.maximum(ch0[:, 1].astype(np.float64), 1e-9) / spot0)
        g = np.exp(m * f + math.sqrt(f) * (x1 - m) - x1).astype(np.float32)
        ch_t = ch0 * g[:, None]
        ch_t[:, 0] = spot0
    theta = pv(ch_t, recul=jour) - v0

    client = {"delta": delta, "gamma": gamma, "vega": vega, "correlation": sens_corr,
              "dividende": sens_div, "rho": sens_r, "theta": theta}
    desk = {k: -float(v) for k, v in client.items()}
    desk.update({"pv_client": float(pv_exact),
                 "marge_desk": float(nominal - pv_exact),
                 "couverture": float(delta * spot0 / nominal)})
    return desk
