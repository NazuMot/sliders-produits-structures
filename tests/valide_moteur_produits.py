"""
Validation chiffree des ajouts au moteur (moteur.py), sans interface.

    python tests/valide_moteur_produits.py

0. Compatibilite : avec leurs reglages par defaut, les fonctions modifiees
   redonnent EXACTEMENT (au bit pres) les resultats de la version
   precedente, copiee plus bas.
1. Autocall step-down          4. Correlation de stress (worst-of)
2. Indice a decrement          5. Wrong-way risk
3. Discount Certificate        6. Grecques de l'autocall

Chaque point a des cas limites qui doivent retrouver un resultat connu.
Sort avec le code 1 au premier probleme.
"""

import math
import os
import sys
import time
import tracemalloc

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import moteur as M                                       # noqa: E402


def coupon_autocall_maturite(*args, **kw):
    """Ce script isole la mecanique des produits : marge comptee sur la maturite
    maximale, comme quand il a ete ecrit. La marge sur la duree de vie attendue
    (defaut du moteur depuis le 18/10/2026) a son propre controle a la fin."""
    return M.coupon_autocall(*args, marge_sur="maturite", **kw)
from moteur import S0, nb_pas, idx_obs, proba_non_touche, pv_recouvrement  # noqa: E402

echecs = []


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def titre(t):
    print("\n" + t + "\n" + "-" * len(t))


# =====================================================================
# Copie figee de la version precedente (avant les ajouts)
# =====================================================================

def ref_simule_bs(spot, T, r, q, sigma, n_paths, pas_par_an=52, seed=42):
    n_steps = nb_pas(T, pas_par_an)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = max(n_paths // 2, 1)
    z = rng.standard_normal((half, n_steps), dtype=np.float32)
    z = np.vstack([z, -z])
    incr = np.float32((r - q - 0.5 * sigma ** 2) * dt) + np.float32(sigma * math.sqrt(dt)) * z
    ch = spot * np.exp(np.cumsum(incr, axis=1, dtype=np.float32))
    return np.hstack([np.full((ch.shape[0], 1), spot, dtype=np.float32), ch]), sigma


def ref_simule_bates(spot, T, r, q, n_paths, v0, theta, kappa, xi, rho,
                     lam, mu_j, sig_j, pas_par_an=52, seed=42):
    n_steps = nb_pas(T, pas_par_an)
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    half = max(n_paths // 2, 1)
    n = 2 * half
    z1 = rng.standard_normal((half, n_steps), dtype=np.float32)
    z1 = np.vstack([z1, -z1])
    z2 = rng.standard_normal((half, n_steps), dtype=np.float32)
    z2 = np.vstack([z2, -z2])
    w2 = np.float32(rho) * z1 + np.float32(math.sqrt(max(1 - rho ** 2, 0.0))) * z2
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


def ref_simule_panier(n_actifs, correlation, spot, T, r, q, sigma, n_paths,
                      pas_par_an=52, seed=42):
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
    worst = perf.min(axis=0)
    worst = spot * worst
    return np.hstack([np.full((worst.shape[0], 1), spot, dtype=np.float32), worst]), sigma


def ref_tire_defaut(n_paths, T, spread_bps, recovery, seed=123):
    if spread_bps <= 0:
        return np.full(n_paths, np.inf)
    lam = (spread_bps / 10000.0) / max(1.0 - recovery, 0.01)
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0 / lam, size=n_paths)


def ref_pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, coupon_pa,
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


def ref_coupon_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, marge_pa,
                   sigma, tau=None, recovery=0.4, spot0=S0):
    base = ref_pv_brc(chemins, T, r, nominal, bar_pct, strike_pct, continu, 0.0,
                      sigma, tau, recovery, spot0)
    if tau is None:
        unite = math.exp(-r * T) * nominal * T
    else:
        unite = (math.exp(-r * T) * nominal * T * (tau > T)).mean()
    cible = nominal * (1.0 - marge_pa * T)
    return max((cible - base) / unite, 0.0) if unite > 1e-9 else 0.0


def ref_pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                    memoire, coupon, tau=None, recovery=0.4, spot0=S0):
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
        pv = np.where(vivant & (tau <= T),
                      pv + pv_recouvrement(tau, T, r, nominal, recovery), pv)
    return pv.mean()


def ref_coupon_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                        memoire, marge_pa, tau=None, recovery=0.4, spot0=S0):
    a = ref_pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                        memoire, 0.0, tau, recovery, spot0)
    b = ref_pv_autocall(chemins, T, r, nominal, n_obs, trigger, bar_cp, bar_cap,
                        memoire, 0.01, tau, recovery, spot0)
    pente = (b - a) / 0.01
    cible = nominal * (1.0 - marge_pa * T)
    return max((cible - a) / pente, 0.0) if pente > 1e-9 else 0.0


def ref_stats_autocall(chemins, T, n_obs, trigger, bar_cap, spot0=S0):
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


# =====================================================================
# Marche de reference (reglages par defaut de la barre laterale)
# =====================================================================

r, q, vol, nominal, marge = 0.03, 0.025, 0.22, 1000.0, 0.008
BATES = dict(v0=0.22 ** 2, theta=0.24 ** 2, kappa=2.0, xi=0.5, rho=-0.7,
             lam=0.3, mu_j=-0.10, sig_j=0.15)
B_ARGS = (BATES["v0"], BATES["theta"], BATES["kappa"], BATES["xi"], BATES["rho"],
          BATES["lam"], BATES["mu_j"], BATES["sig_j"])
T_AC, N_OBS = 3.0, 12                       # autocall 3 ans, trimestriel
V_AC = M.vol_au_strike(0.60, vol, 2.0)      # vol BS au strike de la barriere capital


def pct(x, d=2):
    return f"{x * 100:.{d}f} %"


# =====================================================================
titre("0. Compatibilite : réglages par défaut = version precedente, au bit pres")
# =====================================================================

for n in (10000, 9999):
    a, b = ref_simule_bs(S0, T_AC, r, q, vol, n), M.simule_bs(S0, T_AC, r, q, vol, n)
    controle(f"simule_bs ({n} trajectoires)", np.array_equal(a[0], b[0]) and a[1] == b[1])

a = ref_simule_bates(S0, T_AC, r, q, 10000, *B_ARGS)
b = M.simule_bates(S0, T_AC, r, q, 10000, *B_ARGS)
controle("simule_bates", np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

for na, c in ((2, 0.3), (3, 0.6), (4, 0.9)):
    a = ref_simule_panier(na, c, S0, T_AC, r, q, vol, 10000)
    b = M.simule_panier(na, c, S0, T_AC, r, q, vol, 10000)
    controle(f"simule_panier {na} actifs, corrélation {c}", np.array_equal(a[0], b[0]))

for sp in (0, 150, 400):
    controle(f"tire_defaut {sp} bp",
             np.array_equal(ref_tire_defaut(10000, T_AC, sp, 0.4), M.tire_defaut(10000, T_AC, sp, 0.4)))

CH_AC = M.simule_bs(S0, T_AC, r, q, V_AC, 20000)[0]
CH_WO = M.simule_panier(3, 0.6, S0, T_AC, r, q, V_AC, 20000)[0]
TAU_AC = M.tire_defaut(20000, T_AC, 150, 0.4)
ok = True
for ch in (CH_AC, CH_WO):
    for tau in (None, TAU_AC):
        for mem in (True, False):
            for trig in (1.0, 0.9):
                a = (ref_pv_autocall(ch, T_AC, r, nominal, N_OBS, trig, 0.7, 0.6, mem, 0.02, tau, 0.4),
                     ref_coupon_autocall(ch, T_AC, r, nominal, N_OBS, trig, 0.7, 0.6, mem, marge, tau, 0.4),
                     ref_stats_autocall(ch, T_AC, N_OBS, trig, 0.6))
                b = (M.pv_autocall(ch, T_AC, r, nominal, N_OBS, trig, 0.7, 0.6, mem, 0.02, tau, 0.4),
                     coupon_autocall_maturite(ch, T_AC, r, nominal, N_OBS, trig, 0.7, 0.6, mem, marge, tau, 0.4),
                     M.stats_autocall(ch, T_AC, N_OBS, trig, 0.6))
                ok &= a == b
controle("pv_autocall, coupon_autocall, stats_autocall (32 combinaisons)", ok)

ch1, sg1 = M.simule_bs(S0, 1.0, r, q, vol, 20000)
chb, sgb = M.simule_bates(S0, 1.0, r, q, 20000, *B_ARGS)
tau1 = M.tire_defaut(20000, 1.0, 150, 0.4)
ok = True
for ch, sg in ((ch1, sg1), (chb, sgb)):
    for tau in (None, tau1):
        for cont in (True, False):
            ok &= (ref_pv_brc(ch, 1.0, r, nominal, 0.7, 1.0, cont, 0.05, sg, tau, 0.4)
                   == M.pv_brc(ch, 1.0, r, nominal, 0.7, 1.0, cont, 0.05, sg, tau, 0.4))
            ok &= (ref_coupon_brc(ch, 1.0, r, nominal, 0.7, 1.0, cont, marge, sg, tau, 0.4)
                   == M.coupon_brc(ch, 1.0, r, nominal, 0.7, 1.0, cont, marge, sg, tau, 0.4))
controle("pv_brc, coupon_brc (16 combinaisons, BS et Bates)", ok)


# =====================================================================
titre("1. Autocall step-down")
# =====================================================================

def autocall(ch, trig, **kw):
    c = coupon_autocall_maturite(ch, T_AC, r, nominal, N_OBS, trig, 0.7, 0.6, True, marge, **kw)
    return (c,) + tuple(M.stats_autocall(ch, T_AC, N_OBS, trig, 0.6))


classique = autocall(CH_AC, 1.0)
controle("Suite constante = autocall classique (coupon et stats identiques)",
         autocall(CH_AC, [1.0] * N_OBS) == classique and autocall(CH_AC, [1.0] * (N_OBS - 1)) == classique)
controle("Suite constante = classique avec risque émetteur",
         autocall(CH_AC, [1.0] * N_OBS, tau=TAU_AC) == autocall(CH_AC, 1.0, tau=TAU_AC))
controle("Worst-of : suite constante = classique", autocall(CH_WO, [1.0] * N_OBS) == autocall(CH_WO, 1.0))

jamais = autocall(CH_AC, [math.inf] * N_OBS)
controle("Seuils infinis : aucun rappel, durée = maturité", jamais[2] == 0.0 and jamais[1] == T_AC,
         f"rappel {pct(jamais[2])}, durée {jamais[1]:.2f} ans")

sd = M.trigger_degressif(N_OBS, 1.0, 0.05, obs_par_palier=4)
controle("trigger_degressif : 100 % puis -5 % par an",
         sd == [1.0] * 4 + [0.95] * 4 + [0.9] * 4, str([round(x, 2) for x in sd]))
deg = autocall(CH_AC, sd)
controle("Step-down : plus de rappels, durée plus courte, moins de pertes, coupon plus faible",
         deg[2] > classique[2] and deg[1] < classique[1] and deg[3] < classique[3]
         and deg[0] < classique[0],
         f"coupon {pct(classique[0] * 4)} -> {pct(deg[0] * 4)} p.a., rappel {pct(classique[2], 1)} -> "
         f"{pct(deg[2], 1)}, durée {classique[1]:.2f} -> {deg[1]:.2f} ans, perte "
         f"{pct(classique[3], 1)} -> {pct(deg[3], 1)}")

try:
    M.stats_autocall(CH_AC, T_AC, N_OBS, [1.0] * 5, 0.6)
    controle("Suite de mauvaise longueur refusee", False)
except ValueError:
    controle("Suite de mauvaise longueur refusee", True)


# =====================================================================
titre("2. Indice a décrément")
# =====================================================================

controle("decrement_base100 : 50 points sur 5 000 = 1, sur base 1 000 = 5",
         M.decrement_base100(50, 5000) == 1.0 and M.decrement_base100(50, 1000) == 5.0)
controle("Poids du décrément : 1 point pèse 1 % à 100, 2 % à 50",
         abs(M.dividende_equivalent(100.0, 1.0) - 0.01) < 1e-15
         and abs(M.dividende_equivalent(50.0, 1.0) - 0.02) < 1e-15)

a = M.simule_bs(S0, T_AC, r, 0.0, vol, 10000)[0]
b = M.simule_bs(S0, T_AC, r, 0.0, vol, 10000, decrement=1e-9)[0]
controle("Décrément ~0 = sous-jacent sans dividende (BS)",
         np.max(np.abs(b / a - 1)) < 1e-5, f"écart max {np.max(np.abs(b / a - 1)):.1e}")
a = M.simule_bates(S0, T_AC, r, 0.0, 10000, *B_ARGS)[0]
b = M.simule_bates(S0, T_AC, r, 0.0, 10000, *B_ARGS, decrement=1e-9)[0]
controle("Décrément ~0 = sous-jacent sans dividende (Bates)",
         np.max(np.abs(b / a - 1)) < 1e-4, f"écart max {np.max(np.abs(b / a - 1)):.1e}")
controle("Décrément 0 : exactement le chemin q = 0",
         np.array_equal(M.simule_bs(S0, T_AC, r, 0.0, vol, 10000, decrement=0.0)[0],
                        M.simule_bs(S0, T_AC, r, 0.0, vol, 10000)[0]))

# la forme fermee de la simulation = la regle de l'indice appliquee pas a pas
D = 2.5
ch, _ = M.simule_bs(S0, T_AC, r, 0.0, vol, 200, decrement=D)
tr, _ = M.simule_bs(S0, T_AC, r, 0.0, vol, 200)          # meme tirages, sans decrement
dt = T_AC / (ch.shape[1] - 1)
S = np.full(200, S0)
for k in range(1, ch.shape[1]):
    S = S * tr[:, k] / tr[:, k - 1] - D * dt
controle("Forme fermée = règle de l'indice pas à pas (S <- S x rendement - D dt)",
         np.max(np.abs(ch[:, -1] / S - 1)) < 1e-4, f"écart max {np.max(np.abs(ch[:, -1] / S - 1)):.1e}")

for nom, ch in (("BS", M.simule_bs(S0, T_AC, r, 0.0, vol, 40000, decrement=D)[0]),
                ("Bates", M.simule_bates(S0, T_AC, r, 0.0, 40000, *B_ARGS, decrement=D)[0])):
    f_mc, f_th = ch[:, -1].astype(np.float64).mean(), M.forward_decrement(S0, T_AC, r, D)
    controle(f"Forward Monte Carlo = forme fermée à 0,5 % ({nom}, D = {D} points)",
             abs(f_mc / f_th - 1) < 0.005, f"{f_mc:.2f} contre {f_th:.2f}")

for nom, bt, sig in (("BS", None, V_AC), ("Bates", BATES, vol)):
    for D in (2.5, 5.0):
        t0 = time.time()
        res = M.compare_decrement(T_AC, r, q, sig, N_OBS, 1.0, 0.7, 0.6, True, marge, D, bates=bt)
        el = time.time() - t0
        c, p, s_ = res["classique"], res["decrement_pct"], res["decrement_points"]
        print(f"      {nom:5} D = {D} pts : coupon p.a. classique {pct(c['coupon'] * 4)}, "
              f"décrément % {pct(p['coupon'] * 4)}, décrément points {pct(s_['coupon'] * 4)} | "
              f"proba de perte {pct(c['p_perte'], 1)}, {pct(p['p_perte'], 1)}, {pct(s_['p_perte'], 1)}"
              f"  ({el:.2f} s)")
        if D == q * 100:
            controle(f"{nom} : décrément en % = dividende -> même coupon que l'indice classique",
                     p["coupon"] == c["coupon"] and p["p_perte"] == c["p_perte"])
        controle(f"{nom} D = {D} : effet points -> coupon ET proba de perte plus élevés",
                 s_["coupon"] > p["coupon"] and s_["p_perte"] > p["p_perte"])
    controle(f"{nom} : décrément 5 % > dividende 2,5 % -> coupon plus élevé que le classique",
             res["decrement_points"]["coupon"] > res["classique"]["coupon"])

res = M.compare_decrement(T_AC, r, q, V_AC, N_OBS, 1.0, 0.7, 0.6, True, marge, 2.5, decote_div=0.003)
controle("Décote de dividende du desk (0,3 %) : coupon classique plus faible",
         res["classique"]["coupon"] < res["decrement_pct"]["coupon"],
         f"{pct(res['classique']['coupon'] * 4)} contre {pct(res['decrement_pct']['coupon'] * 4)} p.a.")


# =====================================================================
titre("3. Discount Certificate")
# =====================================================================

for T, K in ((1.0, 1.0), (1.0, 0.9), (2.0, 1.0), (1.0, 1.1), (3.0, 0.8)):
    sig = M.vol_au_strike(K, vol, 2.0)
    fe = M.discount_certificate(S0, K * S0, T, r, sig, q, marge)
    mc = M.discount_mc(M.simule_bs(S0, T, r, q, sig, 20000)[0], T, r, K, marge)
    ecart = mc["valeur"] / fe["valeur"] - 1
    controle(f"Monte Carlo = formule fermée à 1 % (T = {T:g}, cap {K * 100:.0f} %)",
             abs(ecart) < 0.01, f"décote {pct(fe['decote'])}, rendement max {pct(fe['rendement_max'])}, "
                                f"écart MC {ecart * 100:+.3f} %")

fe = M.discount_certificate(S0, 1e7, 2.0, r, vol, q)
controle("Cap infini : valeur = S e^{-qT} (la décote = les dividendes abandonnés)",
         abs(fe["valeur"] - S0 * math.exp(-q * 2.0)) < 1e-9)
fe = M.discount_certificate(S0, 1e-3, 2.0, r, vol, q)
controle("Cap nul : valeur = K e^{-rT} (zéro-coupon)", abs(fe["valeur"] - 1e-3 * math.exp(-r * 2.0)) < 1e-12)
fe = M.discount_certificate(S0, 100.0, 1.0, r, vol, q, 0.01)
controle("Marge banque : prix = valeur / (1 - marge x T)",
         abs(fe["prix"] - fe["valeur"] / 0.99) < 1e-12 and fe["point_mort"] == fe["prix"])
d_bas = M.discount_certificate(S0, 100.0, 1.0, r, 0.15, q)["decote"]
d_haut = M.discount_certificate(S0, 100.0, 1.0, r, 0.35, q)["decote"]
controle("Vol plus haute -> call vendu plus cher -> décote plus forte", d_haut > d_bas,
         f"{pct(d_bas)} à 15 % de vol, {pct(d_haut)} à 35 %")
mb = M.discount_mc(M.simule_bates(S0, 1.0, r, q, 20000, *B_ARGS)[0], 1.0, r, 1.0, marge)
controle("Version Bates calculée", 0 < mb["decote"] < 0.5, f"décote {pct(mb['decote'])} sous Bates, cap 100 %")


# =====================================================================
titre("4. Corrélation de stress (worst-of)")
# =====================================================================

a = M.simule_panier(3, 0.6, S0, T_AC, r, q, V_AC, 20000)[0]
b = M.simule_panier(3, 0.6, S0, T_AC, r, q, V_AC, 20000, correlation_stress=0.6)[0]
controle("Corrélation de stress = corrélation normale -> chemins identiques", np.array_equal(a, b))
b, d = M._panier_stress(3, 0.6, 0.9, 0.99, S0, T_AC, r, q, V_AC, 20000)
controle("Seuil jamais atteint -> chemins identiques", np.array_equal(a, b) and d["part_stress"] == 0.0)

ws, d = M._panier_stress(3, 0.6, 0.9, 0.10, S0, T_AC, r, q, V_AC, 40000)
fwd = math.exp((r - q) * T_AC)
controle("Chaque actif garde son forward risque-neutre (à 1 %)",
         np.all(np.abs(d["perf_finale"] / fwd - 1) < 0.01),
         f"{np.round(d['perf_finale'], 4)} contre {fwd:.4f}")

lignes = {}
for nom, cs in (("constante 60 %", None), ("60 %, 90 % en stress", 0.9)):
    t0 = time.time()
    ch = M.simule_panier(3, 0.6, S0, T_AC, r, q, V_AC, 20000, correlation_stress=cs, seuil_stress=0.10)[0]
    el = time.time() - t0
    lignes[nom] = autocall(ch, 1.0) + (el,)
ch = M.simule_panier(3, 0.9, S0, T_AC, r, q, V_AC, 20000)[0]
lignes["constante 90 %"] = autocall(ch, 1.0) + (0,)
for nom, (c, du, pr, pp, el) in lignes.items():
    print(f"      worst-of 3, corrélation {nom:22}: coupon {pct(c * 4)} p.a., perte {pct(pp, 1)}, "
          f"durée {du:.2f} ans" + (f"  ({el:.2f} s)" if el else ""))
print(f"      part du temps en stress (seuil -10 % sous le plus haut) : {pct(d['part_stress'], 0)}")
c6, cs_, c9 = lignes["constante 60 %"], lignes["60 %, 90 % en stress"], lignes["constante 90 %"]
controle("Stress : coupon et proba de perte entre corrélation constante 60 % et 90 %",
         c9[0] < cs_[0] < c6[0] and c9[3] < cs_[3] < c6[3])


# =====================================================================
titre("5. Wrong-way risk")
# =====================================================================

T1, SP, REC = 1.0, 150, 0.4
lam = SP / 1e4 / (1 - REC)
chb, sgb = M.simule_bates(S0, T1, r, 0.03, 20000, 0.35 ** 2, 0.35 ** 2, 2.0, 0.6, -0.7, 0.5, -0.12, 0.15)
t0 = M.tire_defaut(20000, T1, SP, REC)
controle("Corrélation 0 avec chemins fournis = tirages actuels",
         np.array_equal(M.tire_defaut(20000, T1, SP, REC, correlation=0.0, chemins=chb), t0))
te = M.tire_defaut(20000, T1, SP, REC, correlation=1e-6, chemins=chb)
d_ = t0 <= T1
controle("Corrélation 1e-6 : mêmes défauts, mêmes dates à 1e-4 pres",
         np.array_equal(te <= T1, d_) and np.max(np.abs(te[d_] / t0[d_] - 1)) < 1e-4)
p_th = 1 - math.exp(-lam * T1)
se = math.sqrt(p_th * (1 - p_th) / 20000)
for rho in (0.3, 0.7, -0.5):
    tw = M.tire_defaut(20000, T1, SP, REC, correlation=rho, chemins=chb)
    p = (tw <= T1).mean()
    controle(f"rho {rho:+.1f} : même probabilité de défaut (à 3 écarts-types)", abs(p - p_th) < 3 * se,
             f"{pct(p)} contre {pct(p_th)}")
tw = M.tire_defaut(20000, T1, SP, REC, correlation=0.7, chemins=chb)
fin = chb[:, -1]
controle("rho > 0 : la banque fait défaut quand le sous-jacent s'effondre",
         fin[tw <= T1].mean() < 0.85 * fin.mean(),
         f"niveau final moyen {fin[tw <= T1].mean():.1f} en cas de défaut, {fin.mean():.1f} en moyenne")

controle("créance 'valeur' sans risque émetteur = créance nominale",
         M.coupon_brc(chb, T1, r, nominal, 0.7, 1.0, True, marge, sgb, creance="valeur")
         == M.coupon_brc(chb, T1, r, nominal, 0.7, 1.0, True, marge, sgb))

print("      BRC 1 an barrière 70 % sur une banque (Bates, vol 35 %), émetteur 150 bp, recouvrement 40 %")
prof = {}
for creance in ("nominal", "valeur"):
    for rho in (0.0, 0.6):
        tw = M.tire_defaut(20000, T1, SP, REC, correlation=rho, chemins=chb)
        c = M.coupon_brc(chb, T1, r, nominal, 0.7, 1.0, True, marge, sgb, tw, REC, creance=creance)
        p5 = M.profil_brc(chb, T1, nominal, 0.7, 1.0, True, c, sgb, tw, REC, creance=creance)
        p1 = M.profil_brc(chb, T1, nominal, 0.7, 1.0, True, c, sgb, tw, REC, creance=creance,
                          niveau_es=0.01)
        prof[creance, rho] = (c, p5, p1)
        print(f"      créance {creance:7} rho {rho:.1f} : coupon {pct(c)}, défaut {pct(p5['p_defaut'])}, "
              f"krach ET défaut {pct(p5['p_double'])}, perte moyenne pire 5 % {pct(p5['es'], 1)}, "
              f"pire 1 % {pct(p1['es'], 1)}")
(c0, a5, a1), (c6, b5, b1) = prof["valeur", 0.0], prof["valeur", 0.6]
controle("Wrong-way : krach et défaut ensemble beaucoup plus probables",
         b5["p_double"] > 1.8 * a5["p_double"], f"{pct(a5['p_double'])} -> {pct(b5['p_double'])}")
controle("Wrong-way : le client perd plus dans les pires scénarios (créance = valeur)",
         b1["es"] > a1["es"] + 0.05 and b5["es"] > a5["es"],
         f"pire 1 % : {pct(a1['es'], 1)} -> {pct(b1['es'], 1)} ; pire 5 % : {pct(a5['es'], 1)} -> {pct(b5['es'], 1)}")
controle("Wrong-way : coupon équitable à peine change (l'effet est dans la queue, pas dans la moyenne)",
         abs(c6 - c0) < 0.01, f"{pct(c0)} -> {pct(c6)}")


# =====================================================================
titre("6. Grecques de l'autocall (point de vue du desk, EUR pour 1 000 de nominal)")
# =====================================================================

CPN = coupon_autocall_maturite(CH_AC, T_AC, r, nominal, N_OBS, 1.0, 0.7, 0.6, True, marge)
CPN_WO = coupon_autocall_maturite(CH_WO, T_AC, r, nominal, N_OBS, 1.0, 0.7, 0.6, True, marge)
cas = {"BS mono": dict(),
       "worst-of 3, rho 60 %": dict(n_actifs=3, correlation=0.6),
       "worst-of 3, stress": dict(n_actifs=3, correlation=0.6, correlation_stress=0.9),
       "Bates mono": dict(bates=BATES),
       "décrément 2,5 pts": dict(decrement=2.5)}
G = {}
for nom, kw in cas.items():
    cp = CPN_WO if "worst" in nom else CPN
    qq = 0.0 if 'décrément' in nom else q
    t0 = time.time()
    g = M.grecques_autocall(T_AC, r, qq, V_AC, N_OBS, 1.0, 0.7, 0.6, True, cp, nominal, **kw)
    el = time.time() - t0
    G[nom] = g
    print(f"      {nom:22} delta {g['delta']:+6.2f}  gamma {g['gamma']:+6.3f}  vega {g['vega']:+6.2f}  "
          f"correl {g['correlation']:+6.2f}  divid {g['dividende']:+6.2f}  rho {g['rho']:+6.2f}  "
          f"theta {g['theta']:+6.3f}/j  couverture {pct(g['couverture'], 0)}  ({el:.2f} s)")
    controle(f"{nom} : moins de 2 s avec 10 000 trajectoires", el < 2.0, f"{el:.2f} s")

for nom, g in G.items():
    ok = g["delta"] < 0 and g["vega"] > 0 and g["theta"] < 0
    if 'décrément' not in nom:
        ok &= g["dividende"] > 0
    if "worst" in nom:
        ok &= g["correlation"] < 0
    controle(f"{nom} : signes du desk (delta < 0, vega > 0, theta < 0, dividende > 0, corrélation < 0)", ok)
controle("Indice a décrément : aucune sensibilité au dividende", G["décrément 2,5 pts"]["dividende"] == 0.0)

# delta coherent avec la variation de prix exacte (payoff non lisse, memes tirages)
ch0 = M.simule_bs(S0, T_AC, r, q, V_AC, 10000)[0]
pv = lambda ch: M.pv_autocall(ch, T_AC, r, nominal, N_OBS, 1.0, 0.7, 0.6, True, CPN)
for choc in (0.05, 0.10):
    dp = -(pv(ch0 * np.float32(1 + choc)) - pv(ch0 * np.float32(1 - choc))) / (2 * choc * S0)
    g = G["BS mono"]["delta"]
    controle(f"Delta = variation du prix exact sur +/- {choc * 100:.0f} % (à 10 %)", abs(dp / g - 1) < 0.10,
             f"{g:.3f} contre {dp:.3f} EUR par point")
up = pv(ch0 * np.float32(1.05))
up_sim = pv(M.simule_bs(S0 * 1.05, T_AC, r, q, V_AC, 10000)[0])
controle("Raccourci : spot x 1,05 sur les mêmes chemins = nouvelle simulation", abs(up / up_sim - 1) < 1e-5,
         f"{up:.4f} contre {up_sim:.4f}")

# theta contre equation de Black-Scholes : theta = rV - (r-q) S delta - 1/2 sigma^2 S^2 gamma
th, th_eds = [], []
for s in range(5):
    g = M.grecques_autocall(T_AC, r, q, V_AC, N_OBS, 1.0, 0.7, 0.6, True, CPN, nominal, seed=s)
    th.append(g["theta"])
    v, dl, gm = g["pv_client"], -g["delta"], -g["gamma"]          # retour cote client
    th_eds.append(-(r * v - (r - q) * S0 * dl - 0.5 * V_AC ** 2 * S0 ** 2 * gm) / 365.0)
controle("Theta = equation de Black-Scholes à 20 % (moyenne de 5 graines)",
         abs(np.mean(th) / np.mean(th_eds) - 1) < 0.20, f"{np.mean(th):.3f} contre {np.mean(th_eds):.3f} EUR/jour")

for nom, kw in (("BS mono", dict()), ("worst-of 3", dict(n_actifs=3, correlation=0.6))):
    cp = CPN_WO if kw else CPN
    gg = [M.grecques_autocall(T_AC, r, q, V_AC, N_OBS, 1.0, 0.7, 0.6, True, cp, nominal, seed=s, **kw)
          for s in range(8)]
    out = {k: (np.mean([x[k] for x in gg]), np.std([x[k] for x in gg]))
           for k in ("delta", "gamma", "vega", "theta")}
    controle(f"{nom} : delta stable d'une graine à l'autre (écart-type < 10 %)",
             out["delta"][1] < 0.10 * abs(out["delta"][0]),
             ", ".join(f"{k} {m:+.3f} +/- {s:.3f}" for k, (m, s) in out.items()))

# cas le plus lourd : temps et memoire de pointe du calcul lui-meme (numpy declare
# ses allocations a tracemalloc)
for nom, kw in (("worst-of 4, 6 ans", dict(n_actifs=4, correlation=0.5)),
                ("Bates, 6 ans", dict(bates=BATES))):
    tracemalloc.start()
    t0 = time.time()
    M.grecques_autocall(6.0, r, q, V_AC, 24, 1.0, 0.7, 0.6, True, CPN_WO, nominal, **kw)
    el = time.time() - t0
    pic = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    controle(f"Grecques {nom}, 10 000 trajectoires : moins de 2 s et de 300 Mo", el < 2.0 and pic < 300,
             f"{el:.2f} s, pointe {pic:.0f} Mo")

# =====================================================================
titre("7. Marge sur la durée de vie et volatilité locale (18/10/2026)")
# =====================================================================

for trig in (0.8, 1.0):
    vie = M.coupon_autocall(CH_AC, T_AC, r, nominal, N_OBS, trig, 0.7, 0.6, True, marge)
    mat = coupon_autocall_maturite(CH_AC, T_AC, r, nominal, N_OBS, trig, 0.7, 0.6, True, marge)
    duree = M.stats_autocall(CH_AC, T_AC, N_OBS, trig, 0.6)[0]
    controle(f"Marge sur la vie attendue, trigger {trig:.0%} : coupon positif et >= ancien calcul",
             vie > 0 and vie >= mat - 1e-12,
             f"{pct(mat * 4)} -> {pct(vie * 4)} p.a., vie moyenne {duree:.2f} an sur {T_AC:.0f}")

ch0 = M.simule_lv(100.0, 3.0, r, q, 0.22, 0.0, 10000)
ch1 = M.simule_bs(100.0, 3.0, r, q, 0.22, 10000)
controle("Vol locale avec skew 0 = Black-Scholes, au bit pres",
         np.array_equal(ch0[0], ch1[0]) and ch0[1] == ch1[1])

ch, sg = M.simule_lv(100.0, 3.0, r, q, 0.22, 2.0, 10000)
ks = [0.8, 0.9, 1.0]
vols = M.vol_implicite_bates(ch, 3.0, r, q, ks)
cibles = [M.vol_au_strike(k, 0.22, 2.0) for k in ks]
controle("Vol locale, skew 2 : elle redonne le skew affiché à 1 pt de vol pres (80/90/100 %)",
         all(abs(v - c) < 0.01 for v, c in zip(vols, cibles)),
         " / ".join(f"{v*100:.1f} vs {c*100:.0f}" for v, c in zip(vols, cibles)))

for skew in (2.0, 4.0, 6.0):
    ch, _ = M.simule_lv(100.0, 3.0, r, q, 0.22, skew, 10000)
    cps = [M.coupon_autocall(ch, 3.0, r, nominal, 12, 1.0, 0.7, b, True, marge)
           for b in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    controle(f"Vol locale, skew {skew:.0f} : le coupon monte avec la barrière de capital",
             all(cps[i] <= cps[i + 1] + 1e-12 for i in range(len(cps) - 1)),
             " ".join(f"{c * 400:.1f}" for c in cps))

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
