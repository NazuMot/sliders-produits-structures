"""
Validation chiffree des donnees reelles, du backtest et du calibrage.

    python tests/valide_donnees.py

1. Historique Euro Stoxx 50 : integrite, valeurs officielles connues, krachs.
2. Photo d'options Eurex : coherence (parite, monotonie, vols).
3. Formule de Bates : cas limites a solution connue, accord avec le Monte
   Carlo de moteur.simule_bates.
4. Calibrage : fichier coherent avec le code, erreurs dans les seuils,
   optimiseur qui retrouve une solution connue.
5. Backtest : accord avec moteur.pv_autocall / pv_brc sur les memes
   chemins, cas fictifs calcules a la main, chiffres de reference, < 1 s.

Sort avec le code 1 au premier probleme. ~30 s.
"""

import math
import os
import sys
import time

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

import backtest as B      # noqa: E402
import calibration as C   # noqa: E402
import moteur as M        # noqa: E402

echecs = []


def controle(nom, ok, detail=""):
    print(("OK    " if ok else "ECHEC ") + nom + (f"  ({detail})" if detail else ""))
    if not ok:
        echecs.append(nom)


def jour(h, d):
    i = int(np.searchsorted(h.dates, np.datetime64(d)))
    assert h.dates[i] == np.datetime64(d), d
    return i


# =====================================================================
# 1. Historique
# =====================================================================
print("--- Historique Euro Stoxx 50")
h = B.charge_historique()
n = len(h.dates)
ecarts = np.diff(h.dates).astype(int)
controle("plus de 10 000 seances, de 1986 a 2026",
         n > 10000 and h.debut <= np.datetime64("1987-01-02") and h.fin >= np.datetime64("2026-10-01"),
         f"{n} seances, {h.debut} -> {h.fin}")
controle("dates strictement croissantes, aucun trou de plus de 7 jours",
         ecarts.min() > 0 and ecarts.max() <= 7, f"trou max {ecarts.max()} jours")
controle("valeurs positives et finies",
         np.all(h.prix > 0) and np.all(h.net_return > 0) and np.all(np.isfinite(h.prix)))
controle("record du 06/03/2000 : 5464,43 (valeur officielle STOXX)",
         abs(h.prix[jour(h, "2000-03-06")] - 5464.43) < 0.01)
controle("plus bas du 09/03/2009 : 1809,98",
         abs(h.prix[jour(h, "2009-03-09")] - 1809.98) < 0.01)
rend = np.diff(np.log(h.prix))
controle("krach du 19/10/1987 present (< -7 % sur la seance)",
         rend[jour(h, "1987-10-19") - 1] < -0.07, f"{rend[jour(h, '1987-10-19') - 1]:.1%}")
controle("12/03/2020 : pire seance de l'historique (-12 %)",
         rend.argmin() == jour(h, "2020-03-12") - 1, f"{rend.min():.1%}")
controle("aucune seance au-dela de +/- 15 % (pas de cotation aberrante)",
         np.abs(rend).max() < 0.15 and np.abs(np.diff(np.log(h.net_return))).max() < 0.15)
ratio = h.net_return / h.prix
annees = [jour(h, d) for d in ("1990-01-02", "2000-01-03", "2008-01-02", "2016-10-04", "2026-10-16")]
controle("net return / prix croit dans le temps (dividendes accumules)",
         np.all(np.diff(ratio[annees]) > 0), " ".join(f"{ratio[i]:.2f}" for i in annees))
i0, i1 = jour(h, "2000-01-03"), jour(h, "2026-10-16")
div = math.log(ratio[i1] / ratio[i0]) / ((h.dates[i1] - h.dates[i0]).astype(int) / 365.25)
controle("rendement du dividende net moyen 2000-2026 entre 2 et 4 % par an", 0.02 < div < 0.04,
         f"{div:.2%}")
i = jour(h, "2016-10-04")
saut = abs(math.log(h.net_return[i + 1] / h.net_return[i]) - math.log(h.prix[i + 1] / h.prix[i]))
controle("raccord STOXX / ETF du 04/10/2016 sans saut (< 0,5 %)", saut < 0.005, f"{saut:.2%}")
v = h.vstoxx[np.isfinite(h.vstoxx)]
controle("VSTOXX present depuis 1999, entre 8 et 100",
         np.isfinite(h.vstoxx[jour(h, "1999-01-05")]) and v.min() > 8 and v.max() < 100,
         f"{v.min():.1f} -> {v.max():.1f}")
poids = sum(os.path.getsize(os.path.join(RACINE, "data", f))
            for f in os.listdir(os.path.join(RACINE, "data")))
controle("dossier data/ sous 2 Mo", poids < 2e6, f"{poids / 1e6:.2f} Mo")


# =====================================================================
# 2. Options
# =====================================================================
print("--- Photo d'options Eurex")
o = C.charge_options()
echs = np.unique(o["echeance"])
controle("au moins 100 options, 9 echeances de 1 mois a 5 ans",
         len(o["K"]) >= 100 and len(echs) == 9 and o["T"].min() < 0.15 and o["T"].max() > 5,
         f"{len(o['K'])} options, T {o['T'].min():.2f} -> {o['T'].max():.2f}")
controle("vols implicites finies, entre 5 % et 80 %",
         np.all(np.isfinite(o["vol"])) and o["vol"].min() > 0.05 and o["vol"].max() < 0.8)
revol = C.vol_implicite_b76(o["prix"], o["F"], o["K"], o["T"], o["DF"], o["call"])
controle("la vol implicite stockee correspond au prix de reglement (< 0,01 pt de vol)",
         np.abs(revol - o["vol"]).max() * 100 < 0.01, f"{np.abs(revol - o['vol']).max() * 100:.4f}")
T_e = np.array([o["T"][o["echeance"] == e][0] for e in echs])
DF_e = np.array([o["DF"][o["echeance"] == e][0] for e in echs])
F_e = np.array([o["F"][o["echeance"] == e][0] for e in echs])
r_e = -np.log(DF_e) / T_e
q_e = r_e - np.log(F_e / o["spot"]) / T_e
controle("taux implicites entre 0 et 6 %, dividendes implicites entre -1 et 5 %",
         np.all((r_e > 0) & (r_e < 0.06)) and np.all((q_e > -0.01) & (q_e < 0.05)),
         f"taux {r_e.min():.2%}-{r_e.max():.2%}, dividendes {q_e.min():.2%}-{q_e.max():.2%}")
mono = True
for e in echs:
    m = o["echeance"] == e
    for typ, sens in ((True, -1), (False, 1)):
        s = m & (o["call"] == typ)
        k, p = o["K"][s], o["prix"][s]
        mono &= bool(np.all(sens * np.diff(p[np.argsort(k)]) > 0))
controle("pas d'arbitrage statique : puts croissants et calls decroissants avec le strike", mono)
var_atm = []
for e in echs:
    m = o["echeance"] == e
    j = np.argmin(np.abs(o["strike_pct"][m] - 1.0))
    var_atm.append(o["vol"][m][j] ** 2 * o["T"][m][j])
controle("variance totale ATM croissante avec la maturite (pas d'arbitrage calendaire)",
         np.all(np.diff(var_atm) > 0))


# =====================================================================
# 3. Formule de Bates
# =====================================================================
print("--- Formule de Bates")
K = np.array([50., 70, 90, 100, 110, 130, 160])
pire = 0.0
for T in (0.05, 1.0, 5.0):
    for vol in (0.1, 0.3, 0.5):
        pb = C.prix_bates([vol ** 2, vol ** 2, 1.0, 1e-3, 0.0, 0.0, -0.1, 0.1], 100.0, K, T, 0.97,
                          K >= 100)
        pire = max(pire, np.abs(pb - C.prix_b76(100.0, K, T, 0.97, vol, K >= 100)).max())
controle("sans vol de vol ni sauts, Bates = Black-Scholes (< 1e-4 pt pour 100)", pire < 1e-4,
         f"{pire:.1e}")
lam, mu, sj, vol = 0.5, -0.12, 0.15, 0.18
kb = math.exp(mu + 0.5 * sj * sj) - 1
pire = 0.0
for T in (0.1, 1.0, 4.0):
    pb = C.prix_bates([vol ** 2, vol ** 2, 1.0, 1e-3, 0.0, lam, mu, sj], 100.0, K, T, 1.0, K >= 100)
    ref = np.zeros_like(K)
    for k in range(60):     # Merton : melange de Black-Scholes selon le nombre de sauts
        w = math.exp(-lam * T) * (lam * T) ** k / math.factorial(k)
        Fk = 100.0 * math.exp(-lam * kb * T + k * mu + 0.5 * k * sj * sj)
        ref += w * C.prix_b76(Fk, K, T, 1.0, math.sqrt(vol * vol + k * sj * sj / T), K >= 100)
    pire = max(pire, np.abs(pb - ref).max())
controle("sans vol de vol, Bates = Merton (formule en serie, < 1e-4 pt)", pire < 1e-4, f"{pire:.1e}")
p_test = [0.02, 0.04, 1.0, 1.5, -0.9, 0.3, -0.1, 0.12]      # vol de vol au maximum du curseur
u = np.linspace(0.0, 6000.0, 1200001)
phi = C.fonction_caracteristique(u - 0.5j, [0.05], *p_test)[0]
ref = np.array([100 - math.sqrt(100 * k) / math.pi
                * np.trapezoid((np.exp(1j * math.log(100 / k) * u) * phi).real / (u * u + 0.25), u)
                for k in K])
gl = C.prix_bates(p_test, 100.0, K, 0.05, 1.0, True)
controle("quadrature de Gauss-Legendre = integration fine, meme a xi = 1,5 et T = 3 semaines",
         np.abs(gl - ref).max() < 1e-6, f"{np.abs(gl - ref).max():.1e}")
c_ = C.prix_bates(p_test, 100.0, K, 1.0, 0.95, True)
p_ = C.prix_bates(p_test, 100.0, K, 1.0, 0.95, False)
controle("parite call-put exacte", np.abs(c_ - p_ - 0.95 * (100 - K)).max() < 1e-10)

params = C.charge_parametres()["parametres"]
ks = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
t = time.time()
f_, mc, vf, vm = C.compare_monte_carlo(params, 1.0, 0.03, 0.025, ks, n_paths=80000, pas_par_an=252)
e_prix = np.abs(mc / f_ - 1)[2:5]
controle("formule = Monte Carlo du moteur, pas quotidien : prix a 1,5 % pres de 90 % a 110 %",
         e_prix.max() < 0.015, f"{e_prix.max():.2%}, {time.time() - t:.1f}s")
controle("formule = Monte Carlo du moteur, pas quotidien : vols a 0,3 pt pres de 70 % a 120 %",
         np.abs(vm - vf).max() * 100 < 0.3, f"{np.abs(vm - vf).max() * 100:.2f} pt")
f_, mc, vf, vm = C.compare_monte_carlo(params, 1.0, 0.03, 0.025, ks, n_paths=100000, pas_par_an=52)
controle("pas hebdomadaire de l'appli : vols a 0,4 pt pres (biais de discretisation connu)",
         np.abs(vm - vf).max() * 100 < 0.4, f"{np.abs(vm - vf).max() * 100:.2f} pt")


# =====================================================================
# 4. Calibrage
# =====================================================================
print("--- Calibrage")
res = C.charge_parametres()
p = res["parametres"]
controle("parametres dans les bornes des curseurs",
         all(C.BORNES[k][0] - 1e-9 <= p[k] <= C.BORNES[k][1] + 1e-9 for k in C.NOMS))
cur = res["curseurs"]
sur_grille = all(abs((cur[k] - b) / pas - round((cur[k] - b) / pas)) < 1e-6 and b <= cur[k] <= hmax
                 for k, (pas, b, hmax) in C.GRILLE.items())
controle("reglage curseurs sur la grille de la barre laterale", sur_grille, str(cur))
vol, ecart = C.erreur_vols(p, o)
rmse = float(np.sqrt(np.mean(ecart ** 2)))
controle("fichier coherent avec le code et les donnees (meme erreur recalculee)",
         abs(rmse - res["erreur_pts_vol"]["moyenne_quadratique"]) < 2e-3, f"{rmse:.3f} pt")
controle("erreur de calibrage : moyenne < 1 pt de vol, max < 2,5 pts",
         rmse < 1.0 and np.abs(ecart).max() < 2.5, f"{rmse:.2f} / {np.abs(ecart).max():.2f}")
_, ecart_c = C.erreur_vols(C.depuis_curseurs(cur), o)
controle("reglage curseurs : moyenne < 1,2 pt, max < 3,5 pts",
         np.sqrt(np.mean(ecart_c ** 2)) < 1.2 and np.abs(ecart_c).max() < 3.5,
         f"{np.sqrt(np.mean(ecart_c ** 2)):.2f} / {np.abs(ecart_c).max():.2f}")


def rosenbrock(x):
    return float(sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


x, fx, _ = C.nelder_mead(rosenbrock, np.array([-1.2, 1.0, -0.5, 0.8]), max_eval=20000)
controle("Nelder-Mead retrouve le minimum de Rosenbrock en dimension 4",
         np.abs(x - 1).max() < 1e-3, f"{np.round(x, 5)}")
# Surface fictive generee par des parametres connus : le calibrage doit la reproduire
vrai = {"v0": 0.03, "theta": 0.045, "kappa": 2.0, "xi": 0.6, "rho": -0.7,
        "lam": 0.2, "mu_j": -0.12, "sig_j": 0.15}
fictif = dict(o)
fictif["prix"] = C.prix_bates(vrai, o["F"], o["K"], o["T"], o["DF"], o["call"])
fictif["vol"] = C.vol_implicite_b76(fictif["prix"], o["F"], o["K"], o["T"], o["DF"], o["call"])
t = time.time()
trouve = C.calibre(fictif, departs=C.DEPARTS[:1], verbeux=False)
_, e_fictif = C.erreur_vols(trouve, fictif)
controle("calibrage sur une surface fictive : la retrouve a 0,05 pt de vol pres",
         np.abs(e_fictif).max() < 0.05, f"{np.abs(e_fictif).max():.3f} pt, {time.time() - t:.1f}s")


# =====================================================================
# 5. Backtest
# =====================================================================
print("--- Backtest")
t = time.time()
h2 = B.charge_historique()
ac = B.backtest_autocall(h2)
brc = B.backtest_brc(h2)
acd = B.backtest_autocall(h2, sous_jacent="decrement")
brcd = B.backtest_brc(h2, sous_jacent="decrement")
duree = time.time() - t
controle("lecture + 4 backtests en moins d'1 s", duree < 1.0, f"{duree * 1000:.0f} ms")

# meme logique que le moteur, appliquee aux chemins historiques
chemins = 100 * np.hstack([np.ones((len(ac.emission), 1)), ac.perf_obs])
pv = M.pv_autocall(chemins, 3.0, 0.0, 1.0, ac.n_obs, 1.0, 0.7, 0.6, True, 0.02)
moi = (ac.coupons + ac.capital).mean()
controle("autocall : flux moyens = moteur.pv_autocall sur les memes chemins (taux nul)",
         abs(pv - moi) < 1e-9, f"{moi:.6f} vs {pv:.6f}")
_, p_rappel, p_perte = M.stats_autocall(chemins, 3.0, ac.n_obs, 1.0, 0.6)
controle("autocall : rappels et pertes = moteur.stats_autocall",
         abs(p_rappel - ac.rappele.mean()) < 1e-12 and abs(p_perte - ac.perte.mean()) < 1e-12)
brc_e = B.backtest_brc(h2, continu=False)
ch2 = 100 * np.column_stack([np.ones(len(brc_e.perf_fin)), brc_e.perf_fin])
pv = M.pv_brc(ch2, 1.0, 0.0, 1.0, 0.7, 1.0, False, 0.07, 0.2)
moi = (brc_e.capital + 0.07).mean()
controle("BRC barriere a l'echeance : flux moyens = moteur.pv_brc", abs(pv - moi) < 1e-9,
         f"{moi:.6f} vs {pv:.6f}")

# cas fictifs calcules a la main
jours = np.arange(np.datetime64("2000-01-03"), np.datetime64("2010-01-01"))
plat = B.historique(jours, np.full(len(jours), 100.0), np.full(len(jours), 100.0))
r = B.backtest_autocall(plat, coupon_pa=0.08, debut="2000-01-01", fin="2000-12-31")
controle("indice plat : autocall rappele a la 1re observation, un coupon de 2 %",
         np.all(r.obs_fin == 1) and np.allclose(r.coupons, 0.02) and not r.perte.any())
controle("indice plat : rendement = 2 % par trimestre, soit (1,02)^(365,25/jours) - 1",
         np.allclose(r.tri, 1.02 ** (1 / r.duree) - 1, atol=1e-9))
baisse = np.maximum(100.0 - 55.0 * np.arange(len(jours)) / 1096.0, 45.0)   # -55 % en 3 ans
chute = B.historique(jours, baisse, baisse)
r = B.backtest_autocall(chute, debut="2000-01-01", fin="2000-01-31")
controle("baisse reguliere de 55 % en 3 ans : autocall a l'echeance, perte = niveau final",
         r.obs_fin[0] == 12 and abs(r.capital[0] - r.perf_fin[0]) < 1e-12
         and abs(r.capital[0] - 0.45) < 0.002, f"capital rendu {r.capital[0]:.1%}")
controle("baisse reguliere : 6 coupons, tant que l'indice reste au-dessus de 70 %",
         abs(r.coupons[0] - 0.02 * 6) < 1e-12 and (r.perf_obs[0] >= 0.7).sum() == 6,
         f"{r.nb_coupons[0]:.0f} coupons")
rb = B.backtest_brc(chute, T=1.0, barriere=0.95, debut="2000-01-01", fin="2000-01-31")
controle("baisse reguliere : BRC barriere 95 % touchee, capital = niveau final / strike",
         rb.touche[0] and abs(rb.capital[0] - rb.perf_fin[0]) < 1e-12)
# indice a decrement
d0 = B._perf(h2, "decrement", [jour(h2, "2005-01-03")], np.array([[jour(h2, "2010-01-04")]]),
             points=0.0)[0, 0]
d1 = h2.net_return[jour(h2, "2010-01-04")] / h2.net_return[jour(h2, "2005-01-03")]
controle("decrement de 0 point = net return", abs(d0 - d1) < 1e-12)
ans = np.arange(np.datetime64("2001-01-01"), np.datetime64("2002-01-02"))
const = B.historique(ans, np.full(len(ans), 1.0), np.full(len(ans), 1.0))
_, dec = B.indice_decrement(const, "2001-01-01")
controle("dividendes reinvestis plats : l'indice a decrement perd 50 points en 365 jours",
         abs(dec[365] - 950.0) < 1e-9, f"{dec[365]:.4f}")
d_dec, dec = B.indice_decrement(h2, "2000-01-03")
zero = d_dec[np.argmax(dec <= 0)]
controle("lance a 1000 en janvier 2000, l'indice a decrement 50 pts tombe a zero fin 2014",
         dec.min() == 0.0 and str(zero)[:7] == "2014-12", f"zero le {zero}")
_, dec = B.indice_decrement(h2, "2010-01-04")
perf_px = h2.prix[-1] / h2.prix[jour(h2, "2010-01-04")]
perf_nr = h2.net_return[-1] / h2.net_return[jour(h2, "2010-01-04")]
controle("lance en 2010 : entre le prix et le net return (decrement > dividendes reels)",
         dec[-1] / 1000 < perf_px < perf_nr, f"{dec[-1] / 1000:.2f} < {perf_px:.2f} < {perf_nr:.2f}")

# coherence du calendrier
idx = np.searchsorted(h2.dates, ac.dates_obs)
retard = (ac.dates_obs[:, 0] - ac.emission).astype(int)
controle("observations : jours de cotation, premiere entre 89 et 96 jours apres l'emission",
         np.all(h2.dates[idx] == ac.dates_obs) and retard.min() >= 89 and retard.max() <= 96,
         f"{retard.min()}-{retard.max()} jours")
controle("aucun produit compte apres la fin de l'historique (pas de biais d'anticipation)",
         ac.fin.max() <= h2.fin and brc.fin.max() <= h2.fin)

# chiffres de reference (donnees figees : ils ne bougent que si la logique change)
r = ac.resume
controle("autocall 3 ans depuis 2000 : 286 emissions, 57,7 % rappeles a la 1re date",
         r["n"] == 286 and abs(r["rappel_premiere_date"] - 0.577) < 0.002,
         f"{r['n']}, {r['rappel_premiere_date']:.1%}")
controle("autocall 3 ans depuis 2000 : 3,5 % de pertes, pire cas emis le 02/05/2000 (42,7 % rendu)",
         abs(r["perte"] - 0.035) < 0.002 and r["pire_emission"] == "2000-05-02"
         and abs(r["pire_capital"] - 0.427) < 0.002,
         f"{r['perte']:.1%}, {r['pire_emission']}, {r['pire_capital']:.1%}")
r = brc.resume
controle("BRC 1 an barriere 70 % depuis 2000 : 15,8 % de pertes, pire cas emis le 03/03/2008",
         abs(r["perte"] - 0.158) < 0.002 and r["pire_emission"] == "2008-03-03",
         f"{r['perte']:.1%}, {r['pire_emission']}")
controle("sur indice a decrement, plus de pertes que sur l'indice des prix (meme coupon)",
         acd.resume["perte"] > ac.resume["perte"] and brcd.resume["perte"] > brc.resume["perte"])

print()
if echecs:
    print(f"{len(echecs)} probleme(s) :")
    for e in echecs:
        print("  - " + e)
    sys.exit(1)
print("Tout est bon.")
