"""
=======================================================================
CALIBRAGE DE BATES SUR DE VRAIS PRIX D'OPTIONS
Tom Uzan
=======================================================================

    python calibration.py        # recalibre et reecrit data/bates_calibre.json

L'appli ne recalibre jamais en ligne : elle lit data/bates_calibre.json
(charge_parametres). Ce module ne depend que de numpy et de moteur.py.

Ce que fait un desk, et ce que fait ce fichier :

  1. Une photo du marche : prix de reglement Eurex des options sur l'Euro
     Stoxx 50 (data/options_eurostoxx50.csv), de 1 mois a 5 ans, de 60 %
     a 130 % du spot.
  2. Un pricer rapide des vanilles sous Bates. Le Monte Carlo du moteur
     prend une seconde par jeu de parametres ; un calibrage en demande des
     milliers. On passe donc par la fonction caracteristique de Bates, connue
     en formule fermee, et la formule de Lewis (une integrale sur une
     frequence, calculee par quadrature de Gauss-Legendre) : 1 ms pour
     toute la surface.
  3. Un optimiseur : Nelder-Mead (simplexe), code ici en numpy. On cherche
     les 8 parametres qui minimisent l'ecart de vol implicite modele /
     marche, dans les bornes des curseurs de la barre laterale.

Le resultat se lit comme un desk le lirait : un smile de marche, un smile
modele, et l'ecart en points de vol strike par strike. Un modele a 8
parametres constants ne colle jamais parfaitement de 1 mois a 5 ans : la
taille de l'ecart residuel EST l'information (voir Limites du modele).
=======================================================================
"""

import datetime as dt
import json
import math
import os

import numpy as np

import moteur as M

RACINE = os.path.dirname(os.path.abspath(__file__))
FICHIER_OPTIONS = os.path.join(RACINE, "data", "options_eurostoxx50.csv")
FICHIER_PARAMS = os.path.join(RACINE, "data", "bates_calibre.json")

NOMS = ("v0", "theta", "kappa", "xi", "rho", "lam", "mu_j", "sig_j")

# Bornes = celles des curseurs Bates de la barre laterale, pour que le
# resultat soit directement affichable. v0 et theta sont des variances.
BORNES = {
    "v0": (0.08 ** 2, 0.70 ** 2),
    "theta": (0.10 ** 2, 0.50 ** 2),
    "kappa": (0.2, 6.0),
    "xi": (0.05, 1.5),
    "rho": (-0.95, 0.0),
    "lam": (0.0, 2.0),
    "mu_j": (-0.30, 0.05),
    "sig_j": (0.01, 0.40),
}


# ---------------------------------------------------------------------
# Black-76 vectorise (prix sur le forward) et vol implicite
# ---------------------------------------------------------------------

def prix_b76(F, K, T, DF, vol, call):
    """Prix Black-76. Toutes les entrees sont diffusables (numpy)."""
    F, K, T, DF, vol, call = np.broadcast_arrays(
        np.asarray(F, float), np.asarray(K, float), np.asarray(T, float),
        np.asarray(DF, float), np.asarray(vol, float), np.asarray(call, bool))
    sq = np.maximum(vol * np.sqrt(T), 1e-12)
    d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    d2 = d1 - sq
    c = DF * (F * M.norm_cdf_vec(d1) - K * M.norm_cdf_vec(d2))
    p = DF * (K * M.norm_cdf_vec(-d2) - F * M.norm_cdf_vec(-d1))
    return np.where(call, c, p)


def vega_b76(F, K, T, DF, vol):
    sq = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    return DF * F * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)


def vol_implicite_b76(prix, F, K, T, DF, call, iterations=80):
    """Vol implicite par dichotomie vectorisee (monotone, toujours convergente)."""
    prix = np.asarray(prix, float)
    lo = np.full(prix.shape, 1e-4)
    hi = np.full(prix.shape, 4.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        trop_bas = prix_b76(F, K, T, DF, mid, call) < prix
        lo = np.where(trop_bas, mid, lo)
        hi = np.where(trop_bas, hi, mid)
    vol = 0.5 * (lo + hi)
    # prix hors des bornes d'arbitrage : pas de vol
    intr = np.where(call, DF * np.maximum(F - K, 0), DF * np.maximum(K - F, 0))
    return np.where((prix <= intr + 1e-12) | (vol > 3.99), np.nan, vol)


# ---------------------------------------------------------------------
# Bates : fonction caracteristique et formule de Lewis
# ---------------------------------------------------------------------

def fonction_caracteristique(u, T, v0, theta, kappa, xi, rho, lam, mu_j, sig_j):
    """
    E[ exp(i u ln(S_T / F_T)) ] sous Bates, u complexe (n_u,), T (n_T,)
    -> tableau (n_T, n_u).

    Partie Heston : formulation d'Albrecher et al. (« little Heston trap »),
    qui evite les sauts de branche du logarithme complexe sur les longues
    maturites. Partie sauts : Poisson compose de sauts lognormaux, avec le
    meme compensateur que moteur.simule_bates (drift risque-neutre).
    """
    u = np.asarray(u, complex)[None, :]
    T = np.asarray(T, float)[:, None]
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi ** 2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (b - d) / xi ** 2 * (1.0 - e) / (1.0 - g * e)
    kbar = math.exp(mu_j + 0.5 * sig_j ** 2) - 1.0
    J = lam * T * (np.exp(iu * mu_j - 0.5 * sig_j ** 2 * u * u) - 1.0 - iu * kbar)
    return np.exp(C + D * v0 + J)


def _noeuds():
    """
    Quadrature de Gauss-Legendre composite sur [0, 2000]. Les segments
    s'elargissent la ou l'integrande s'ecrase ; le nombre de points suit
    l'oscillation de e^{iuk} (strikes jusqu'a 50 % / 150 % du forward).
    Couper a 250 suffit en Black-Scholes, pas avec une vol de la vol
    elevee et une maturite courte : la queue y pese jusqu'a 1e-2 point.
    """
    segments = [(0, 8, 48), (8, 30, 48), (30, 90, 48), (90, 250, 64),
                (250, 700, 160), (700, 2000, 420)]
    us, ws = [], []
    for a, b, n in segments:
        x, w = np.polynomial.legendre.leggauss(n)
        us.append(0.5 * (b - a) * x + 0.5 * (b + a))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(us), np.concatenate(ws)


U_NOEUDS, W_NOEUDS = _noeuds()


class Surface:
    """
    Un jeu d'options fixe, des parametres qui changent : tout ce qui ne
    depend pas des parametres (les e^{iuk}, les poids) est calcule une fois.
    Un prix de toute la surface coute alors ~1 ms, ce qui rend le calibrage
    possible en quelques secondes.

    Formule de Lewis (2001) :

        C = DF * [ F - sqrt(F K) / pi * Int_0^inf Re( e^{i u k} phi(u - i/2) ) / (u^2 + 1/4) du ]

    avec k = ln(F/K) et phi la fonction caracteristique de ln(S_T/F_T).
    Le put s'en deduit par parite call-put.
    """

    def __init__(self, F, K, T, DF, call):
        F, K, T, DF, call = np.broadcast_arrays(
            np.asarray(F, float), np.asarray(K, float), np.asarray(T, float),
            np.asarray(DF, float), np.asarray(call, bool))
        self.forme = F.shape
        self.F, self.K, self.DF, self.call = (a.ravel() for a in (F, K, DF, call))
        self.Tu, self.inv = np.unique(T.ravel(), return_inverse=True)
        angle = np.log(self.F / self.K)[:, None] * U_NOEUDS[None, :]
        poids = W_NOEUDS / (U_NOEUDS ** 2 + 0.25)
        self.cos, self.sin = np.cos(angle) * poids, np.sin(angle) * poids

    def prix(self, params):
        with np.errstate(all="ignore"):
            phi = fonction_caracteristique(U_NOEUDS - 0.5j, self.Tu, *_vecteur(params))[self.inv]
            integrale = (np.einsum("ij,ij->i", self.cos, phi.real)
                         - np.einsum("ij,ij->i", self.sin, phi.imag))
            c = self.DF * (self.F - np.sqrt(self.F * self.K) / math.pi * integrale)
        out = np.where(self.call, c, c - self.DF * (self.F - self.K))
        return out.reshape(self.forme)


def prix_bates(params, F, K, T, DF, call):
    """
    Prix de vanilles europeennes sous Bates (formule de Lewis, voir Surface).
    params : dict ou sequence dans l'ordre NOMS. Les options peuvent avoir
    des maturites differentes ; phi n'est calculee qu'une fois par maturite.
    """
    return Surface(F, K, T, DF, call).prix(params)


def vol_bates(params, T, strikes_pct, r=0.0, q=0.0, spot=M.S0):
    """
    Smile genere par Bates, en formule fermee : vol implicite de chaque
    strike (en % du spot) pour une maturite T. Remplace en 1 ms le Monte
    Carlo de moteur.vol_implicite_bates.
    """
    K = np.asarray(strikes_pct, float) * spot
    F = spot * math.exp((r - q) * T)
    DF = math.exp(-r * T)
    call = K >= F
    prix = prix_bates(params, F, K, T, DF, call)
    return vol_implicite_b76(prix, F, K, T, DF, call)


def _vecteur(params):
    if isinstance(params, dict):
        return [float(params[n]) for n in NOMS]
    return [float(x) for x in params]


# ---------------------------------------------------------------------
# Optimiseur : Nelder-Mead, en numpy
# ---------------------------------------------------------------------

def nelder_mead(f, x0, pas=0.4, tol_f=1e-12, tol_x=1e-6, max_eval=5000):
    """
    Simplexe de Nelder-Mead, coefficients adaptes a la dimension (Gao et
    Han, 2012), plus robustes que les coefficients classiques au-dela de
    quelques parametres. Sans derivees : adapte a une fonction d'erreur
    calculee par integration numerique.
    """
    x0 = np.asarray(x0, float)
    n = len(x0)
    a, g, c, s = 1.0, 1.0 + 2.0 / n, 0.75 - 1.0 / (2 * n), 1.0 - 1.0 / n
    pts = np.vstack([x0] + [x0 + pas * np.eye(n)[i] for i in range(n)])
    val = np.array([f(x) for x in pts])
    nev = n + 1
    while nev < max_eval:
        o = np.argsort(val)
        pts, val = pts[o], val[o]
        if (val[-1] - val[0] <= tol_f * (abs(val[0]) + tol_f)
                and np.abs(pts[1:] - pts[0]).max() <= tol_x):
            break
        cen = pts[:-1].mean(axis=0)
        xr = cen + a * (cen - pts[-1])
        fr = f(xr)
        nev += 1
        if val[0] <= fr < val[-2]:
            pts[-1], val[-1] = xr, fr
            continue
        if fr < val[0]:
            xe = cen + g * (xr - cen)
            fe = f(xe)
            nev += 1
            pts[-1], val[-1] = (xe, fe) if fe < fr else (xr, fr)
            continue
        if fr < val[-1]:
            xc = cen + c * (xr - cen)
        else:
            xc = cen - c * (cen - pts[-1])
        fc = f(xc)
        nev += 1
        if fc < min(fr, val[-1]):
            pts[-1], val[-1] = xc, fc
            continue
        pts[1:] = pts[0] + s * (pts[1:] - pts[0])
        val[1:] = [f(x) for x in pts[1:]]
        nev += n
    i = int(np.argmin(val))
    return pts[i], float(val[i]), nev


# Passage borne <-> libre : p = bas + (haut - bas) * sigmoide(x)
_BAS = np.array([BORNES[n][0] for n in NOMS])
_HAUT = np.array([BORNES[n][1] for n in NOMS])


def _vers_params(x):
    return _BAS + (_HAUT - _BAS) / (1.0 + np.exp(-np.asarray(x)))


def _vers_libre(p):
    z = np.clip((np.asarray(p, float) - _BAS) / (_HAUT - _BAS), 1e-6, 1 - 1e-6)
    return np.log(z / (1 - z))


# ---------------------------------------------------------------------
# Donnees de marche
# ---------------------------------------------------------------------

def charge_options(chemin=FICHIER_OPTIONS):
    """Photo d'options : dict de tableaux numpy + date_marche et spot."""
    meta, lignes = {}, []
    with open(chemin) as f:
        for ligne in f:
            if ligne.startswith("#"):
                if ":" in ligne:
                    cle, val = ligne[1:].split(":", 1)
                    meta[cle.strip()] = val.strip()
                continue
            lignes.append(ligne.rstrip("\n").split(","))
    entete, lignes = lignes[0], lignes[1:]
    col = {nom: [l[i] for l in lignes] for i, nom in enumerate(entete)}
    out = {
        "echeance": np.array(col["echeance"], dtype="datetime64[D]"),
        "T": np.array(col["maturite"], float),
        "call": np.array([t == "C" for t in col["type"]]),
        "K": np.array(col["strike"], float),
        "prix": np.array(col["prix"], float),
        "F": np.array(col["forward"], float),
        "DF": np.array(col["actualisation"], float),
        "vol": np.array(col["vol_implicite"], float),
        "open_interest": np.array(col["open_interest"], int),
        "date_marche": meta["date_marche"],
        "spot": float(meta["spot"]),
    }
    out["strike_pct"] = out["K"] / out["spot"]
    return out


def erreur_vols(params, o):
    """Vol implicite modele et ecart au marche (en points de vol), option par option."""
    prix = prix_bates(params, o["F"], o["K"], o["T"], o["DF"], o["call"])
    vol = vol_implicite_b76(prix, o["F"], o["K"], o["T"], o["DF"], o["call"])
    return vol, (vol - o["vol"]) * 100.0


# ---------------------------------------------------------------------
# Calibrage
# ---------------------------------------------------------------------

DEPARTS = [
    # v0     theta  kappa  xi    rho    lam   mu_j   sig_j
    [0.025, 0.040, 2.0, 0.50, -0.70, 0.30, -0.10, 0.15],
    [0.020, 0.035, 1.0, 0.30, -0.80, 0.10, -0.20, 0.10],
    [0.030, 0.050, 4.0, 1.00, -0.60, 0.80, -0.05, 0.05],
    [0.020, 0.030, 0.5, 0.20, -0.90, 0.05, -0.25, 0.20],
    [0.025, 0.045, 3.0, 0.70, -0.50, 1.50, -0.08, 0.08],
]


def calibre(o=None, departs=DEPARTS, verbeux=True, fixes=None):
    """
    Cherche les 8 parametres de Bates qui reproduisent le mieux la surface.

    Critere : moyenne des carres des ecarts de vol implicite. Pour aller
    vite, l'ecart de vol est approche par ecart de prix / vega de marche
    (exact au premier ordre, et c'est au voisinage de la solution qu'on en
    a besoin) ; les ecarts affiches a la fin sont recalcules exactement.

    Plusieurs points de depart, car le probleme a des minima locaux : sauts
    frequents et petits, ou rares et violents, peuvent produire des smiles
    proches. On garde le meilleur, puis on relance le simplexe dessus
    jusqu'a ce qu'il ne gagne plus rien.

    fixes : parametres imposes (ex. {"lam": 0.1}), les autres sont calibres.
    """
    o = charge_options() if o is None else o
    fixes = fixes or {}
    libres = [i for i, n in enumerate(NOMS) if n not in fixes]
    vega = vega_b76(o["F"], o["K"], o["T"], o["DF"], o["vol"])
    surface = Surface(o["F"], o["K"], o["T"], o["DF"], o["call"])

    def complet(x):
        p = _vers_params(np.zeros(len(NOMS)))
        p[libres] = _vers_params(_insere(x, libres))[libres]
        for n, v in fixes.items():
            p[NOMS.index(n)] = v
        return p

    def critere(x):
        e = (surface.prix(complet(x)) - o["prix"]) / vega
        return float(np.mean(e * e)) if np.all(np.isfinite(e)) else 1e6

    meilleur = None
    for k, p0 in enumerate(departs):
        x, fx, nev = nelder_mead(critere, _vers_libre(_vecteur(p0))[libres], max_eval=3000)
        if verbeux:
            print(f"  depart {k + 1} : erreur moyenne {math.sqrt(fx) * 100:.3f} pt de vol "
                  f"({nev} evaluations)")
        if meilleur is None or fx < meilleur[1]:
            meilleur = (x, fx)
    x, fx = meilleur
    for _ in range(6):
        x2, fx2, _ = nelder_mead(critere, x, pas=0.15, max_eval=3000)
        gain = fx - fx2
        if fx2 < fx:
            x, fx = x2, fx2
        if gain < 1e-9:
            break
    return dict(zip(NOMS, (float(v) for v in complet(x))))


def _insere(x, libres):
    plein = np.zeros(len(NOMS))
    plein[libres] = x
    return plein


def vers_curseurs(p):
    """Parametres -> unites et pas des curseurs de la barre laterale."""
    def arrondi(v, pas):
        return round(round(v / pas) * pas, 4)
    return {
        "vol_atm": arrondi(math.sqrt(p["v0"]) * 100, 1.0),
        "theta_v": arrondi(math.sqrt(p["theta"]) * 100, 1.0),
        "kappa": arrondi(p["kappa"], 0.2),
        "xi": arrondi(p["xi"], 0.05),
        "rho": arrondi(p["rho"], 0.05),
        "lam": arrondi(p["lam"], 0.1),
        "mu_j": arrondi(p["mu_j"] * 100, 1.0),
        "sig_j": arrondi(p["sig_j"] * 100, 1.0),
    }


def depuis_curseurs(c):
    return {"v0": (c["vol_atm"] / 100) ** 2, "theta": (c["theta_v"] / 100) ** 2,
            "kappa": c["kappa"], "xi": c["xi"], "rho": c["rho"], "lam": c["lam"],
            "mu_j": c["mu_j"] / 100, "sig_j": c["sig_j"] / 100}


# pas, minimum, maximum de chaque curseur Bates (barre.py)
GRILLE = {"vol_atm": (1.0, 8.0, 70.0), "theta_v": (1.0, 10.0, 50.0), "kappa": (0.2, 0.2, 6.0),
          "xi": (0.05, 0.05, 1.5), "rho": (0.05, -0.95, 0.0), "lam": (0.1, 0.0, 2.0),
          "mu_j": (1.0, -30.0, 5.0), "sig_j": (1.0, 1.0, 40.0)}


def calibre_curseurs(p, o=None, verbeux=True):
    """
    Meilleur reglage atteignable avec les curseurs. Arrondir chaque
    parametre au cran le plus proche ne suffit pas : une frequence de sauts
    de 0,04 par an s'arrondit a 0 et le skew court disparait (3 a 4 points
    de vol d'ecart sur les echeances de 1 a 2 mois).

    Methode : on recalibre les 7 autres parametres a frequence de sauts
    imposee sur la grille (0 ; 0,1 ; 0,2 ; 0,3), on arrondit, puis on
    deplace un curseur a la fois, de 1 a 3 crans, tant que l'ecart de vol
    implicite (exact, cette fois) baisse. On garde le meilleur.
    """
    o = charge_options() if o is None else o
    surface = Surface(o["F"], o["K"], o["T"], o["DF"], o["call"])

    def critere(c):
        prix = surface.prix(depuis_curseurs(c))
        vol = vol_implicite_b76(prix, o["F"], o["K"], o["T"], o["DF"], o["call"], iterations=40)
        e = np.where(np.isfinite(vol), vol - o["vol"], 0.5)
        return float(np.mean(e * e))

    departs = [vers_curseurs(p)]
    for lam in (0.1, 0.2, 0.3):
        p_lam = calibre(o, departs=[dict(p, lam=lam)], verbeux=False, fixes={"lam": lam})
        departs.append(vers_curseurs(p_lam))
    meilleur = None
    for c in departs:
        fc = critere(c)
        while True:
            essais = []
            for nom, (pas, bas, haut) in GRILLE.items():
                for crans in (-3, -2, -1, 1, 2, 3):
                    v = round(c[nom] + crans * pas, 4)
                    if bas - 1e-9 <= v <= haut + 1e-9:
                        essai = dict(c, **{nom: v})
                        essais.append((critere(essai), essai))
            f_mieux, c_mieux = min(essais, key=lambda e: e[0])
            if f_mieux >= fc - 1e-12:
                break
            c, fc = c_mieux, f_mieux
        if verbeux:
            print(f"  grille des curseurs (lam={c['lam']:.1f}) : "
                  f"erreur moyenne {math.sqrt(fc) * 100:.3f} pt de vol")
        if meilleur is None or fc < meilleur[1]:
            meilleur = (c, fc)
    return meilleur[0]


def ecrit_resultat(p, o, curs=None, chemin=FICHIER_PARAMS):
    vol, ecart = erreur_vols(p, o)
    curs = calibre_curseurs(p, o, verbeux=False) if curs is None else curs
    vol_c, ecart_c = erreur_vols(depuis_curseurs(curs), o)
    ok = np.isfinite(ecart) & np.isfinite(ecart_c)

    def rmse(e):
        return round(float(np.sqrt(np.mean(e ** 2))), 3)

    par_echeance = []
    for ech in np.unique(o["echeance"]):
        m = ok & (o["echeance"] == ech)
        par_echeance.append({"echeance": str(ech), "maturite": round(float(o["T"][m][0]), 4),
                             "n": int(m.sum()), "moyenne_quadratique": rmse(ecart[m]),
                             "moyenne_quadratique_curseurs": rmse(ecart_c[m])})
    res = {
        "source": "Eurex OESX, prix de reglement officiels (data/options_eurostoxx50.csv)",
        "date_marche": o["date_marche"],
        "spot": o["spot"],
        "calibre_le": dt.date.today().isoformat(),
        "n_options": int(ok.sum()),
        "unites": "v0 et theta en variance ; curseurs dans les unites de la barre laterale "
                  "(vol_atm = racine de v0 en %, theta_v = racine de theta en %, mu_j et sig_j en %)",
        "parametres": {k: round(v, 6) for k, v in p.items()},
        "curseurs": curs,
        "feller": round(2 * p["kappa"] * p["theta"] / p["xi"] ** 2, 3),
        "erreur_pts_vol": {
            "moyenne_quadratique": rmse(ecart[ok]),
            "moyenne_absolue": round(float(np.mean(np.abs(ecart[ok]))), 3),
            "max": round(float(np.max(np.abs(ecart[ok]))), 3),
            "moyenne_quadratique_curseurs": rmse(ecart_c[ok]),
            "max_curseurs": round(float(np.max(np.abs(ecart_c[ok]))), 3),
        },
        "par_echeance": par_echeance,
        "par_option": [
            {"echeance": str(o["echeance"][i]), "maturite": round(float(o["T"][i]), 4),
             "strike_pct": round(float(o["strike_pct"][i]), 4),
             "type": "C" if o["call"][i] else "P",
             "vol_marche": round(float(o["vol"][i]), 4),
             "vol_modele": round(float(vol[i]), 4) if ok[i] else None,
             "vol_curseurs": round(float(vol_c[i]), 4) if ok[i] else None}
            for i in range(len(vol))],
    }
    with open(chemin, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=True)
    return res


def charge_parametres(chemin=FICHIER_PARAMS):
    """Ce que lit l'appli : le dict complet (parametres, curseurs, erreurs)."""
    with open(chemin) as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Controle croise avec le Monte Carlo du moteur
# ---------------------------------------------------------------------

def compare_monte_carlo(params, T, r, q, strikes_pct, n_paths=100000, pas_par_an=52, seed=7):
    """
    Price les memes vanilles par la formule et par moteur.simule_bates.
    Renvoie (prix_formule, prix_mc, vol_formule, vol_mc) en points pour un
    spot de 100. Si les deux ne concordent pas, l'un des deux est faux.
    """
    p = _vecteur(params)
    ch, _ = M.simule_bates(M.S0, T, r, q, n_paths, *p, pas_par_an=pas_par_an, seed=seed)
    ST = ch[:, -1].astype(np.float64)
    del ch
    K = np.asarray(strikes_pct, float) * M.S0
    F = M.S0 * math.exp((r - q) * T)
    DF = math.exp(-r * T)
    call = K >= F
    mc = np.array([DF * (np.maximum(ST - k, 0) if c else np.maximum(k - ST, 0)).mean()
                   for k, c in zip(K, call)])
    formule = prix_bates(p, F, K, T, DF, call)
    return (formule, mc, vol_implicite_b76(formule, F, K, T, DF, call),
            vol_implicite_b76(mc, F, K, T, DF, call))


if __name__ == "__main__":
    import time
    o = charge_options()
    print(f"Calibrage de Bates sur {len(o['K'])} options Eurex du {o['date_marche']} "
          f"(spot {o['spot']})")
    t = time.time()
    p = calibre(o)
    res = ecrit_resultat(p, o, calibre_curseurs(p, o))
    print(f"  {time.time() - t:.0f} s")
    print("Parametres :", {k: round(v, 4) for k, v in p.items()})
    print("Curseurs   :", res["curseurs"])
    print("Feller 2 kappa theta / xi^2 =", res["feller"])
    print("Erreur (pts de vol) :", res["erreur_pts_vol"])
    print(f"-> {FICHIER_PARAMS}")
