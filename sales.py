"""
=======================================================================
OUTILS DU SALES - module de calcul
Tom Uzan
=======================================================================

Trois outils qu'un Sales utilise face au client, construits sur le
moteur. Aucune interface ici, aucun import streamlit.

  1. Reverse enquiry : le client donne une contrainte (« je veux 8 % »,
     « je veux ma barriere sous 60 % »), on trouve la structure qui la
     respecte. Si c'est hors d'atteinte, on le dit, avec le meilleur
     resultat possible : c'est la reponse que le Sales rapporte au client.
  2. Ou va l'argent : decomposition du nominal en obligation, options,
     risque emetteur, marge banque et retrocession du distributeur. La
     somme retombe exactement sur le nominal.
  3. Marche secondaire : valeur du produit en cours de vie, prix de
     rachat par la banque, et ce qui a fait bouger la valeur depuis
     l'emission (temps, sous-jacent, volatilite, taux, spread).

Conventions :
  - niveaux en fraction du niveau initial (0.70 = 70 %), sous-jacent
    initial a 100 comme dans le moteur ;
  - coupons en fraction par an (autocall : par observation, comme la
    page, et la cible du client en equivalent annuel) ;
  - montants en EUR.

Coherence avec les pages : memes trajectoires (meme graine, memes pas
hebdomadaires), meme skew applique au niveau pertinent (barriere pour le
BRC, barriere de capital pour l'autocall), memes defauts emetteur. Avec
une retrocession nulle, les coupons et participations sont exactement
ceux affiches par les pages.

Retrocession : nouveau parametre. Comme la marge banque, elle est payee
d'avance et prelevee sur le budget option : sur la duree maximale pour le
BRC et le capital garanti, sur la duree de vie ATTENDUE pour l'autocall
(un autocall rappele en moyenne apres 9 mois ne paie pas 3 ans de frais).
=======================================================================
"""

import math
from dataclasses import dataclass, field, fields, replace

import numpy as np

import moteur as M
from moteur import S0


# =====================================================================
# VUE DE MARCHE
# =====================================================================

@dataclass(frozen=True)
class Marche:
    """
    Les reglages de la barre laterale, plus la retrocession.
    Objet fige : il se compare, se hache, et peut servir de cle de cache.
    """
    modele: str = "bs"            # "bs" (Black-Scholes + skew) ou "bates"
    r: float = 0.03
    q: float = 0.025
    vol_atm: float = 0.22
    skew: float = 2.0             # pts de vol pour -10 % de strike (BS seulement)
    theta_v: float = 0.24         # Bates : vol long terme (en vol, pas en variance)
    kappa: float = 2.0
    xi: float = 0.50
    rho: float = -0.70
    lam: float = 0.3
    mu_j: float = -0.10
    sig_j: float = 0.15
    spread: float = 0.0           # spread de credit de l'emetteur, en bp
    recovery: float = 0.40
    marge: float = 0.008          # marge banque, par an
    retro: float = 0.0            # retrocession distributeur, par an, payee d'avance
    nominal: float = 1000.0
    n_paths: int = 10000

    @classmethod
    def depuis_barre(cls, m, retro=0.0):
        """Construit la vue de marche a partir de l'objet renvoye par barre_laterale()."""
        d = cls()

        def ou(x, defaut):
            return defaut if x is None else x

        return cls(modele="bates" if m.BATES else "bs", r=m.r, q=m.q, vol_atm=m.vol_atm,
                   skew=m.skew, theta_v=ou(m.theta_v, d.theta_v), kappa=ou(m.kappa, d.kappa),
                   xi=ou(m.xi, d.xi), rho=ou(m.rho, d.rho), lam=ou(m.lam, d.lam),
                   mu_j=ou(m.mu_j, d.mu_j), sig_j=ou(m.sig_j, d.sig_j), spread=m.spread,
                   recovery=m.recovery, marge=m.marge, retro=retro, nominal=m.nominal,
                   n_paths=m.n_paths)

    @property
    def bates(self):
        return self.modele == "bates"

    def avec(self, **kw):
        return replace(self, **kw)

    def vol(self, niveau):
        """Vol Black-Scholes au niveau donne (en fraction du spot du jour)."""
        if self.bates:
            return self.vol_atm
        return M.vol_au_strike(niveau, self.vol_atm, self.skew)

    def trajectoires(self, T, niveau_ref=1.0, spot=S0, seed=42):
        """Meme appel que barre.trajectoires : (chemins, sigma)."""
        if self.bates:
            return M.simule_bates(spot, T, self.r, self.q, self.n_paths, self.vol_atm ** 2,
                                  self.theta_v ** 2, self.kappa, self.xi, self.rho,
                                  self.lam, self.mu_j, self.sig_j, seed=seed)
        return M.simule_lv(spot, T, self.r, self.q, self.vol_atm, self.skew, self.n_paths,
                           seed=seed)

    def panier(self, n_actifs, correlation, T, niveau_ref, spot=S0, seed=42):
        """Worst-of comme la page autocall (Black-Scholes multi-actifs, meme sous Bates)."""
        if n_actifs <= 1:
            return self.trajectoires(T, niveau_ref, spot, seed)
        return M.simule_panier(n_actifs, correlation, spot, T, self.r, self.q,
                               self.vol(niveau_ref), self.n_paths, seed=seed)

    def defauts(self, n, T):
        return M.tire_defaut(n, T, self.spread, self.recovery) if self.spread > 0 else None


# Champs regroupes pour expliquer ce qui fait bouger la valeur en cours de vie
CHAMPS_VOL = ("modele", "vol_atm", "skew", "theta_v", "kappa", "xi", "rho",
              "lam", "mu_j", "sig_j")
CHAMPS_TAUX = ("r", "q")
CHAMPS_CREDIT = ("spread", "recovery")


def _prend(source, cible, champs):
    return cible.avec(**{c: getattr(source, c) for c in champs})


class _Sims:
    """
    Garde la derniere simulation. Sous Bates, les trajectoires ne dependent
    pas du niveau de la barriere : on ne simule qu'une fois. Sous
    Black-Scholes + skew, la vol depend du niveau, donc on resimule, mais
    toujours avec la meme graine : memes tirages pour tous les essais.
    """

    def __init__(self, mk, T, n_actifs=1, correlation=0.0):
        self.mk, self.T, self.n_actifs, self.correlation = mk, T, n_actifs, correlation
        self.cle, self.res = None, None

    def __call__(self, niveau_ref):
        # mono sous-jacent : vol locale, memes trajectoires pour tous les niveaux
        cle = self.mk.vol(niveau_ref) if self.n_actifs > 1 else "vol locale"
        if cle != self.cle:
            self.res = None                  # libere l'ancienne simulation avant d'en faire une autre
            ch, sg = self.mk.panier(self.n_actifs, self.correlation, self.T, niveau_ref)
            self.res = (ch, sg, self.mk.defauts(len(ch), self.T))
            self.cle = cle
        return self.res


# =====================================================================
# PRIX A L'EMISSION (identiques aux pages quand retro = 0)
# =====================================================================

def _n_obs(T, par_an):
    return max(int(T * par_an), 1)


def _coupon_brc_sur(sims, mk, T, barriere, strike, continu):
    ch, sg, tau = sims(barriere)
    return float(M.coupon_brc(ch, T, mk.r, mk.nominal, barriere, strike, continu,
                              mk.marge + mk.retro, sg, tau, mk.recovery))


def _coupon_autocall_sur(sims, mk, T, n_obs, trigger, bar_coupon, bar_capital, memoire):
    ch, _, tau = sims(bar_capital)
    return float(M.coupon_autocall(ch, T, mk.r, mk.nominal, n_obs, trigger, bar_coupon,
                                   bar_capital, memoire, mk.marge + mk.retro, tau, mk.recovery))


def coupon_brc(mk, T, barriere, strike=1.0, continu=True):
    """Coupon equitable du BRC, par an. Identique a la page."""
    return _coupon_brc_sur(_Sims(mk, T), mk, T, barriere, strike, continu)


def coupon_autocall(mk, T, par_an, trigger, bar_coupon, bar_capital, memoire=True,
                    n_actifs=1, correlation=0.6):
    """Coupon de l'autocall par observation. Identique a la page (x par_an = equivalent annuel)."""
    return _coupon_autocall_sur(_Sims(mk, T, n_actifs, correlation), mk, T,
                                _n_obs(T, par_an), trigger, bar_coupon, bar_capital, memoire)


def _cout_call_spread(mk, T, cap):
    """Les deux calls de la page Capital Protection, par unite de nominal."""
    call = M.bs_call(S0, S0, T, mk.r, mk.vol(1.0), mk.q) / S0
    call_cap = M.bs_call(S0, cap * S0, T, mk.r, mk.vol(cap), mk.q) / S0
    return call, call_cap, max(call - call_cap, 1e-9)


def _budget_cp(mk, T, protection):
    r_fin = mk.r + mk.spread / 10000.0
    zc = protection * math.exp(-r_fin * T)
    return 1.0 - zc - (mk.marge + mk.retro) * T


def participation_cp(mk, T, protection=1.0, cap=3.0):
    """Participation offerte. Identique a la page (cap a 300 % = pas de cap)."""
    return max(_budget_cp(mk, T, protection) / _cout_call_spread(mk, T, cap)[2], 0.0)


# =====================================================================
# 1. REVERSE ENQUIRY
# =====================================================================

@dataclass
class Reponse:
    """
    Reponse d'un solveur. Si la cible est hors d'atteinte, possible=False,
    valeur et obtenu donnent la meilleure structure atteignable.
    courbe : les points calcules sur la grille, [(parametre, resultat)],
    prets a tracer (resultat en fonction du parametre, avec la cible).
    """
    possible: bool
    parametre: str
    valeur: float
    obtenu: float
    cible: float
    message: str
    courbe: list = field(default_factory=list)


def _grille(debut, fin, pas):
    n = max(int(round(abs(fin - debut) / pas)), 1)
    return [debut + (fin - debut) * k / n for k in range(n + 1)]


def _premier_franchissement(f, grille, cible, iterations=7):
    """
    Grille rangee du plus favorable au client au moins favorable.
    Renvoie le premier point ou f atteint la cible, affine par dichotomie
    entre ce point et le precedent. On ne suppose pas f monotone : sous
    Black-Scholes + skew, la vol depend du niveau et le coupon peut
    redescendre. Si la cible n'est jamais atteinte : le maximum de la grille.
    Renvoie (x, f(x), atteint, deja_a_la_borne, points).
    """
    points = [(x, f(x)) for x in grille]
    for i, (x, fx) in enumerate(points):
        if fx >= cible:
            if i == 0:
                return x, fx, True, True, points
            a, b, fb = points[i - 1][0], x, fx
            for _ in range(iterations):
                mil = 0.5 * (a + b)
                fm = f(mil)
                if fm >= cible:
                    b, fb = mil, fm
                else:
                    a = mil
            return b, fb, True, False, points
    x, fx = max(points, key=lambda p: p[1])
    return x, fx, False, False, points


def _pct(x, d=1):
    return f"{x * 100:,.{d}f}%"


def brc_barriere(mk, coupon_cible, T, strike=1.0, continu=True, bornes=(0.40, None)):
    """
    BRC : coupon cible -> barriere la plus basse qui le paie.

    Pourquoi : c'est LA question du client prudent (« je veux 7 %, mais
    avec le plus de marge possible avant que mon capital soit en risque »).
    Plus la barriere est basse, moins le put vendu vaut cher, moins il y a
    de coupon. Au-dessus du strike, la barriere ne sert plus a rien (toute
    trajectoire qui finit sous le strike l'a forcement franchie) : on
    s'arrete donc au strike.
    """
    haut = strike if bornes[1] is None else bornes[1]
    sims = _Sims(mk, T)
    x, fx, ok, borne, pts = _premier_franchissement(
        lambda b: _coupon_brc_sur(sims, mk, T, b, strike, continu),
        _grille(bornes[0], haut, 0.05), coupon_cible)
    if ok and borne:
        msg = (f"Cible depassee : meme avec une barriere a {_pct(x)}, le coupon est de "
               f"{_pct(fx, 2)} p.a.")
    elif ok:
        msg = (f"Barriere a {_pct(x)} pour un coupon de {_pct(fx, 2)} p.a. "
               f"({T:g} an(s), strike {_pct(strike, 0)}).")
    else:
        msg = (f"Impossible : {_pct(coupon_cible, 2)} p.a. n'est pas atteignable a {T:g} an(s) "
               f"et strike {_pct(strike, 0)}. Au mieux {_pct(fx, 2)} p.a., avec une barriere "
               f"a {_pct(x)}. Pour aller plus haut : allonger la maturite ou monter le strike.")
    return Reponse(ok, "barriere", x, fx, coupon_cible, msg, pts)


def brc_strike(mk, coupon_cible, T, barriere, continu=True, bornes=(None, 1.20)):
    """
    BRC : coupon cible -> strike le plus bas qui le paie.

    Pourquoi : un strike sous 100 % (BRC « decote ») veut dire que le client
    ne commence a perdre que sous ce niveau, et perd moins ensuite. Plus le
    strike est bas, moins le put vendu vaut cher. Sous la barriere, le
    strike n'a plus de sens pour un BRC : la borne basse est la barriere.
    """
    bas = barriere if bornes[0] is None else bornes[0]
    sims = _Sims(mk, T)
    x, fx, ok, borne, pts = _premier_franchissement(
        lambda k: _coupon_brc_sur(sims, mk, T, barriere, k, continu),
        _grille(bas, bornes[1], 0.05), coupon_cible)
    if ok and borne:
        msg = (f"Cible depassee : meme avec un strike a {_pct(x)}, le coupon est de "
               f"{_pct(fx, 2)} p.a.")
    elif ok:
        msg = (f"Strike a {_pct(x)} pour un coupon de {_pct(fx, 2)} p.a. "
               f"({T:g} an(s), barriere {_pct(barriere, 0)}).")
    else:
        msg = (f"Impossible : {_pct(coupon_cible, 2)} p.a. n'est pas atteignable avec une "
               f"barriere a {_pct(barriere, 0)}. Au mieux {_pct(fx, 2)} p.a. avec un strike a "
               f"{_pct(x)}. Pour aller plus haut : remonter la barriere.")
    return Reponse(ok, "strike", x, fx, coupon_cible, msg, pts)


def brc_maturite(mk, coupon_cible, barriere, strike=1.0, continu=True, grille=None):
    """
    BRC : coupon cible + barriere -> maturite la plus courte qui le paie.

    Pourquoi : le client ne veut pas immobiliser son argent plus que
    necessaire. Plus le produit est long, plus la barriere a le temps
    d'etre touchee, plus le put vaut cher. Mais le coupon est annuel, et
    au-dela d'un certain point il redescend : on balaie donc toutes les
    maturites (par defaut celles de la page, de 3 mois a 3 ans) au lieu de
    supposer la courbe croissante. Chaque maturite est simulee exactement
    comme la page le ferait.
    """
    grille = grille or [0.25 * k for k in range(1, 13)]
    pts = []
    for T in grille:
        pts.append((T, coupon_brc(mk, T, barriere, strike, continu)))
    ok = [p for p in pts if p[1] >= coupon_cible]
    if ok:
        T, c = ok[0]
        msg = (f"Maturite minimale {T:g} an(s) : coupon {_pct(c, 2)} p.a. avec une barriere "
               f"a {_pct(barriere, 0)}.")
        if T == grille[0]:
            msg = f"Cible depassee des la maturite la plus courte ({T:g} an) : {_pct(c, 2)} p.a."
        return Reponse(True, "maturite", T, c, coupon_cible, msg, pts)
    T, c = max(pts, key=lambda p: p[1])
    msg = (f"Impossible : {_pct(coupon_cible, 2)} p.a. n'est atteint a aucune maturite avec une "
           f"barriere a {_pct(barriere, 0)}. Au mieux {_pct(c, 2)} p.a. a {T:g} an(s). "
           "Pour aller plus haut : remonter la barriere ou le strike.")
    return Reponse(False, "maturite", T, c, coupon_cible, msg, pts)


def autocall_barriere_capital(mk, coupon_cible_pa, T, par_an, trigger=1.0, bar_coupon=0.70,
                              memoire=True, n_actifs=1, correlation=0.6, bornes=(0.30, 1.0)):
    """
    Autocall : coupon annuel cible -> barriere de capital la plus basse.

    Pourquoi : la barriere de capital est le vrai risque du produit (sous
    elle, le client encaisse toute la baisse). Le client veut la plus basse
    possible pour un coupon donne. Attention : sous Black-Scholes + skew,
    la page applique la vol de la barriere a toute la trajectoire, et le
    coupon peut redescendre quand la barriere monte. Le solveur balaie donc
    la grille au lieu de supposer la courbe croissante.
    """
    n_obs = _n_obs(T, par_an)
    sims = _Sims(mk, T, n_actifs, correlation)
    x, fx, ok, borne, pts = _premier_franchissement(
        lambda b: par_an * _coupon_autocall_sur(sims, mk, T, n_obs, trigger, bar_coupon, b,
                                                memoire),
        _grille(bornes[0], bornes[1], 0.05), coupon_cible_pa)
    if ok and borne:
        msg = (f"Cible depassee : meme avec une barriere de capital a {_pct(x)}, le coupon est "
               f"de {_pct(fx, 2)} p.a.")
    elif ok:
        msg = (f"Barriere de capital a {_pct(x)} pour {_pct(fx, 2)} p.a. "
               f"({_pct(fx / par_an, 2)} par observation).")
    else:
        msg = (f"Impossible : {_pct(coupon_cible_pa, 2)} p.a. n'est pas atteignable. Au mieux "
               f"{_pct(fx, 2)} p.a. avec une barriere de capital a {_pct(x)}. Pour aller plus "
               "haut : remonter la barriere de coupon, passer en worst-of, ou allonger.")
    return Reponse(ok, "barriere_capital", x, fx, coupon_cible_pa, msg, pts)


def autocall_trigger(mk, coupon_cible_pa, T, par_an, bar_coupon=0.70, bar_capital=0.60,
                     memoire=True, n_actifs=1, correlation=0.6, bornes=(0.80, 1.20)):
    """
    Autocall : coupon annuel cible -> trigger de rappel le plus bas.

    Pourquoi : un trigger bas fait rappeler le produit plus souvent, donc
    plus tot ; le client recupere son nominal et son coupon sans attendre,
    et reste moins longtemps expose a la barriere de capital. C'est un
    argument commercial (« rappele meme si le marche a baisse de 10 % »).
    En contrepartie, chaque rappel anticipe coupe les coupons futurs : le
    coupon par periode baisse. Ici les trajectoires ne dependent pas du
    trigger, une seule simulation suffit.
    """
    n_obs = _n_obs(T, par_an)
    sims = _Sims(mk, T, n_actifs, correlation)
    x, fx, ok, borne, pts = _premier_franchissement(
        lambda tg: par_an * _coupon_autocall_sur(sims, mk, T, n_obs, tg, bar_coupon,
                                                 bar_capital, memoire),
        _grille(bornes[0], bornes[1], 0.05), coupon_cible_pa)
    if ok and borne:
        msg = (f"Cible depassee : meme avec un trigger a {_pct(x)}, le coupon est de "
               f"{_pct(fx, 2)} p.a.")
    elif ok:
        msg = f"Trigger a {_pct(x)} pour {_pct(fx, 2)} p.a."
    else:
        msg = (f"Impossible : {_pct(coupon_cible_pa, 2)} p.a. n'est pas atteignable en jouant "
               f"sur le trigger. Au mieux {_pct(fx, 2)} p.a. avec un trigger a {_pct(x)}.")
    return Reponse(ok, "trigger", x, fx, coupon_cible_pa, msg, pts)


def cp_protection(mk, participation_cible, T, cap=3.0, bornes=(0.70, 1.0)):
    """
    Capital Protection : participation cible -> protection la plus haute.

    Pourquoi : chaque point de protection abandonne libere du budget pour
    le call. Formule directe : la participation est budget / cout du call
    spread, et le budget est 1 - protection actualisee - frais.
    """
    r_fin = mk.r + mk.spread / 10000.0
    cout = _cout_call_spread(mk, T, cap)[2]
    zc_max = 1.0 - (mk.marge + mk.retro) * T - participation_cible * cout
    prot = zc_max * math.exp(r_fin * T)
    pts = [(p, participation_cp(mk, T, p, cap)) for p in _grille(bornes[1], bornes[0], 0.025)]
    if prot >= bornes[1]:
        p = participation_cp(mk, T, bornes[1], cap)
        msg = (f"Cible depassee : avec {_pct(bornes[1], 0)} de protection, la participation est "
               f"deja de {_pct(p, 0)}.")
        return Reponse(True, "protection", bornes[1], p, participation_cible, msg, pts)
    if prot < bornes[0]:
        p = participation_cp(mk, T, bornes[0], cap)
        il_faudrait = (f" Il faudrait descendre a {_pct(prot)} de protection." if prot > 0
                       else "")
        msg = (f"Impossible : {_pct(participation_cible, 0)} de participation demanderait une "
               f"protection sous {_pct(bornes[0], 0)}.{il_faudrait} Au mieux "
               f"{_pct(p, 0)} avec {_pct(bornes[0], 0)} de protection. Pour aller plus haut : "
               "mettre un cap ou allonger la maturite.")
        return Reponse(False, "protection", bornes[0], p, participation_cible, msg, pts)
    p = participation_cp(mk, T, prot, cap)
    msg = (f"Protection a {_pct(prot)} pour {_pct(p, 0)} de participation : le client accepte "
           f"de perdre au plus {_pct(1 - prot)}.")
    return Reponse(True, "protection", prot, p, participation_cible, msg, pts)


def cp_cap(mk, participation_cible, T, protection=1.0, bornes=(1.05, 3.0)):
    """
    Capital Protection : participation cible -> cap le plus haut.

    Pourquoi : capper, c'est vendre un call de strike eleve ; sa prime
    revient dans le budget et paie de la participation. Le client renonce
    aux tres fortes hausses. Le cap a 300 % de la page vaut « pas de cap ».
    """
    pts = [(c, participation_cp(mk, T, protection, c)) for c in _grille(bornes[1], bornes[0], 0.05)]
    if _budget_cp(mk, T, protection) <= 0:
        msg = (f"Impossible : a ce niveau de taux, garantir {_pct(protection, 0)} du nominal "
               "coute plus que le budget, frais compris. Aucun cap n'y change rien : baisser la "
               "protection, allonger la maturite ou reduire les frais.")
        return Reponse(False, "cap", bornes[1], 0.0, participation_cible, msg, pts)
    haut = participation_cp(mk, T, protection, bornes[1])
    if haut >= participation_cible:
        msg = f"Pas besoin de cap : sans cap, la participation est deja de {_pct(haut, 0)}."
        return Reponse(True, "cap", bornes[1], haut, participation_cible, msg, pts)
    bas = participation_cp(mk, T, protection, bornes[0])
    if bas < participation_cible:
        msg = (f"Impossible : meme avec un cap a {_pct(bornes[0], 0)}, la participation "
               f"plafonne a {_pct(bas, 0)}.")
        return Reponse(False, "cap", bornes[0], bas, participation_cible, msg, pts)
    a, b = bornes[0], bornes[1]          # f(a) >= cible > f(b) ; la participation baisse avec le cap
    for _ in range(40):
        mil = 0.5 * (a + b)
        if participation_cp(mk, T, protection, mil) >= participation_cible:
            a = mil
        else:
            b = mil
    p = participation_cp(mk, T, protection, a)
    msg = (f"Cap a {_pct(a)} pour {_pct(p, 0)} de participation : gain maximum de "
           f"{_pct(p * (a - 1.0))} du nominal.")
    return Reponse(True, "cap", a, p, participation_cible, msg, pts)


# =====================================================================
# 2. OU VA L'ARGENT
# =====================================================================

@dataclass
class Ligne:
    cle: str
    libelle: str
    montant: float        # EUR. Negatif = ce que le client apporte (option vendue, risque porte)
    explication: str


@dataclass
class Cascade:
    """
    Decomposition du nominal. La somme des lignes vaut exactement le nominal.
    resultat : coupon par an (BRC), coupon par observation (autocall) ou
    participation (Capital Protection).
    infos["cout_retro_par_pt"] : ce que coute au client 1 % de retrocession
    par an, en coupon annuel (ou en participation) : la reponse a « pourquoi
    deux reseaux proposent des coupons differents sur le meme produit ».
    """
    produit: str
    nominal: float
    resultat: float
    lignes: list
    infos: dict

    @property
    def somme(self):
        return sum(l.montant for l in self.lignes)

    def table(self):
        return [{"Composante": l.libelle, "EUR": l.montant,
                 "% du nominal": 100.0 * l.montant / self.nominal} for l in self.lignes]


def _frais(mk, T, autres):
    """Marge (residuelle, pour retomber au centime sur le nominal) et retrocession."""
    N = mk.nominal
    retro = mk.retro * T * N
    marge = N - autres - retro
    return [
        Ligne("marge", "Marge banque", marge,
              "Prelevee d'avance sur le budget option, invisible dans le payoff."),
        Ligne("retro", "Retrocession distributeur", retro,
              "Payee d'avance au reseau qui place le produit. Deux reseaux, deux retros, "
              "deux coupons differents sur le meme produit."),
    ]


def _composantes_brc(ch, sg, T_rest, mk, barriere, strike, continu, coupon_total, touchee, tau):
    """
    Valeur actuelle du BRC et de ses briques, sur des trajectoires qui
    partent du niveau du jour. coupon_total : coupons dus a l'echeance,
    en fraction du nominal (convention de la page : verses a l'echeance).
    """
    N, r = mk.nominal, mk.r
    B, K = barriere * S0, strike * S0
    ST = ch[:, -1].astype(np.float64)
    if touchee:
        p_no = np.zeros_like(ST)
    elif continu:
        p_no = M.proba_non_touche(ch, B, sg, T_rest)
    else:
        p_no = (ST > B).astype(np.float64)
    D = math.exp(-r * T_rest)
    perte = N * (1.0 - p_no) * np.maximum(1.0 - ST / K, 0.0)    # le put down-and-in
    sain = (N + coupon_total * N - perte) * D
    pv_sain = sain.mean()
    if tau is None:
        pv = pv_sain
    else:
        pv = np.where(tau > T_rest, sain,
                      M.pv_recouvrement(tau, T_rest, r, N, mk.recovery)).mean()
    return {"zc": N * D, "coupons": coupon_total * N * D, "put": float((perte * D).mean()),
            "credit": float(pv - pv_sain), "pv": float(pv),
            "proba_touche": float(1.0 - p_no.mean()),
            "proba_perte": float(((1.0 - p_no) * (ST < K)).mean())}


def _observations(T, par_an, t, n_steps):
    """Observations restantes vues d'aujourd'hui : [(delai, indice, derniere)]."""
    n_obs = _n_obs(T, par_an)
    dt_obs = T / n_obs
    if t == 0:
        idx = M.idx_obs(n_steps, n_obs)                 # exactement comme la page
        return [((k + 1) * dt_obs, idx[k], k == n_obs - 1) for k in range(n_obs)]
    T_rest = T - t
    obs = []
    for k in range(n_obs):
        d = (k + 1) * dt_obs
        if d > t + 1e-9:
            obs.append((d - t, int(round(n_steps * (d - t) / T_rest)), k == n_obs - 1))
    return obs


def _composantes_autocall(ch, T_rest, obs, mk, trigger, bar_coupon, bar_capital, memoire,
                          coupon, manques0, tau):
    """
    Valeur actuelle de l'autocall et de ses briques. Meme logique que
    moteur.pv_autocall, generalisee a une date en cours de vie : dates
    d'observation quelconques et coupons manques deja accumules (memoire).
    """
    N, r = mk.nominal, mk.r
    n = ch.shape[0]
    D = math.exp(-r * T_rest)
    ST = ch[:, -1].astype(np.float64)

    def passe(defaut):
        vivant = np.ones(n, dtype=bool)
        manques = np.full(n, float(manques0))
        pv_cp, pv_rappel = np.zeros(n), np.zeros(n)
        inf = np.full(n, np.inf) if defaut is None else defaut
        rappel_1 = 0.0
        for j, (t_k, i, derniere) in enumerate(obs):
            disc = math.exp(-r * t_k)
            solvable = inf > t_k
            S = ch[:, i].astype(np.float64)
            paye = vivant & solvable & (S >= bar_coupon * S0)
            n_cp = (manques + 1.0) if memoire else np.ones(n)
            pv_cp += np.where(paye, coupon * N * n_cp * disc, 0.0)
            manques = np.where(paye, 0.0, manques + 1.0)
            if not derniere:
                rappel = vivant & solvable & (S >= trigger * S0)
                if j == 0:
                    rappel_1 = rappel.mean()
                pv_rappel += np.where(rappel, N * disc, 0.0)
                vivant = vivant & ~rappel
        final = np.where(ST >= bar_capital * S0, N, N * ST / S0)
        pv = pv_cp + pv_rappel + np.where(vivant & (inf > T_rest), final * D, 0.0)
        if defaut is not None:
            pv = np.where(vivant & (defaut <= T_rest),
                          pv + M.pv_recouvrement(defaut, T_rest, r, N, mk.recovery), pv)
        return pv.mean(), pv_cp.mean(), pv_rappel.mean(), vivant, rappel_1

    pv_sain, coupons, rappels, vivant, rappel_1 = passe(None)
    pv = passe(tau)[0] if tau is not None else pv_sain
    perte = vivant & (ST < bar_capital * S0)
    put = (np.where(perte, N * (1.0 - ST / S0), 0.0) * D).mean()
    return {"zc": N * D, "coupons": float(coupons),
            "rappel": float(rappels - N * D * (1.0 - vivant.mean())),   # nominal rendu plus tot
            "put": float(put), "credit": float(pv - pv_sain), "pv": float(pv),
            "proba_perte": float(perte.mean()), "proba_rappel_prochaine": float(rappel_1)}


def cascade_brc(mk, T, barriere, strike=1.0, continu=True):
    """
    Ou vont les 1 000 EUR d'un BRC. Le client paie le nominal ; la banque
    place une obligation, verse les coupons, encaisse la prime du put que
    le client lui vend, se remunere par le risque de credit que le client
    porte, et preleve marge et retrocession.
    """
    N = mk.nominal
    ch, sg = mk.trajectoires(T, barriere)
    tau = mk.defauts(len(ch), T)
    cpn = float(M.coupon_brc(ch, T, mk.r, N, barriere, strike, continu, mk.marge + mk.retro,
                             sg, tau, mk.recovery))
    c = _composantes_brc(ch, sg, T, mk, barriere, strike, continu, cpn * T, False, tau)
    lignes = [
        Ligne("zc", "Obligation zero-coupon (taux sans risque)", c["zc"],
              "Ce qu'il faut placer aujourd'hui pour rendre le nominal a l'echeance."),
        Ligne("coupons", "Coupons verses au client", c["coupons"],
              "Valeur actuelle du coupon garanti."),
        Ligne("put", "Put down-and-in vendu par le client", -c["put"],
              "La prime d'assurance que la banque paie au client : c'est elle qui finance "
              "le coupon."),
        Ligne("credit", "Risque emetteur porte par le client", c["credit"],
              "Ce que vaut le risque de defaut de la banque. Plus le spread est eleve, plus "
              "cette ligne finance de coupon."),
    ]
    lignes += _frais(mk, T, sum(l.montant for l in lignes))
    unite = float(math.exp(-mk.r * T) * N * T if tau is None
                  else (math.exp(-mk.r * T) * N * T * (tau > T)).mean())
    infos = {"coupon_pa": cpn,
             "zc_taux_financement": N * math.exp(-(mk.r + mk.spread / 10000.0) * T),
             "marge_affichee": mk.marge * T * N,
             "cout_retro_par_pt": 0.01 * T * N / unite,
             "proba_touche": c["proba_touche"], "proba_perte": c["proba_perte"]}
    return Cascade("brc", N, cpn, lignes, infos)


def cascade_autocall(mk, T, par_an, trigger, bar_coupon, bar_capital, memoire=True,
                     n_actifs=1, correlation=0.6):
    """
    Ou vont les 1 000 EUR d'un autocall. Le client achete des coupons
    conditionnels et beneficie du remboursement anticipe ; il les paie en
    vendant un put down-and-in (la barriere de capital) et en portant le
    risque de l'emetteur.
    """
    N = mk.nominal
    n_obs = _n_obs(T, par_an)
    ch, _ = mk.panier(n_actifs, correlation, T, bar_capital)
    tau = mk.defauts(len(ch), T)
    cpn = float(M.coupon_autocall(ch, T, mk.r, N, n_obs, trigger, bar_coupon, bar_capital,
                                  memoire, mk.marge + mk.retro, tau, mk.recovery))
    # marge et retro prelevees sur la duree de vie attendue (comme moteur.coupon_autocall)
    duree = float(M.stats_autocall(ch, T, n_obs, trigger, bar_capital)[0])
    obs = _observations(T, par_an, 0.0, ch.shape[1] - 1)
    args = (ch, T, obs, mk, trigger, bar_coupon, bar_capital, memoire)
    c = _composantes_autocall(*args, cpn, 0, tau)
    lignes = [
        Ligne("zc", "Obligation zero-coupon (taux sans risque)", c["zc"],
              "Ce qu'il faut placer aujourd'hui pour rendre le nominal a l'echeance maximale."),
        Ligne("rappel", "Remboursement anticipe", c["rappel"],
              "Le nominal rendu plus tot quand le produit est rappele vaut plus que le meme "
              "nominal a l'echeance."),
        Ligne("coupons", "Coupons conditionnels achetes", c["coupons"],
              "Valeur des coupons, qui ne tombent que si le sous-jacent tient la barriere de "
              "coupon et s'arretent au rappel."),
        Ligne("put", "Put down-and-in vendu par le client", -c["put"],
              "La barriere de capital : le client assure la banque contre une forte baisse a "
              "l'echeance. C'est la principale source du coupon."),
        Ligne("credit", "Risque emetteur porte par le client", c["credit"],
              "Ce que vaut le risque de defaut de la banque."),
    ]
    lignes += _frais(mk, duree, sum(l.montant for l in lignes))
    pente = _composantes_autocall(*args, cpn + 0.01, 0, tau)["pv"] - c["pv"]
    infos = {"coupon_obs": cpn, "coupon_pa": cpn * par_an,
             "zc_taux_financement": N * math.exp(-(mk.r + mk.spread / 10000.0) * T),
             "marge_affichee": mk.marge * duree * N,
             "duree_vie": duree,
             "cout_retro_par_pt": (0.01 * duree * N / pente * 0.01 * par_an) if pente > 1e-9 else 0.0,
             "proba_perte": c["proba_perte"]}
    return Cascade("autocall", N, cpn, lignes, infos)


def cascade_cp(mk, T, protection=1.0, cap=3.0):
    """
    Ou vont les 1 000 EUR d'un Capital Protection. La banque place de quoi
    rendre la protection (au taux de financement de l'emetteur, ce qui
    revient a faire porter au client le risque de credit), achete un call
    et revend le call au cap, puis preleve marge et retrocession.
    """
    N = mk.nominal
    part = participation_cp(mk, T, protection, cap)
    call, call_cap, _ = _cout_call_spread(mk, T, cap)
    zc_rf = protection * N * math.exp(-mk.r * T)
    zc_fin = protection * N * math.exp(-(mk.r + mk.spread / 10000.0) * T)
    lignes = [
        Ligne("zc", "Obligation zero-coupon (taux sans risque)", zc_rf,
              f"Ce qu'il faut placer aujourd'hui pour rendre {_pct(protection, 0)} du nominal."),
        Ligne("credit", "Risque emetteur porte par le client", zc_fin - zc_rf,
              "La banque actualise a son propre taux de financement : la garantie coute moins "
              "cher, parce que le client porte le risque de defaut."),
        Ligne("call", f"Call 100% achete (x {_pct(part, 0)})", part * call * N,
              "Le moteur de la participation a la hausse."),
        Ligne("cap", "Call au cap vendu", -part * call_cap * N,
              "Le client renonce aux tres fortes hausses ; la prime revient dans le budget."),
    ]
    lignes += _frais(mk, T, sum(l.montant for l in lignes))
    infos = {"participation": part, "zc_taux_financement": zc_fin,
             "marge_affichee": mk.marge * T * N,
             "cout_retro_par_pt": 0.01 * T / _cout_call_spread(mk, T, cap)[2]}
    return Cascade("cp", N, part, lignes, infos)


# =====================================================================
# 3. MARCHE SECONDAIRE
# =====================================================================

@dataclass
class ValeurSecondaire:
    """
    valeur : valeur juste des flux restants (milieu de fourchette).
    prix_rachat / prix_vente : ce que la banque paie si le client revend,
    et ce qu'elle demande a un acheteur.
    valeur_emission : valeur juste au lancement = nominal - marge - retro.
    decomposition : [(facteur, effet en EUR)] ; valeur_emission + somme des
    effets = valeur, au centime.
    """
    valeur: float
    prix_rachat: float
    prix_vente: float
    valeur_emission: float
    coupons_encaisses: float
    decomposition: list
    infos: dict
    message: str
    nominal: float = 1000.0

    @property
    def bilan_client(self):
        """Ce que le client a gagne (ou perdu) s'il revend aujourd'hui."""
        return self.prix_rachat + self.coupons_encaisses - self.nominal


def _verifie_date(t, T):
    if not 0.0 <= t < T:
        raise ValueError("la date doit etre comprise entre l'emission et l'echeance")


def _decompose(valorise, mk_e, mk_j, t, niveau, etat, etat_neuf):
    """
    Effets successifs, toujours sur les memes tirages : chaque etape ne
    change qu'un facteur, la somme retombe exactement sur la valeur du jour.
    valorise(mk, t, niveau, etat) -> dict avec la cle "pv".
    """
    mk_v = _prend(mk_j, mk_e, CHAMPS_VOL)
    mk_vt = _prend(mk_j, mk_v, CHAMPS_TAUX)
    mk_vtc = _prend(mk_j, mk_vt, CHAMPS_CREDIT)
    v0 = valorise(mk_e, 0.0, 1.0, etat_neuf)
    etapes = [("Temps ecoule", mk_e, 1.0, etat_neuf),
              ("Sous-jacent", mk_e, niveau, etat),
              ("Volatilite", mk_v, niveau, etat),
              ("Taux et dividende", mk_vt, niveau, etat),
              ("Spread emetteur", mk_vtc, niveau, etat)]
    deco, prec, res = [], v0["pv"], v0
    for nom, mk, niv, et in etapes:
        res = valorise(mk, t, niv, et)
        deco.append((nom, res["pv"] - prec))
        prec = res["pv"]
    return v0["pv"], res, deco


def _resultat(valeur, v0, coupons, deco, infos, fourchette, N, debut):
    demi = 0.5 * fourchette * N
    effets = ", ".join(f"{nom.lower()} {e:+,.0f}" for nom, e in deco)
    msg = (f"{debut}Valeur {valeur:,.0f} EUR ({_pct(valeur / N)} du nominal). La banque rachete "
           f"a {valeur - demi:,.0f} EUR. Depuis l'emission ({v0:,.0f} EUR apres marge et "
           f"retro) : {effets}.")
    return ValeurSecondaire(valeur, valeur - demi, valeur + demi, v0, coupons, deco, infos,
                            msg, N)


def valeur_secondaire_brc(mk_emission, mk_jour, T, barriere, strike, continu, t, niveau,
                          touchee=False, fourchette=0.01, coupon=None):
    """
    Valeur d'un BRC en cours de vie.

    Pourquoi : un client qui veut sortir avant l'echeance revend a la
    banque. Le prix depend du chemin (barriere deja touchee ou non), du
    marche du jour et du temps restant. Au lancement, la valeur juste est
    deja le nominal moins la marge et la retro : revendre tout de suite
    coute ces frais plus la demi-fourchette.

    t : annees ecoulees depuis l'emission. niveau : sous-jacent du jour en
    fraction du niveau initial. touchee : barriere deja franchie (deduit
    automatiquement si le niveau est sous une barriere continue). coupon :
    coupon du term sheet, par defaut celui calcule a l'emission.
    """
    _verifie_date(t, T)
    if coupon is None:
        coupon = coupon_brc(mk_emission, T, barriere, strike, continu)
    touchee = bool(continu and (touchee or niveau <= barriere))
    N = mk_emission.nominal

    def valorise(mk, t_, niv, deja):
        T_rest = T - t_
        ch, sg = mk.trajectoires(T_rest, (strike if deja else barriere) / niv, spot=niv * S0)
        tau = mk.defauts(len(ch), T_rest)
        return _composantes_brc(ch, sg, T_rest, mk, barriere, strike, continu, coupon * T,
                                deja, tau)

    v0, res, deco = _decompose(valorise, mk_emission, mk_jour, t, niveau, touchee, False)
    infos = {"coupon_pa": coupon, "touchee": touchee,
             "proba_touche": 1.0 if touchee else res["proba_touche"],
             "proba_perte": res["proba_perte"]}
    return _resultat(res["pv"], v0, 0.0, deco, infos, fourchette, N, "")


def etat_autocall(niveaux_observes, trigger, bar_coupon, memoire=True):
    """
    Rejoue les observations deja passees (niveaux en fraction de l'initial)
    pour savoir ou en est le produit : rappele ou non, coupons verses,
    coupons manques en attente (effet memoire).
    """
    manques, verses = 0, 0
    for k, s in enumerate(niveaux_observes):
        if s >= bar_coupon:
            verses += (manques + 1) if memoire else 1
            manques = 0
        else:
            manques += 1
        if s >= trigger:
            return {"vivant": False, "rappel_obs": k + 1, "manques": 0, "coupons_verses": verses}
    return {"vivant": True, "rappel_obs": None, "manques": manques, "coupons_verses": verses}


def valeur_secondaire_autocall(mk_emission, mk_jour, T, par_an, trigger, bar_coupon,
                               bar_capital, memoire, t, niveau, niveaux_observes=None,
                               manques=0, coupons_verses=0, n_actifs=1, correlation=0.6,
                               fourchette=0.01, coupon=None):
    """
    Valeur d'un autocall en cours de vie.

    Pourquoi : la valeur depend des observations deja passees (un coupon
    verse est sorti du produit, un coupon manque peut revenir grace a la
    memoire) et de la distance au trigger a la prochaine date : juste
    avant une observation, un sous-jacent au-dessus du trigger fait coller
    la valeur au nominal plus le coupon.

    niveaux_observes : niveaux du sous-jacent aux observations deja passees
    (en fraction de l'initial). S'ils sont donnes, l'etat (rappele, coupons
    verses, coupons manques) en est deduit ; sinon on prend manques et
    coupons_verses. Worst-of : tous les sous-jacents sont supposes au
    niveau du plus mauvais (hypothese prudente, qui sous-estime la valeur).
    """
    _verifie_date(t, T)
    if coupon is None:
        coupon = coupon_autocall(mk_emission, T, par_an, trigger, bar_coupon, bar_capital,
                                 memoire, n_actifs, correlation)
    N = mk_emission.nominal
    n_obs = _n_obs(T, par_an)
    passees = sum(1 for k in range(n_obs) if (k + 1) * (T / n_obs) <= t + 1e-9)
    if niveaux_observes is not None:
        if len(niveaux_observes) != passees:
            raise ValueError(f"{passees} observation(s) passee(s) a cette date, "
                             f"{len(niveaux_observes)} niveau(x) fourni(s)")
        etat = etat_autocall(niveaux_observes, trigger, bar_coupon, memoire)
        if not etat["vivant"]:
            cp = etat["coupons_verses"] * coupon * N
            msg = (f"Produit rappele a l'observation {etat['rappel_obs']} : nominal rendu, "
                   f"{cp:,.0f} EUR de coupons encaisses. Il n'y a plus rien a racheter.")
            ch_e, _ = mk_emission.panier(n_actifs, correlation, T, bar_capital)
            vie = float(M.stats_autocall(ch_e, T, n_obs, trigger, bar_capital)[0])
            del ch_e
            return ValeurSecondaire(0.0, 0.0, 0.0, N * (1 - (mk_emission.marge
                                    + mk_emission.retro) * vie), cp, [], etat, msg, N)
        manques, coupons_verses = etat["manques"], etat["coupons_verses"]

    def valorise(mk, t_, niv, attente):
        T_rest = T - t_
        ch, _ = mk.panier(n_actifs, correlation, T_rest, bar_capital / niv, spot=niv * S0)
        tau = mk.defauts(len(ch), T_rest)
        obs = _observations(T, par_an, t_, ch.shape[1] - 1)
        return _composantes_autocall(ch, T_rest, obs, mk, trigger, bar_coupon, bar_capital,
                                     memoire, coupon, attente, tau)

    v0, res, deco = _decompose(valorise, mk_emission, mk_jour, t, niveau, manques, 0)
    cp = coupons_verses * coupon * N
    infos = {"coupon_obs": coupon, "observations_passees": passees,
             "coupons_manques": manques, "proba_perte": res["proba_perte"],
             "proba_rappel_prochaine": res["proba_rappel_prochaine"]}
    debut = f"{cp:,.0f} EUR de coupons deja encaisses. " if cp else ""
    return _resultat(res["pv"], v0, cp, deco, infos, fourchette, N, debut)


def valeur_secondaire_cp(mk_emission, mk_jour, T, protection, cap, t, niveau,
                         fourchette=0.01, participation=None):
    """
    Valeur d'un Capital Protection en cours de vie, en formule fermee comme
    la page : obligation au taux de financement + participation x call spread.

    Pourquoi : la garantie ne joue qu'a l'echeance. En cours de vie, une
    hausse des taux fait baisser l'obligation et le produit peut valoir
    nettement moins que la protection : le client « garanti » qui revend
    avant l'echeance peut perdre.
    """
    _verifie_date(t, T)
    if participation is None:
        participation = participation_cp(mk_emission, T, protection, cap)
    N = mk_emission.nominal

    def valorise(mk, t_, niv, _etat):
        T_rest = T - t_
        spot = niv * S0
        zc = protection * N * math.exp(-(mk.r + mk.spread / 10000.0) * T_rest)
        call = M.bs_call(spot, S0, T_rest, mk.r, mk.vol(1.0 / niv), mk.q)
        call_cap = M.bs_call(spot, cap * S0, T_rest, mk.r, mk.vol(cap / niv), mk.q)
        return {"pv": zc + participation * N * (call - call_cap) / S0, "zc": zc}

    v0, res, deco = _decompose(valorise, mk_emission, mk_jour, t, niveau, None, None)
    infos = {"participation": participation, "obligation": res["zc"],
             "options": res["pv"] - res["zc"]}
    return _resultat(res["pv"], v0, 0.0, deco, infos, fourchette, N, "")
