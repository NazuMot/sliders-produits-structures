"""
=======================================================================
BACKTEST SUR L'HISTORIQUE REEL DE L'EURO STOXX 50
Tom Uzan
=======================================================================

    python backtest.py        # resume des produits par defaut

La question que pose tout client, et a laquelle le Monte Carlo ne repond
pas : « si j'avais achete ce produit par le passe, que se serait-il
passe ? ». On lance le meme produit a chaque date d'emission possible
(chaque mois par defaut) et on le deroule sur les vraies clotures.

  - Autocall Phoenix : memes parametres que la page Autocall (maturite,
    frequence, trigger, barriere de coupon, barriere de capital, memoire).
    Quand a-t-il ete rappele, combien de coupons, quelle perte.
  - BRC : barriere observee chaque jour a la cloture (« continue » au sens
    des termsheets francaises) ou a l'echeance seulement.

Sous-jacent au choix :
  prix        l'Euro Stoxx 50 classique (SX5E), dividendes exclus
  decrement   indice a decrement reconstitue : dividendes nets reinvestis,
              moins 50 points par an preleves chaque jour (base 1000 a la
              date d'emission, soit 5 % la premiere annee). C'est le
              sous-jacent de la plupart des autocalls vendus en France :
              le prelevement forfaitaire est connu d'avance, la banque
              n'a plus de risque de dividende et peut payer un coupon
              plus eleve. Le prix a payer : si les dividendes reels sont
              inferieurs au prelevement, l'indice decroche du marche.
  net_return  dividendes nets reinvestis, sans prelevement

Comparaison : la detention directe de l'indice dividendes reinvestis
(net return) sur exactement la meme periode que le produit.

Biais a connaitre : seuls les produits arrives a leur terme (ou rappeles)
avant la derniere date de l'historique sont comptes. Les emissions
mensuelles se chevauchent : 300 emissions ne sont pas 300 experiences
independantes, quelques grandes periodes de marche dominent.

Aucune dependance a streamlit ; uniquement numpy. Tout tourne en
quelques dizaines de millisecondes.
=======================================================================
"""

import os
from types import SimpleNamespace

import numpy as np

RACINE = os.path.dirname(os.path.abspath(__file__))
FICHIER_INDICE = os.path.join(RACINE, "data", "eurostoxx50.csv")
SOUS_JACENTS = ("prix", "decrement", "net_return")


# ---------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------

def charge_historique(chemin=FICHIER_INDICE):
    """
    Historique quotidien : dates (datetime64[D]), prix, net_return, vstoxx
    (nan avant 1999), plus le cumul qui sert a l'indice a decrement.
    """
    dates, prix, nr, vs = [], [], [], []
    with open(chemin) as f:
        for ligne in f:
            if ligne[0] in "#d":                 # commentaires et en-tete
                continue
            d, p, n, v = ligne.rstrip("\n").split(",")
            dates.append(d)
            prix.append(float(p))
            nr.append(float(n))
            vs.append(float(v) if v else np.nan)
    return historique(dates, prix, nr, vs)


def historique(dates, prix, net_return, vstoxx=None):
    """Assemble un historique a partir de series (sert aussi aux tests sur series fictives)."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    nr = np.asarray(net_return, float)
    vs = np.full(len(dates), np.nan) if vstoxx is None else np.asarray(vstoxx, float)
    # A_t = somme des (jours / 365) / TR_s : voir _perf, cas decrement
    jours = np.diff(dates).astype(float)
    cumul = np.concatenate([[0.0], np.cumsum(jours / 365.0 / nr[1:])])
    return SimpleNamespace(dates=dates, prix=np.asarray(prix, float), net_return=nr,
                           vstoxx=vs, cumul_decrement=cumul,
                           debut=dates[0], fin=dates[-1])


def _perf(h, nature, i0, idx, points=50.0, base=1000.0):
    """
    Performance du sous-jacent entre l'emission i0 (n,) et les dates idx
    (n, m) : 1,0 = niveau initial.

    Indice a decrement, prelevement quotidien au prorata (jours / 365) :
        D_t = D_{t-1} * TR_t / TR_{t-1} - points * jours / 365
    En divisant par TR_t, la recurrence devient une somme :
        D_t / TR_t = D_i / TR_i - points * (A_t - A_i)
    d'ou un calcul vectorise pour toutes les emissions a la fois.
    Le prelevement pese 50 / niveau de l'indice : 5 % a 1000, 10 % a 500.
    Quand l'indice baisse, le decrement accelere la baisse.
    """
    i0 = np.asarray(i0)[:, None]
    if nature == "prix":
        return h.prix[idx] / h.prix[i0]
    tr = h.net_return
    if nature == "net_return":
        return tr[idx] / tr[i0]
    if nature == "decrement":
        A = h.cumul_decrement
        perf = tr[idx] / tr[i0] - (points / base) * tr[idx] * (A[idx] - A[i0])
        # Un prelevement fixe en points peut faire tomber l'indice a zero si
        # les dividendes reels restent durablement en dessous : lance a 1000
        # en janvier 2000, il touche zero en decembre 2014. On s'arrete a zero.
        return np.maximum(perf, 0.0)
    raise ValueError(f"sous-jacent inconnu : {nature} (choix : {SOUS_JACENTS})")


def indice_decrement(h, date_base, points=50.0, base=1000.0):
    """Indice a decrement complet depuis date_base (valeur base ce jour-la), pour le tracer."""
    i = int(np.searchsorted(h.dates, np.datetime64(date_base)))
    idx = np.arange(i, len(h.dates))[None, :]
    return h.dates[i:], base * _perf(h, "decrement", [i], idx, points, base)[0]


def dates_emission(h, debut="2000-01-01", fin=None, pas="mensuel"):
    """Indices des dates d'emission : premier jour de cotation de chaque mois / semaine, ou chaque jour."""
    d = h.dates
    garde = d >= np.datetime64(debut)
    if fin is not None:
        garde &= d <= np.datetime64(fin)
    pos = np.nonzero(garde)[0]
    if pas == "quotidien":
        return pos
    if pas == "mensuel":
        cle = d[pos].astype("datetime64[M]").astype(int)
    elif pas == "hebdomadaire":
        cle = (d[pos] - np.datetime64("1970-01-05")).astype(int) // 7
    else:
        raise ValueError("pas : mensuel, hebdomadaire ou quotidien")
    _, premier = np.unique(cle, return_index=True)
    return pos[premier]


def _calendrier(h, i0, mois):
    """
    Date d'observation = date d'emission + n mois (meme quantieme, borne a
    la fin du mois), puis premier jour de cotation a partir de cette date.
    Renvoie les indices (n_emissions, n_dates) ; len(h.dates) si la date
    tombe apres la fin de l'historique.
    """
    d0 = h.dates[i0]
    m0 = d0.astype("datetime64[M]")
    quantieme = (d0 - m0.astype("datetime64[D]")).astype(int)
    mo = m0[:, None] + np.asarray(mois)[None, :]
    debut_mois = mo.astype("datetime64[D]")
    fin_mois = (mo + 1).astype("datetime64[D]") - 1
    cible = np.minimum(debut_mois + quantieme[:, None], fin_mois)
    return np.searchsorted(h.dates, cible)


def _annees(h, i0, i1):
    return (h.dates[i1] - h.dates[i0]).astype(float) / 365.25


def _tri(flux, t):
    """
    Taux de rendement interne annualise de chaque ligne : le r tel que
    sum flux_k / (1 + r)^t_k = 1 (le nominal investi). Dichotomie
    vectorisee : la valeur actuelle baisse quand r monte.
    """
    lo = np.full(flux.shape[0], -0.99)
    hi = np.full(flux.shape[0], 3.0)
    for _ in range(70):
        r = 0.5 * (lo + hi)
        va = (flux * (1.0 + r[:, None]) ** (-t)).sum(axis=1)
        lo = np.where(va > 1.0, r, lo)
        hi = np.where(va > 1.0, hi, r)
    return 0.5 * (lo + hi)


def _direct(h, i0, i1):
    """Detention directe, dividendes nets reinvestis, sur la meme periode (rendement annualise)."""
    ans = np.maximum(_annees(h, i0, i1), 1e-6)
    nr = (h.net_return[i1] / h.net_return[i0]) ** (1.0 / ans) - 1.0
    px = (h.prix[i1] / h.prix[i0]) ** (1.0 / ans) - 1.0
    return nr, px


# ---------------------------------------------------------------------
# Autocall Phoenix
# ---------------------------------------------------------------------

def backtest_autocall(h, T=3.0, par_an=4, trigger=1.0, bar_cp=0.7, bar_cap=0.6,
                      memoire=True, coupon_pa=0.08, sous_jacent="prix",
                      debut="2000-01-01", fin=None, pas="mensuel", points=50.0):
    """
    Deroule un autocall lance a chaque date d'emission. Niveaux en
    fraction du niveau initial (1.0 = 100 %), coupon en % annuel : le
    coupon de chaque observation vaut coupon_pa / par_an (c'est
    l'« equivalent annuel » affiche par la page Autocall).

    Meme regle que moteur.pv_autocall, appliquee a des chemins reels :
      a chaque observation, coupon si >= barriere de coupon (+ coupons
      manques si memoire) ; rappel si >= trigger, sauf a la derniere ;
      a l'echeance, nominal si >= barriere de capital, sinon niveau final.

    Renvoie un SimpleNamespace : une valeur par emission (tableaux numpy)
    et un resume (voir resume_autocall).
    """
    n_obs = max(int(T * par_an), 1)
    c = coupon_pa / par_an
    mois = [int(round(12 * (k + 1) * T / n_obs)) for k in range(n_obs)]
    tous = dates_emission(h, debut, fin, pas)
    idx_tous = _calendrier(h, tous, mois)
    termine = idx_tous[:, -1] < len(h.dates)
    i0, idx = tous[termine], idx_tous[termine]
    n = len(i0)

    perf = _perf(h, sous_jacent, i0, idx, points)
    vivant = np.ones(n, dtype=bool)
    manques = np.zeros(n)
    coupons = np.zeros((n, n_obs))
    nb_coupons = np.zeros(n)
    k_fin = np.full(n, n_obs - 1)
    for k in range(n_obs):
        S = perf[:, k]
        paye = vivant & (S >= bar_cp)
        nb = (manques + 1.0) if memoire else np.ones(n)
        coupons[:, k] = np.where(paye, c * nb, 0.0)
        nb_coupons += np.where(paye, nb, 0.0)
        manques = np.where(paye, 0.0, manques + 1.0)
        if k < n_obs - 1:
            rappel = vivant & (S >= trigger)
            k_fin = np.where(rappel, k, k_fin)
            vivant &= ~rappel

    lignes = np.arange(n)
    i_fin = idx[lignes, k_fin]
    S_fin = perf[lignes, k_fin]
    capital = np.where(vivant & (S_fin < bar_cap), S_fin, 1.0)
    flux = coupons.copy()
    flux[lignes, k_fin] += capital
    t = _annees(h, i0[:, None], idx)
    direct, direct_prix = _direct(h, i0, i_fin)

    res = SimpleNamespace(
        produit="autocall", sous_jacent=sous_jacent, n_obs=n_obs,
        parametres=dict(T=T, par_an=par_an, trigger=trigger, bar_cp=bar_cp, bar_cap=bar_cap,
                        memoire=memoire, coupon_pa=coupon_pa, points=points),
        emission=h.dates[i0], fin=h.dates[i_fin],
        dates_obs=h.dates[idx],
        perf_obs=perf,                 # niveau du sous-jacent a chaque observation
        obs_fin=k_fin + 1,             # 1 = rappele a la premiere observation ; n_obs = echeance
        rappele=k_fin < n_obs - 1,
        coupons=coupons.sum(axis=1),   # somme des coupons touches, en % du nominal
        nb_coupons=nb_coupons,
        capital=capital,               # capital rendu, en fraction du nominal
        perte=capital < 1.0,
        perf_fin=S_fin,
        duree=_annees(h, i0, i_fin),
        tri=_tri(flux, t),
        direct=direct, direct_prix=direct_prix,
        vstoxx=h.vstoxx[i0],
        n_en_cours=int((~termine).sum()),
    )
    res.resume = resume_autocall(res)
    return res


def resume_autocall(res):
    n, n_obs = len(res.emission), res.n_obs
    if n == 0:
        return {"n": 0}
    fin_par_obs = np.array([(res.obs_fin == k + 1).mean() for k in range(n_obs)])
    a_terme = res.obs_fin == n_obs
    i_pire = int(np.argmin(res.tri))
    perte = res.perte
    return {
        "n": n,
        "premiere_emission": str(res.emission[0]),
        "derniere_emission": str(res.emission[-1]),
        # proportion d'emissions terminees a chaque observation (rappel), la
        # derniere case = alle a l'echeance (avec ou sans perte)
        "fin_par_observation": fin_par_obs,
        "rappel_premiere_date": float(fin_par_obs[0]),
        "rappel_anticipe": float(res.rappele.mean()),
        "echeance_sans_perte": float((a_terme & ~perte).mean()),
        "perte": float(perte.mean()),
        "perte_moyenne_si_perte": float(1 - res.capital[perte].mean()) if perte.any() else 0.0,
        "duree_moyenne": float(res.duree.mean()),
        "tri_moyen": float(res.tri.mean()),
        "tri_median": float(np.median(res.tri)),
        "tri_par_annee": _par_annee(res.tri, res.duree),
        "direct_moyen": float(res.direct.mean()),
        "direct_median": float(np.median(res.direct)),
        "direct_par_annee": _par_annee(res.direct, res.duree),
        "bat_direct": float((res.tri > res.direct).mean()),
        "pire_emission": str(res.emission[i_pire]),
        "pire_tri": float(res.tri[i_pire]),
        "pire_capital": float(res.capital[i_pire]),
        "n_en_cours": res.n_en_cours,
    }


# ---------------------------------------------------------------------
# Barrier Reverse Convertible
# ---------------------------------------------------------------------

def backtest_brc(h, T=1.0, barriere=0.7, strike=1.0, continu=True, coupon_pa=0.07,
                 sous_jacent="prix", debut="2000-01-01", fin=None, pas="mensuel",
                 points=50.0):
    """
    Deroule un BRC lance a chaque date d'emission. Coupon garanti, verse a
    l'echeance avec le capital (comme moteur.pv_brc). Capital rendu :
    100 % sauf si la barriere a ete touchee (cloture <= barriere, chaque
    jour si continu, a l'echeance sinon) ET que le niveau final est sous
    le strike ; le client recoit alors niveau final / strike.
    """
    mois = int(round(12 * T))
    tous = dates_emission(h, debut, fin, pas)
    i_tous = _calendrier(h, tous, [mois])[:, 0]
    termine = i_tous < len(h.dates)
    i0, i1 = tous[termine], i_tous[termine]
    n = len(i0)

    S_fin = _perf(h, sous_jacent, i0, i1[:, None], points)[:, 0]
    if continu:
        plus_bas = np.empty(n)
        for j in range(n):
            chemin = _perf(h, sous_jacent, i0[j:j + 1],
                           np.arange(i0[j], i1[j] + 1)[None, :], points)[0]
            plus_bas[j] = chemin.min()
    else:
        plus_bas = S_fin
    touche = plus_bas <= barriere
    capital = np.where(touche & (S_fin < strike), S_fin / strike, 1.0)
    duree = _annees(h, i0, i1)
    total = capital + coupon_pa * T
    direct, direct_prix = _direct(h, i0, i1)

    res = SimpleNamespace(
        produit="brc", sous_jacent=sous_jacent,
        parametres=dict(T=T, barriere=barriere, strike=strike, continu=continu,
                        coupon_pa=coupon_pa, points=points),
        emission=h.dates[i0], fin=h.dates[i1],
        plus_bas=plus_bas, perf_fin=S_fin,
        touche=touche, capital=capital, perte=capital < 1.0,
        duree=duree,
        tri=total ** (1.0 / duree) - 1.0,
        direct=direct, direct_prix=direct_prix,
        vstoxx=h.vstoxx[i0],
        n_en_cours=int((~termine).sum()),
    )
    res.resume = resume_brc(res)
    return res


def resume_brc(res):
    n = len(res.emission)
    if n == 0:
        return {"n": 0}
    i_pire = int(np.argmin(res.capital))
    perte = res.perte
    return {
        "n": n,
        "premiere_emission": str(res.emission[0]),
        "derniere_emission": str(res.emission[-1]),
        "barriere_touchee": float(res.touche.mean()),
        "perte": float(perte.mean()),
        "perte_moyenne_si_perte": float(1 - res.capital[perte].mean()) if perte.any() else 0.0,
        "tri_moyen": float(res.tri.mean()),
        "tri_median": float(np.median(res.tri)),
        "tri_par_annee": _par_annee(res.tri, res.duree),
        "direct_moyen": float(res.direct.mean()),
        "direct_median": float(np.median(res.direct)),
        "direct_par_annee": _par_annee(res.direct, res.duree),
        "bat_direct": float((res.tri > res.direct).mean()),
        "pire_emission": str(res.emission[i_pire]),
        "pire_capital": float(res.capital[i_pire]),
        "pire_tri": float(res.tri[i_pire]),
        "n_en_cours": res.n_en_cours,
    }


# ---------------------------------------------------------------------
# Resume en clair
# ---------------------------------------------------------------------

def _par_annee(taux, duree):
    """
    Rendement par annee investie : moyenne geometrique ponderee par la
    duree. La moyenne simple des rendements annualises est trompeuse sur un
    autocall : un rappel apres 3 mois avec +5 % d'indice compte pour +21 %
    par an cote detention directe. Ici, 3 mois pesent 3 mois.
    """
    return float(np.exp((np.log1p(taux) * duree).sum() / duree.sum()) - 1.0)


def _pct(x):
    return f"{x * 100:.1f} %"


def affiche_resume(res):
    r = res.resume
    p = res.parametres
    if res.produit == "autocall":
        print(f"Autocall {p['T']:g} ans, {res.n_obs} observations, trigger {p['trigger']:.0%}, "
              f"coupon {p['bar_cp']:.0%}, capital {p['bar_cap']:.0%}, "
              f"{p['coupon_pa']:.1%} p.a., sous-jacent {res.sous_jacent}")
        print(f"  {r['n']} emissions de {r['premiere_emission']} a {r['derniere_emission']} "
              f"({r['n_en_cours']} encore en cours, non comptees)")
        print("  fin a chaque observation :",
              " ".join(f"{v * 100:.0f}" for v in r["fin_par_observation"]), "(%)")
        print(f"  rappel a la 1re date {_pct(r['rappel_premiere_date'])} | rappel anticipe "
              f"{_pct(r['rappel_anticipe'])} | echeance sans perte {_pct(r['echeance_sans_perte'])}"
              f" | perte {_pct(r['perte'])} (moyenne {_pct(r['perte_moyenne_si_perte'])})")
    else:
        print(f"BRC {p['T']:g} an(s), barriere {p['barriere']:.0%} "
              f"{'continue' if p['continu'] else 'a l echeance'}, strike {p['strike']:.0%}, "
              f"{p['coupon_pa']:.1%} p.a., sous-jacent {res.sous_jacent}")
        print(f"  {r['n']} emissions de {r['premiere_emission']} a {r['derniere_emission']}")
        print(f"  barriere touchee {_pct(r['barriere_touchee'])} | perte {_pct(r['perte'])} "
              f"(moyenne {_pct(r['perte_moyenne_si_perte'])})")
    print(f"  rendement par annee investie {_pct(r['tri_par_annee'])} vs detention directe "
          f"{_pct(r['direct_par_annee'])} (medianes {_pct(r['tri_median'])} vs "
          f"{_pct(r['direct_median'])}) ; bat l'indice dans {_pct(r['bat_direct'])} des cas")
    print(f"  pire cas : emis le {r['pire_emission']}, capital rendu {_pct(r['pire_capital'])}, "
          f"rendement {_pct(r['pire_tri'])} par an")


if __name__ == "__main__":
    import time
    t = time.time()
    h = charge_historique()
    t_charge = time.time() - t
    print(f"Euro Stoxx 50 : {len(h.dates)} jours, {h.debut} -> {h.fin} "
          f"(lecture {t_charge * 1000:.0f} ms)\n")
    for sj in ("prix", "decrement"):
        t = time.time()
        a = backtest_autocall(h, sous_jacent=sj)
        b = backtest_brc(h, sous_jacent=sj)
        duree = time.time() - t
        affiche_resume(a)
        affiche_resume(b)
        print(f"  ({duree * 1000:.0f} ms pour les deux)\n")
