"""
Recuperation des donnees reelles, figees dans data/.

    python scripts/recupere_donnees.py             # tout
    python scripts/recupere_donnees.py --indice    # Euro Stoxx 50 seulement
    python scripts/recupere_donnees.py --options   # photo d'options Eurex seulement

L'appli ne va JAMAIS chercher de donnees en ligne. Ce script est lance a la
main, une fois ; son resultat (les CSV de data/) est committe. Il sert a
documenter d'ou vient chaque chiffre et a refaire la photo plus tard.

---------------------------------------------------------------------------
1. data/eurostoxx50.csv : Euro Stoxx 50 quotidien depuis le 31/12/1986
---------------------------------------------------------------------------
  prix        SX5E, indice des prix (celui que suivent les produits)
  net_return  SX5T, dividendes nets reinvestis (ce que touche un detenteur
              direct ; base des indices a decrement)
  vstoxx      V2TX, vol implicite 30 jours de l'Euro Stoxx 50 (depuis 1999)

Sources, par ordre de priorite :
  - STOXX, fichiers historiques officiels (gratuits, sans compte) :
      hbrbcpe.txt  prix SX5E        31/12/1986 -> 04/10/2016 (fichier fige)
      hbrbcte.txt  net return SX5T  31/12/1986 -> 04/10/2016 (fichier fige)
      h_3msx5e.txt, h_3msx5t.txt    les 3 derniers mois, officiels
      h_v2tx.txt   VSTOXX           04/01/1999 -> aujourd'hui
    Avant le lancement de l'indice (26/02/1998), les valeurs sont un
    historique recalcule par STOXX.
  - Yahoo Finance, ^STOXX50E : le prix entre octobre 2016 et le debut des
    3 derniers mois. Controle : egal au fichier STOXX a 1e-8 pres sur les
    3 derniers mois. Yahoo renvoie parfois 429 (trop de requetes) : le
    script reessaie.
  - Yahoo Finance, SXRT.DE : l'ETF iShares Core EURO STOXX 50 (Acc), cote
    sur Xetra, dividendes reinvestis. Il comble le net return entre 2016 et
    les 3 derniers mois, car l'historique complet de SX5T / SX5GT demande un
    compte STOXX payant. L'ETF fait un peu mieux que SX5T (il recupere une
    partie des retenues a la source) : on retire une derive constante,
    calee pour que la serie retombe exactement sur la valeur officielle
    SX5T au debut des 3 derniers mois. Controle : la meme derive mesuree
    independamment sur 2010-2016 donne le meme ordre de grandeur (~0,5 %/an).

La version dividendes BRUTS (SX5GT) n'existe en acces libre que sur 3 mois :
on utilise donc le net return. Un indice a decrement reconstruit sur le net
est legerement plus conservateur (~0,5 %/an) qu'un decrement sur le brut.

---------------------------------------------------------------------------
2. data/options_eurostoxx50.csv : photo de prix d'options Eurex OESX
---------------------------------------------------------------------------
Prix de reglement officiels (daily settlement) des options europeennes sur
l'Euro Stoxx 50, via l'API publique du site Eurex :
    https://www.eurex.com/api/v1/overallstatistics/69660?...
Echeances retenues : les 2 premieres mensuelles, les 2 trimestrielles
suivantes, puis chaque decembre jusqu'a ~5 ans. Strikes : de 60 % a 130 %
du spot par pas de 5 %, option hors de la monnaie seulement (put sous le
forward, call au-dessus), prix >= 1 point.
Forward et facteur d'actualisation de chaque echeance : implicites, par
regression de la parite call-put C - P = DF * (F - K) sur les strikes a
+/- 10 % du spot. Ils contiennent donc le taux ET les dividendes attendus
par le marche, sans hypothese de notre part.
"""

import argparse
import datetime as dt
import json
import math
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
DATA = os.path.join(RACINE, "data")

import calibration as C  # noqa: E402  (inversion Black-76 vectorisee)

UA = "Mozilla/5.0"
STOXX = "https://www.stoxx.com/document/Indices/Current/HistoricalData/"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
EUREX = "https://www.eurex.com/api/v1/overallstatistics/69660"
DATE_RACCORD = dt.date(2016, 10, 4)          # dernier jour des fichiers STOXX figes


# ---------------------------------------------------------------------
# Telechargement
# ---------------------------------------------------------------------

def telecharge(url, essais=6):
    """
    Renvoie le contenu (bytes). Reessaie sur 429 / 5xx. Si le Python local
    n'a pas les certificats racine (cas du Python de python.org sur Mac),
    passe par curl, qui utilise ceux du systeme. La verification TLS n'est
    jamais desactivee.
    """
    attente = 5
    for k in range(essais):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as rep:
                return rep.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and k < essais - 1:
                print(f"    {e.code} sur {url[:70]}... nouvel essai dans {attente}s")
                time.sleep(attente)
                attente *= 2
                continue
            raise
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLError):
                return _curl(url)
            raise
    raise RuntimeError(f"echec du telechargement : {url}")


def _curl(url):
    out = subprocess.run(["curl", "-sSL", "--fail", "--retry", "4", "--retry-delay", "5",
                          "-A", UA, "--max-time", "120", url],
                         check=True, capture_output=True)
    return out.stdout


# ---------------------------------------------------------------------
# Lecture des formats sources
# ---------------------------------------------------------------------

def _date_stoxx(s):
    return dt.datetime.strptime(s.strip(), "%d.%m.%Y").date()


def stoxx_multi(texte, colonne):
    """Fichiers hbrbc*.txt : plusieurs indices par ligne, separateur ';'."""
    out = {}
    for ligne in texte.splitlines():
        p = [x.strip() for x in ligne.split(";")]
        if len(p) <= colonne or not p[0][:2].isdigit():
            continue
        try:
            v = float(p[colonne])
        except ValueError:
            continue
        if v > 0:
            out[_date_stoxx(p[0])] = v
    return out


def stoxx_simple(texte):
    """Fichiers h_*.txt : Date;Symbol;Indexvalue."""
    out = {}
    for ligne in texte.splitlines():
        p = ligne.strip().split(";")
        if len(p) >= 3 and p[0][:2].isdigit():
            try:
                v = float(p[2])
            except ValueError:
                continue
            if v > 0:
                out[_date_stoxx(p[0])] = v
    return out


def yahoo(symbole):
    fin = int(time.time())
    url = f"{YAHOO}{symbole}?period1=946684800&period2={fin}&interval=1d"
    j = json.loads(telecharge(url))["chart"]["result"][0]
    decal = j["meta"].get("gmtoffset", 0)
    out = {}
    for t, v in zip(j["timestamp"], j["indicators"]["quote"][0]["close"]):
        if v is None or v <= 0:
            continue
        d = dt.datetime.fromtimestamp(t + decal, tz=dt.timezone.utc).date()
        out[d] = float(v)
    return out


# ---------------------------------------------------------------------
# 1. Indice
# ---------------------------------------------------------------------

def construit_indice():
    print("Euro Stoxx 50 :")
    print("  STOXX, fichiers historiques officiels")
    prix_hist = stoxx_multi(telecharge(STOXX + "hbrbcpe.txt").decode("latin-1"), 2)
    nr_hist = stoxx_multi(telecharge(STOXX + "hbrbcte.txt").decode("latin-1"), 2)
    prix_3m = stoxx_simple(telecharge(STOXX + "h_3msx5e.txt").decode("latin-1"))
    nr_3m = stoxx_simple(telecharge(STOXX + "h_3msx5t.txt").decode("latin-1"))
    vstoxx = stoxx_simple(telecharge(STOXX + "h_v2tx.txt").decode("latin-1"))
    print("  Yahoo Finance : ^STOXX50E et SXRT.DE")
    prix_y = yahoo("%5ESTOXX50E")
    time.sleep(2)
    etf = yahoo("SXRT.DE")

    d0 = DATE_RACCORD
    d1 = min(prix_3m)                      # debut des 3 derniers mois officiels
    assert d0 in prix_hist and d0 in nr_hist, "fichier STOXX fige inattendu"
    assert d1 in nr_3m and d1 in etf and d1 in prix_y, "date de raccord absente"

    # Controles de coherence entre sources
    e = abs(prix_y[d0] / prix_hist[d0] - 1)
    assert e < 2e-3, f"Yahoo et STOXX divergent au raccord ({e:.2%})"
    communs = [d for d in prix_3m if d in prix_y]
    e3 = max(abs(prix_y[d] / prix_3m[d] - 1) for d in communs)
    assert e3 < 5e-4, f"Yahoo et STOXX divergent sur les 3 derniers mois ({e3:.2%})"

    # Prix : officiel -> Yahoo -> officiel
    prix = {d: v for d, v in prix_hist.items() if d <= d0}
    prix.update({d: v for d, v in prix_y.items() if d0 < d < d1})
    prix.update(prix_3m)

    # Net return entre d0 et d1 : prix * (ETF / prix), derive retiree
    jours = sorted(d for d in prix if d0 <= d <= d1)
    ratio, dernier = {}, etf[d0] / prix[d0]
    for d in jours:
        if d in etf:
            dernier = etf[d] / prix[d]
        ratio[d] = dernier
    brut = {d: prix[d] * ratio[d] for d in jours}
    ans = (d1 - d0).days / 365.25
    derive = math.log(nr_hist[d0] * brut[d1] / brut[d0] / nr_3m[d1]) / ans

    # Controle independant : meme derive mesuree sur 2010-2016 (ETF vs SX5T officiel)
    rec = sorted(d for d in etf if d in nr_hist)
    a, b = rec[0], rec[-1]
    ans_r = (b - a).days / 365.25
    derive_2010 = math.log(etf[b] / etf[a]) / ans_r - math.log(nr_hist[b] / nr_hist[a]) / ans_r
    print(f"  derive ETF vs SX5T : {derive*100:.2f} %/an (calee), "
          f"{derive_2010*100:.2f} %/an (mesuree {a.year}-{b.year})")
    assert abs(derive - derive_2010) < 0.004, "derive ETF incoherente : verifier les sources"

    nr = {d: v for d, v in nr_hist.items() if d <= d0}
    for d in jours:
        t = (d - d0).days / 365.25
        nr[d] = nr_hist[d0] * brut[d] / brut[d0] * math.exp(-derive * t)
    nr.update(nr_3m)

    # Assemblage : jours ou les deux series existent, sans les jours feries
    # (le fichier STOXX repete la valeur de la veille les 1er janvier, etc.)
    dates = sorted(d for d in prix if d in nr)
    lignes, prec = [], None
    for d in dates:
        cle = (round(prix[d], 2), round(nr[d], 2))
        if cle == prec:
            continue
        prec = cle
        v = vstoxx.get(d)
        lignes.append(f"{d.isoformat()},{prix[d]:.2f},{nr[d]:.2f},"
                      + (f"{v:.2f}" if v else ""))

    chemin = os.path.join(DATA, "eurostoxx50.csv")
    entete = [
        "# Euro Stoxx 50, clotures quotidiennes",
        "# prix       : SX5E (indice des prix)",
        "# net_return : SX5T (dividendes nets reinvestis)",
        "# vstoxx     : V2TX (vol implicite 30 jours, depuis 1999)",
        f"# Periode    : {dates[0].isoformat()} -> {dates[-1].isoformat()}",
        "# Sources    : STOXX fichiers historiques officiels hbrbcpe.txt / hbrbcte.txt "
        f"(-> {d0.isoformat()}), h_3msx5e.txt / h_3msx5t.txt ({d1.isoformat()} ->), "
        "h_v2tx.txt ;",
        "#              Yahoo Finance ^STOXX50E (prix entre les deux) ; ETF iShares "
        f"SXRT.DE pour le net return entre les deux (derive {derive*100:.2f} %/an retiree)",
        "# Avant le 26/02/1998 : historique recalcule par STOXX (lancement de l'indice)",
        f"# Recupere le {dt.date.today().isoformat()} par scripts/recupere_donnees.py",
        "date,prix,net_return,vstoxx",
    ]
    with open(chemin, "w") as f:
        f.write("\n".join(entete + lignes) + "\n")
    print(f"  -> {chemin} : {len(lignes)} jours, {os.path.getsize(chemin)/1024:.0f} Ko")


# ---------------------------------------------------------------------
# 2. Options
# ---------------------------------------------------------------------

def _eurex(params):
    return json.loads(telecharge(EUREX + "?" + params))


def choisit_echeances(lignes, date_marche):
    """2 premieres mensuelles, 2 trimestrielles suivantes, puis les decembres."""
    mens = sorted(dt.datetime.strptime(r["date"], "%Y%m%d").date()
                  for r in lignes if r["contractType"] == "M")
    mens = [e for e in mens if (e - date_marche).days > 20]
    choix = mens[:2]
    trim = [e for e in mens if e.month in (3, 6, 9, 12) and e not in choix]
    choix += trim[:2]
    choix += [e for e in mens if e.month == 12 and e not in choix
              and (e - date_marche).days / 365.0 <= 5.5]
    return sorted(choix)


def construit_options(date_marche=None):
    print("Options Eurex OESX :")
    vue = _eurex("filtertype=overview" + (f"&busdate={date_marche}" if date_marche else ""))
    if date_marche is None:
        date_marche = dt.datetime.strptime(vue["header"]["tradingDates"][0][:10],
                                           "%d-%m-%Y").strftime("%Y%m%d")
        vue = _eurex(f"busdate={date_marche}&filtertype=overview")
    jour = dt.datetime.strptime(date_marche, "%Y%m%d").date()
    spot = round(vue["header"]["underlyingClosingPrice"], 2)
    echeances = choisit_echeances(vue["dataRows"], jour)
    print(f"  date de marche {jour}, spot {spot}, {len(echeances)} echeances")

    lignes = []
    grille = np.round(np.arange(0.60, 1.3001, 0.05), 2)
    for ech in echeances:
        time.sleep(1)
        det = _eurex(f"filtertype=detail&productdate={ech:%Y%m%d}&busdate={date_marche}"
                     "&contracttype=M")
        calls = {r["strike"]: r for r in det["dataRowsCall"] if r["dSettle"] > 0}
        puts = {r["strike"]: r for r in det["dataRowsPut"] if r["dSettle"] > 0}
        T = (ech - jour).days / 365.0

        # forward et actualisation implicites : C - P = DF * (F - K)
        ks = np.array([k for k in sorted(calls) if k in puts and abs(k / spot - 1) <= 0.10])
        y = np.array([calls[k]["dSettle"] - puts[k]["dSettle"] for k in ks])
        A = np.vstack([np.ones_like(ks), ks]).T
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        DF, F = -b, -a / b
        residu = np.abs(y - A @ np.array([a, b])).max()
        assert 0.7 < DF <= 1.0 and residu < 1.0, f"parite call-put douteuse sur {ech}"

        tous = sorted(set(calls) | set(puts))
        pris = set()
        for m in grille:
            k = min(tous, key=lambda x: abs(x - m * spot))
            if k in pris or abs(k / spot - m) > 0.02:
                continue
            est_call = k >= F
            src = calls if est_call else puts
            if k not in src or src[k]["dSettle"] < 1.0:
                continue
            pris.add(k)
            prix = src[k]["dSettle"]
            vol = C.vol_implicite_b76(np.array([prix]), F, np.array([k]), T, DF,
                                      np.array([est_call]))[0]
            lignes.append(f"{ech.isoformat()},{T:.4f},{'C' if est_call else 'P'},{k:.0f},"
                          f"{prix:.1f},{F:.2f},{DF:.5f},{vol:.4f},"
                          f"{int(src[k]['openInterest'])}")
        r_impl = -math.log(DF) / T
        q_impl = r_impl - math.log(F / spot) / T
        print(f"  {ech}  T={T:.2f}  F={F:.1f}  taux implicite {r_impl*100:.2f} %  "
              f"dividende implicite {q_impl*100:.2f} %")

    chemin = os.path.join(DATA, "options_eurostoxx50.csv")
    entete = [
        "# Options europeennes sur l'Euro Stoxx 50 (Eurex OESX), prix de reglement officiels",
        f"# date_marche: {jour.isoformat()}",
        f"# spot: {spot}",
        "# Source : API publique Eurex, https://www.eurex.com/api/v1/overallstatistics/69660",
        "# forward et actualisation : implicites par parite call-put (strikes a +/- 10 % du spot)",
        "# vol_implicite : Black-76 sur le forward ; option hors de la monnaie seulement",
        "# maturite en annees exact/365 ; prix et strike en points d'indice",
        f"# Recupere le {dt.date.today().isoformat()} par scripts/recupere_donnees.py",
        "echeance,maturite,type,strike,prix,forward,actualisation,vol_implicite,open_interest",
    ]
    with open(chemin, "w") as f:
        f.write("\n".join(entete + lignes) + "\n")
    print(f"  -> {chemin} : {len(lignes)} options, {os.path.getsize(chemin)/1024:.0f} Ko")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--indice", action="store_true")
    ap.add_argument("--options", action="store_true")
    ap.add_argument("--date", help="date de marche des options, AAAAMMJJ (defaut : la derniere)")
    args = ap.parse_args()
    tout = not (args.indice or args.options)
    os.makedirs(DATA, exist_ok=True)
    if tout or args.indice:
        construit_indice()
    if tout or args.options:
        construit_options(args.date)
