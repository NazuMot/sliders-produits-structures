"""
Photo du marche du jour : ecrit data/marche_du_jour.json.

    python scripts/recupere_marche_du_jour.py

Lance chaque jour ouvre par GitHub (.github/workflows/marche_du_jour.yml),
le lendemain de la seance vers 13 h 30 UTC : a cette heure, la cloture de la
veille, le VSTOXX, l'ESTR et la courbe BCE de la veille sont tous publies, et
toutes les donnees portent la meme date. L'appli ne va jamais chercher de
donnees en ligne : elle lit le dernier fichier committe (donnees_du_jour.py).

Sources (gratuites, sans compte) :
  - STOXX, fichiers officiels : h_3msx5e.txt (Euro Stoxx 50, prix, 3 mois),
    h_3msx5t.txt (net return SX5T, 3 mois), h_v2tx.txt (VSTOXX depuis 1999).
  - BCE, data-api.ecb.europa.eu : ESTR (EST), Euribor 3 mois (FM, moyenne
    mensuelle), taux de depot (FM, controle), courbe des emprunts d'Etat
    notes AAA de la zone euro (YC, taux zero-coupon 1, 2, 3 et 5 ans).
  - data/eurostoxx50.csv (deja dans le depot) : les niveaux d'il y a un an,
    pour le dividende net sur 12 mois, et l'historique des dividendes.
  - Yahoo Finance ^STOXX50E : seulement pour controler le niveau de l'indice.

Dividende : rendement net realise sur 12 mois = ecart de performance entre
SX5T (dividendes nets reinvestis) et SX5E (prix). Controle du 18/10/2026 :
2,1 % contre 2,1 a 2,4 % de dividende implicite dans les options Eurex.

Niveau habituel : mediane depuis le 06/09/2004 (debut de la courbe AAA de la
BCE) du VSTOXX, de la vol realisee 3 mois, du taux AAA 3 ans et du dividende
net sur 12 mois, avec le rang du jour dans cet historique.

Une source en panne n'arrete rien : le champ garde sa valeur du fichier
precedent (avec sa date) et il est liste dans "anciens". Le script sort en
erreur seulement si aucune source n'a repondu (le fichier n'est pas touche).
"""

import csv
import datetime as dt
import io
import json
import math
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FICHIER = os.path.join(RACINE, "data", "marche_du_jour.json")
HISTORIQUE = os.path.join(RACINE, "data", "eurostoxx50.csv")

UA = "Mozilla/5.0"
STOXX = "https://www.stoxx.com/document/Indices/Current/HistoricalData/"
BCE = "https://data-api.ecb.europa.eu/service/data/"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/%5ESTOXX50E?range=1mo&interval=1d"
DEBUT_HABITUEL = dt.date(2004, 9, 6)

SERIES_BCE = {
    "estr": "EST/B.EU000A2X2A25.WT",
    "euribor_3m": "FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
    "taux_depot_bce": "FM/D.U2.EUR.4F.KR.DFR.LEV",
    "taux_1_an": "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y",
    "taux_2_ans": "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
    "taux_3_ans": "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3Y",
    "taux_5_ans": "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y",
}

# Bornes de vraisemblance : une valeur hors bornes est refusee (source cassee)
BORNES = {
    "eurostoxx50": (1000, 20000), "vstoxx": (8, 100), "vol_realisee_1m": (2, 150),
    "vol_realisee_3m": (2, 150), "estr": (-1.5, 10), "euribor_3m": (-1.5, 10),
    "taux_depot_bce": (-1.5, 10), "taux_1_an": (-2, 10), "taux_2_ans": (-2, 10),
    "taux_3_ans": (-2, 10), "taux_5_ans": (-2, 10), "dividende": (0.3, 8),
}


# ---------------------------------------------------------------------
# Telechargement (meme methode que scripts/recupere_donnees.py)
# ---------------------------------------------------------------------

def telecharge(url, essais=4):
    attente = 5
    for k in range(essais):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as rep:
                return rep.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and k < essais - 1:
                time.sleep(attente)
                attente *= 2
                continue
            raise
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLError):
                out = subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-A", UA,
                                      "--max-time", "90", url], check=True, capture_output=True)
                return out.stdout
            if k < essais - 1:
                time.sleep(attente)
                attente *= 2
                continue
            raise
    raise RuntimeError(f"echec du telechargement : {url}")


def stoxx(nom):
    """Fichier STOXX Date;Symbole;Valeur -> liste triee de (date, valeur)."""
    out = {}
    for ligne in telecharge(STOXX + nom).decode("latin-1").splitlines():
        p = ligne.strip().split(";")
        if len(p) >= 3 and p[0][:2].isdigit():
            try:
                v = float(p[2])
                d = dt.datetime.strptime(p[0].strip(), "%d.%m.%Y").date()
            except ValueError:
                continue
            if v > 0:
                out[d] = v
    return sorted(out.items())


def bce(serie, debut=None, derniers=None):
    """Serie BCE -> liste triee de (periode texte, valeur)."""
    url = f"{BCE}{serie}?format=csvdata&detail=dataonly"
    if debut:
        url += f"&startPeriod={debut}"
    if derniers:
        url += f"&lastNObservations={derniers}"
    lignes = csv.DictReader(io.StringIO(telecharge(url).decode("utf-8")))
    out = [(r["TIME_PERIOD"], float(r["OBS_VALUE"])) for r in lignes if r.get("OBS_VALUE")]
    return sorted(out)


def yahoo():
    j = json.loads(telecharge(YAHOO))["chart"]["result"][0]
    decal = j["meta"].get("gmtoffset", 0)
    out = {}
    for t, v in zip(j["timestamp"], j["indicators"]["quote"][0]["close"]):
        if v:
            out[dt.datetime.fromtimestamp(t + decal, tz=dt.timezone.utc).date()] = float(v)
    return out


# ---------------------------------------------------------------------
# Calculs
# ---------------------------------------------------------------------

def vol_realisee(prix, n):
    """Vol annualisee des n derniers rendements quotidiens, en %."""
    v = [p for _, p in prix[-(n + 1):]]
    r = [math.log(b / a) for a, b in zip(v[:-1], v[1:])]
    moy = sum(r) / len(r)
    return math.sqrt(sum((x - moy) ** 2 for x in r) / (len(r) - 1) * 252) * 100


def mediane(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def rang(xs, x):
    """Part de l'historique strictement sous x."""
    return sum(1 for v in xs if v < x) / len(xs)


def charge_historique():
    """data/eurostoxx50.csv -> {date: (prix, net_return)}."""
    out = {}
    with open(HISTORIQUE) as f:
        for ligne in f:
            if ligne[0] in "#d":
                continue
            d, p, n, _ = ligne.rstrip("\n").split(",")
            out[dt.date.fromisoformat(d)] = (float(p), float(n))
    return out


def a_la_date(serie, d):
    """Derniere valeur connue a la date d (ou avant). serie : dict trie par date."""
    avant = [k for k in serie if k <= d]
    return serie[max(avant)] if avant else None


def dividendes_12m(hist):
    """Dividende net sur 12 mois, jour par jour : {date: % par an}."""
    dates = sorted(hist)
    out, j = {}, 0
    for d in dates:
        cible = d - dt.timedelta(days=365)
        if cible < dates[0]:
            continue
        while dates[j + 1] <= cible:
            j += 1
        p0, n0 = hist[dates[j]]
        p1, n1 = hist[d]
        out[d] = (math.log(n1 / n0) - math.log(p1 / p0)) * 100
    return out


# ---------------------------------------------------------------------
# Assemblage
# ---------------------------------------------------------------------

def champ(valeur, date, source, **autres):
    return {"valeur": round(valeur, 4), "date": str(date), "source": source, **autres}


def recupere():
    """Renvoie (champs, habituel, erreurs). Chaque source est essayee a part."""
    champs, habituel, erreurs = {}, {}, []

    def essaie(nom, f):
        try:
            f()
        except Exception as e:                       # une source en panne n'arrete rien
            erreurs.append(f"{nom} : {type(e).__name__} {str(e)[:150]}")

    sx5e, sx5t = [], []

    def indice():
        sx5e.extend(stoxx("h_3msx5e.txt"))
        d, v = sx5e[-1]
        champs["eurostoxx50"] = champ(v, d, STOXX + "h_3msx5e.txt")
        champs["vol_realisee_1m"] = champ(vol_realisee(sx5e, 21), d, STOXX + "h_3msx5e.txt",
                                          jours=21)
        champs["vol_realisee_3m"] = champ(vol_realisee(sx5e, min(63, len(sx5e) - 1)), d,
                                          STOXX + "h_3msx5e.txt", jours=min(63, len(sx5e) - 1))
        # niveau habituel de la vol realisee : pour comparer a armes egales sans VSTOXX
        hist = sorted((k, v[0]) for k, v in charge_historique().items())
        h = [vol_realisee(hist[:i + 1], 63) for i in range(64, len(hist))
             if hist[i][0] >= DEBUT_HABITUEL]
        x = champs["vol_realisee_3m"]["valeur"]
        habituel["vol_realisee_3m"] = {"mediane": round(mediane(h), 4), "rang": round(rang(h, x), 4),
                                       "n": len(h)}

    def controle_yahoo():
        if "eurostoxx50" not in champs:
            return
        y = yahoo()
        d = dt.date.fromisoformat(champs["eurostoxx50"]["date"])
        if d in y:
            champs["eurostoxx50"]["controle_yahoo"] = round(y[d], 2)

    def vstoxx():
        v = stoxx("h_v2tx.txt")
        d, x = v[-1]
        champs["vstoxx"] = champ(x, d, STOXX + "h_v2tx.txt")
        hist = [y for k, y in v if k >= DEBUT_HABITUEL]
        habituel["vstoxx"] = {"mediane": round(mediane(hist), 4), "rang": round(rang(hist, x), 4),
                              "n": len(hist)}

    def taux(nom, serie):
        def f():
            obs = bce(serie, derniers=1)
            if obs:
                champs[nom] = champ(obs[-1][1], obs[-1][0], BCE + serie)
        return f

    def taux_habituel():
        h = [v for _, v in bce(SERIES_BCE["taux_3_ans"], debut=DEBUT_HABITUEL.isoformat())]
        if h and "taux_3_ans" in champs:
            habituel["taux_3_ans"] = {"mediane": round(mediane(h), 4),
                                      "rang": round(rang(h, champs["taux_3_ans"]["valeur"]), 4),
                                      "n": len(h)}

    def dividende():
        sx5t.extend(stoxx("h_3msx5t.txt"))
        hist = charge_historique()
        d, n1 = sx5t[-1]
        p1 = dict(sx5e).get(d) if sx5e else None
        if p1 is None:
            raise ValueError("cloture SX5E absente pour la date du SX5T")
        il_y_a_un_an = d - dt.timedelta(days=365)
        if il_y_a_un_an < min(hist) or il_y_a_un_an > max(hist):
            raise ValueError("data/eurostoxx50.csv ne couvre plus la date d'il y a un an")
        p0, n0 = a_la_date(hist, il_y_a_un_an)
        x = (math.log(n1 / n0) - math.log(p1 / p0)) * 100
        champs["dividende"] = champ(x, d, STOXX + "h_3msx5t.txt + data/eurostoxx50.csv")
        h = [v for k, v in dividendes_12m(hist).items() if k >= DEBUT_HABITUEL]
        habituel["dividende"] = {"mediane": round(mediane(h), 4), "rang": round(rang(h, x), 4),
                                 "n": len(h)}

    essaie("Euro Stoxx 50 (STOXX)", indice)
    essaie("controle Yahoo", controle_yahoo)
    essaie("VSTOXX (STOXX)", vstoxx)
    for nom, serie in SERIES_BCE.items():
        essaie(f"{nom} (BCE)", taux(nom, serie))
    essaie("historique du taux 3 ans (BCE)", taux_habituel)
    essaie("dividende (STOXX)", dividende)
    return champs, habituel, erreurs


def valide(champs):
    """Retire les valeurs invraisemblables et renvoie la liste des refus."""
    refus = []
    for nom, (bas, haut) in BORNES.items():
        if nom in champs and not bas <= champs[nom]["valeur"] <= haut:
            refus.append(f"{nom} = {champs[nom]['valeur']} hors de [{bas}, {haut}]")
            del champs[nom]
    i = champs.get("eurostoxx50", {})
    if i.get("controle_yahoo") and abs(i["controle_yahoo"] / i["valeur"] - 1) > 0.005:
        refus.append(f"Euro Stoxx 50 : STOXX {i['valeur']} et Yahoo {i['controle_yahoo']} "
                     "divergent de plus de 0,5 %")
        del champs["eurostoxx50"]
    if "estr" in champs and "taux_depot_bce" in champs:
        if abs(champs["estr"]["valeur"] - champs["taux_depot_bce"]["valeur"]) > 0.30:
            refus.append("ESTR trop loin du taux de dépôt BCE : ESTR retiré")
            del champs["estr"]
    return refus


def assemble(champs, habituel, precedent):
    """Complete avec le fichier precedent ce qui manque aujourd'hui."""
    anciens = []
    for nom, v in (precedent.get("champs") or {}).items():
        if nom not in champs:
            champs[nom] = v
            anciens.append(nom)
    for nom, v in (precedent.get("habituel") or {}).items():
        habituel.setdefault(nom, v)
    dates = [c["date"] for c in champs.values() if len(c["date"]) == 10]
    date = champs["eurostoxx50"]["date"] if "eurostoxx50" in champs else max(dates)
    return {
        "date_donnees": date,
        "genere_le": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "habituel_depuis": DEBUT_HABITUEL.isoformat(),
        "anciens": sorted(anciens),
        "champs": dict(sorted(champs.items())),
        "habituel": dict(sorted(habituel.items())),
    }


def main():
    try:
        precedent = json.load(open(FICHIER, encoding="utf-8"))
    except (OSError, ValueError):
        precedent = {}
    champs, habituel, erreurs = recupere()
    erreurs += valide(champs)
    for e in erreurs:
        print("ATTENTION " + e)
    if not champs:
        print("Aucune source n'a répondu : fichier inchangé.")
        return 1
    sortie = assemble(champs, habituel, precedent)
    with open(FICHIER, "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"data/marche_du_jour.json : données du {sortie['date_donnees']}")
    for nom, c in sortie["champs"].items():
        print(f"  {nom:16} {c['valeur']:>10.3f}  {c['date']}"
              + ("  (ancienne valeur)" if nom in sortie["anciens"] else ""))
    for nom, h in sortie["habituel"].items():
        print(f"  habituel {nom:12} médiane {h['mediane']:.2f}, rang du jour {h['rang']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
