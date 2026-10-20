"""
Lecture du marche du jour (data/marche_du_jour.json), sans jamais planter.

Le fichier est ecrit chaque jour ouvre par scripts/recupere_marche_du_jour.py
(GitHub Actions, .github/workflows/marche_du_jour.yml). L'appli ne va jamais
chercher de donnees en ligne : elle lit le dernier fichier committe. Fichier
absent, illisible ou incomplet : chaque reglage retombe sur une source de
secours, puis sur la valeur par defaut de la barre laterale.

Aucun texte affiche ici (pas d'import streamlit) : les libelles sont dans
barre.py et vues/aujourdhui.py, traduits en anglais.
"""

import datetime as dt
import json
import math
import os

FICHIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                       "marche_du_jour.json")

# Reglages qui ne viennent pas du marche du jour : ceux de la photo Eurex du
# 16/10/2026 (scenario « Octobre 2026 ») pour le skew et le spread d'une banque.
SKEW, SPREAD, ECART_VOL_LT = 2.5, 50, 3.0
# Valeurs par defaut de la barre laterale, si aucune source n'est disponible
DEFAUT = {"taux": 3.0, "dividende": 2.5, "vol": 22.0}
# Medianes constatees du 06/09/2004 au 19/10/2026, si le fichier n'en donne pas
HABITUEL_DEFAUT = {"vstoxx": 19.6, "vol_realisee_3m": 16.5, "taux_3_ans": 1.1,
                   "dividende": 2.6}

# Sources par ordre de preference, pour chaque curseur
SOURCES_VOL = ("vstoxx", "vol_realisee_3m", "vol_realisee_1m")
SOURCES_TAUX = ("taux_3_ans", "taux_2_ans", "taux_5_ans", "estr", "euribor_3m")
SOURCES_DIVIDENDE = ("dividende",)


def charge(chemin=None):
    """
    Le contenu du fichier, ou {} s'il manque ou ne se lit pas. Ne leve jamais.
    Les tests pointent un autre fichier par la variable SLIDERS_MARCHE_DU_JOUR.
    """
    chemin = chemin or os.environ.get("SLIDERS_MARCHE_DU_JOUR") or FICHIER
    try:
        with open(chemin, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


def _nombre(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def valeur(d, nom):
    """Valeur d'un champ (en % ou en points), None si absente ou invalide."""
    c = (d.get("champs") or {}).get(nom) if isinstance(d.get("champs"), dict) else None
    return _nombre(c.get("valeur")) if isinstance(c, dict) else None


def date_champ(d, nom):
    c = (d.get("champs") or {}).get(nom) if isinstance(d.get("champs"), dict) else None
    return c.get("date") if isinstance(c, dict) and isinstance(c.get("date"), str) else None


def habituel(d, nom):
    """(mediane, rang du jour) depuis 2004 ; mediane par defaut si le fichier n'en a pas."""
    h = (d.get("habituel") or {}).get(nom) if isinstance(d.get("habituel"), dict) else None
    med = _nombre(h.get("mediane")) if isinstance(h, dict) else None
    rg = _nombre(h.get("rang")) if isinstance(h, dict) else None
    return (med if med is not None else HABITUEL_DEFAUT.get(nom)), rg


def date_fr(iso):
    """'2026-10-19' -> '19/10/2026', '2026-09' -> '09/2026', None -> None."""
    if not isinstance(iso, str):
        return None
    p = iso[:10].split("-")
    return "/".join(reversed(p)) if len(p) in (2, 3) and all(x.isdigit() for x in p) else None


def date_donnees(d):
    """Date des donnees (cloture de l'Euro Stoxx 50), en datetime.date, ou None."""
    try:
        return dt.date.fromisoformat(str(d.get("date_donnees"))[:10])
    except ValueError:
        return None


def anciennete(d, aujourdhui=None):
    """Jours calendaires entre la date des donnees et aujourd'hui (None si inconnue)."""
    jour = date_donnees(d)
    return None if jour is None else ((aujourdhui or dt.date.today()) - jour).days


def _arrondi(x, pas, bas, haut):
    return float(min(max(round(x / pas) * pas, bas), haut))


def _premiere(d, sources):
    for nom in sources:
        v = valeur(d, nom)
        if v is not None:
            return nom, v
    return None, None


def _curseurs(vol, taux, dividende):
    """Valeurs exactes -> crans des curseurs de la barre laterale."""
    vol = _arrondi(vol, 1.0, 8.0, 70.0)
    return {"taux": _arrondi(taux, 0.25, 0.0, 7.0),
            "dividende": _arrondi(dividende, 0.25, 0.0, 7.0),
            "vol": vol, "skew": SKEW,
            "vol_lt": _arrondi(vol + ECART_VOL_LT, 1.0, 10.0, 50.0),
            "spread": SPREAD}


def reglages_du_jour(d):
    """
    Le scenario « Aujourd'hui » : valeurs des curseurs, et d'ou vient chacune
    (nom du champ, ou None = valeur par defaut). Vol : VSTOXX, sinon vol
    realisee 3 mois puis 1 mois. Taux : AAA 3 ans de la BCE (milieu de la gamme
    de maturites des produits), sinon 2 ans, 5 ans, ESTR, Euribor 3 mois.
    """
    o_vol, vol = _premiere(d, SOURCES_VOL)
    o_taux, taux = _premiere(d, SOURCES_TAUX)
    o_div, div = _premiere(d, SOURCES_DIVIDENDE)
    exact = {"vol": DEFAUT["vol"] if vol is None else vol,
             "taux": DEFAUT["taux"] if taux is None else taux,
             "dividende": DEFAUT["dividende"] if div is None else div}
    return {"valeurs": _curseurs(exact["vol"], exact["taux"], exact["dividende"]),
            "exact": exact,
            "origine": {"vol": o_vol, "taux": o_taux, "dividende": o_div},
            "date": d.get("date_donnees") if isinstance(d.get("date_donnees"), str) else None}


def reglages_habituels(d, origine_vol=None):
    """
    Le marche « habituel » : medianes depuis 2004. La vol de reference suit la
    source de la vol du jour (VSTOXX contre VSTOXX, realisee contre realisee),
    pour ne pas comparer une vol realisee a une vol implicite.
    """
    nom_vol = "vol_realisee_3m" if origine_vol in ("vol_realisee_3m", "vol_realisee_1m") \
        else "vstoxx"
    vol = habituel(d, nom_vol)[0]
    taux = habituel(d, "taux_3_ans")[0]
    div = habituel(d, "dividende")[0]
    return {"valeurs": _curseurs(vol, taux, div),
            "exact": {"vol": vol, "taux": taux, "dividende": div},
            "origine": {"vol": nom_vol, "taux": "taux_3_ans", "dividende": "dividende"},
            "depuis": d.get("habituel_depuis") if isinstance(d.get("habituel_depuis"), str)
            else "2004-09-06"}
