"""
Atelier Sales : les outils du Sales face au client (sales.py), plus un pitch.

Cinq sections, un seul produit a la fois (BRC, Autocall, Capital
Protection), une seule structure partagee :
  1. Le client demande : reverse enquiry, la contrainte du client donne la
     structure ; « Retenir » ecrit la reponse dans la structure.
  2. Ou va l'argent : sources et emplois du budget, cout de la retro, deux
     reseaux de distribution cote a cote.
  3. Revendre en cours de vie : valeur, prix de rachat, ce qui a fait bouger
     la valeur depuis l'emission.
  4. Pitch client : phrases construites sur des modeles, selon le profil et
     la vue de marche du client. Toujours le scenario de perte et le risque
     emetteur.
  5. Objections clients : le client pose une objection classique, la
     reponse du Sales (phrase, chiffre a citer, erreur a eviter) est
     chiffree avec le produit a l'ecran et reste cachee jusqu'a ce qu'on
     la demande.

Seule la section choisie est calculee, et chaque appel a sales.py est en
cache (l'objet Marche est hachable) : les curseurs restent rapides.
"""

import json
import math
import random
from dataclasses import dataclass, replace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import graphiques as G
import moteur as M
import sales as S
from communs import MAX_CACHE

BRC, AC, CP = "Barrier Reverse Convertible", "Autocall Phoenix", "Capital Protection"
PRODUITS = [BRC, AC, CP]
DEMANDE, ARGENT, REVENTE, PITCH, OBJECTIONS = ("Le client demande", "Où va l'argent",
                                               "Revendre en cours de vie", "Pitch client",
                                               "Objections clients")
SECTIONS = [DEMANDE, ARGENT, REVENTE, PITCH, OBJECTIONS]
PARTICULIER, CGP, INSTIT = "Particulier", "Conseiller en gestion de patrimoine", "Institutionnel"
PROFILS = [PARTICULIER, CGP, INSTIT]
VUES = {"Forte hausse": "le marché va fortement monter",
        "Hausse modérée": "le marché va monter un peu",
        "Stabilité": "le marché va peu bouger",
        "Baisse modérée": "le marché peut baisser un peu",
        "Forte baisse": "le marché va fortement baisser"}
FREQUENCES = {"Trimestrielle": 4, "Semestrielle": 2, "Annuelle": 1}
PERIODE = {4: "trimestre", 2: "semestre", 1: "an"}

# Ce que le client fixe, et le parametre que le desk cherche (cle de sales.py)
CHERCHE = {
    BRC: {"Un coupon, avec la barrière la plus basse possible": "barriere",
          "Un coupon, avec le strike le plus bas possible": "strike",
          "Un coupon, sur la durée la plus courte possible": "maturite"},
    AC: {"Un coupon, avec la barrière de capital la plus basse possible": "barriere_capital",
         "Un coupon, avec le trigger de rappel le plus bas possible": "trigger"},
    CP: {"Une participation, avec le cap le plus haut possible": "cap",
         "Une participation, avec la protection la plus haute possible": "protection"},
}
# Curseur de la structure que « Retenir » met a jour, et sens d'arrondi qui
# garde la demande du client satisfaite (moins favorable au client = plus de budget)
CURSEUR = {"barriere": ("as_brc_barriere", math.ceil), "strike": ("as_brc_strike", math.ceil),
           "maturite": ("as_brc_maturite", None),
           "barriere_capital": ("as_ac_bar_capital", math.ceil),
           "trigger": ("as_ac_trigger", math.ceil),
           "protection": ("as_cp_protection", math.floor), "cap": ("as_cp_cap", math.floor)}
EXTREME = {"barriere": "la plus basse", "strike": "le plus bas", "maturite": "la plus courte",
           "barriere_capital": "la plus basse", "trigger": "le plus bas",
           "protection": "la plus haute", "cap": "le plus haut"}
NOM_PARAMETRE = {"barriere": "Barrière", "strike": "Strike", "maturite": "Maturité",
                 "barriere_capital": "Barrière de capital", "trigger": "Trigger de rappel",
                 "protection": "Protection", "cap": "Cap"}

# Libelles courts des lignes de sales.py pour l'axe des graphes en cascade
COURT = {"zc": "Taux", "coupons": "Coupons", "rappel": "Rappel anticipé", "put": "Put vendu",
         "credit": "Risque émetteur", "call": "Call acheté", "cap": "Call au cap vendu",
         "marge": "Marge banque", "retro": "Rétro"}


# =====================================================================
# STRUCTURE
# =====================================================================

@dataclass(frozen=True)
class Structure:
    """Le produit tel que le Sales l'a regle. Fige : sert de cle de cache."""
    produit: str
    T: float
    barriere: float = 0.70
    strike: float = 1.0
    continu: bool = True
    par_an: int = 4
    trigger: float = 1.0
    bar_coupon: float = 0.70
    bar_capital: float = 0.60
    memoire: bool = True
    n_actifs: int = 1
    correlation: float = 0.6
    protection: float = 1.0
    cap: float = 3.0

    @property
    def n_obs(self):
        return max(int(self.T * self.par_an), 1)


def _v(cle, defaut):
    """Defaut d'un curseur que « Retenir » peut aussi regler : ne pas le donner deux fois."""
    return None if cle in st.session_state else defaut


def _structure(produit, cherche=None):
    """Les curseurs du produit. Le parametre cherche par le desk est grise."""
    def gris(p):
        return cherche == p

    def aide(p, texte):
        return ("Trouvé par le desk dans la section « Le client demande » : ce curseur est "
                "ignoré ici. « Retenir cette structure » y inscrit la réponse.\n\n" + texte
                if gris(p) else texte)

    retro_aide = ("Ce que la banque reverse au réseau qui place le produit (banque privée, "
                  "réseau d'agences, conseiller). Payée d'avance sur toute la durée, comme la "
                  "marge, et prise sur le budget option : chaque point de rétro se paie en "
                  "coupon. Toutes les sections en tiennent compte.")
    with st.container(border=True):
        if produit == BRC:
            c = st.columns(4)
            T = c[0].slider("Maturité (années)", 0.25, 3.0, _v("as_brc_maturite", 1.0), 0.25,
                            disabled=gris("maturite"), key="as_brc_maturite",
                            bind="query-params", help=aide(
                                "maturite", "Plus c'est long, plus la barrière a le temps "
                                "d'être touchée : le put vendu vaut plus, le coupon monte."))
            bar = c[1].slider("Barrière (%)", 40, 100, _v("as_brc_barriere", 70), 1,
                              disabled=gris("barriere"), key="as_brc_barriere",
                              bind="query-params", help=aide(
                                  "barriere", "Le niveau qui arme le risque. Plus elle est "
                                  "basse, plus le client a de marge, moins il touche."))
            strike = c[2].slider("Strike (%)", 70, 120, _v("as_brc_strike", 100), 1,
                                 disabled=gris("strike"), key="as_brc_strike",
                                 bind="query-params", help=aide(
                                     "strike", "Sous ce niveau à l'échéance, et si la barrière "
                                     "a été touchée, le client perd comme le sous-jacent "
                                     "rapporte au strike. Un strike sous 100 % (BRC décote) "
                                     "fait perdre plus tard et moins."))
            retro = c[3].slider("Rétrocession (% par an)", 0.0, 2.5, 1.0, 0.1, key="as_retro",
                                bind="query-params", help=retro_aide)
            continu = st.radio(
                "Observation de la barrière", ["Continue (américaine)", "À l'échéance (européenne)"],
                horizontal=True, key="as_brc_observation", bind="query-params",
                help="Continue : la barrière est surveillée chaque jour, un seul passage suffit "
                     "à l'armer. Européenne : seul le niveau final compte, le client est mieux "
                     "protégé et touche moins.").startswith("Continue")
            s = Structure(BRC, T, barriere=bar / 100, strike=strike / 100, continu=continu)
        elif produit == AC:
            c = st.columns(4)
            T = c[0].slider("Maturité maximale (années)", 1.0, 6.0, 3.0, 0.5,
                            key="as_ac_maturite", bind="query-params",
                            help="Durée maximale : le produit est presque toujours rappelé avant.")
            freq = c[1].selectbox("Fréquence d'observation", list(FREQUENCES),
                                  key="as_ac_frequence", bind="query-params",
                                  help="Une date d'observation par période : coupon, rappel "
                                       "et mémoire se jouent à ces dates seulement.")
            trig = c[2].slider("Trigger de rappel (%)", 80, 120, _v("as_ac_trigger", 100), 1,
                               disabled=gris("trigger"), key="as_ac_trigger",
                               bind="query-params", help=aide(
                                   "trigger", "Au-dessus à une date d'observation, le produit "
                                   "s'arrête et rembourse le nominal plus le coupon. Plus il "
                                   "est bas, plus le rappel est fréquent, plus le coupon baisse."))
            retro = c[3].slider("Rétrocession (% par an)", 0.0, 2.5, 1.0, 0.1, key="as_retro",
                                bind="query-params", help=retro_aide)
            c = st.columns(4)
            bcp = c[0].slider("Barrière de coupon (%)", 50, 100, 70, 1, key="as_ac_bar_coupon",
                              bind="query-params",
                              help="Le coupon n'est versé que si le sous-jacent est au-dessus "
                                   "à la date d'observation.")
            bcap = c[1].slider("Barrière de capital (%)", 30, 100, _v("as_ac_bar_capital", 60), 1,
                               disabled=gris("barriere_capital"), key="as_ac_bar_capital",
                               bind="query-params", help=aide(
                                   "barriere_capital", "Observée à l'échéance seulement. "
                                   "Dessous, le client encaisse toute la baisse : c'est le vrai "
                                   "risque du produit."))
            n_actifs = c[2].select_slider(
                "Sous-jacents (worst-of)", [1, 2, 3, 4], 1, key="as_ac_sous_jacents",
                bind="query-params",
                help="Le plus mauvais des sous-jacents décide de tout. Plus il y en a, plus le "
                     "coupon monte, et la probabilité de perte aussi.")
            correl = c[3].slider("Corrélation", 0.0, 0.95, 0.60, 0.05, disabled=n_actifs == 1,
                                 key="as_ac_correlation", bind="query-params",
                                 help="Corrélation basse : il y a presque toujours un traînard, "
                                      "coupon élevé et risque élevé.")
            memoire = st.checkbox("Effet mémoire", value=True, key="as_ac_memoire",
                                  bind="query-params",
                                  help="Un coupon manque est rattrapé à la première date où la "
                                       "barrière de coupon est de nouveau respectée.")
            s = Structure(AC, T, par_an=FREQUENCES[freq], trigger=trig / 100,
                          bar_coupon=bcp / 100, bar_capital=bcap / 100, memoire=memoire,
                          n_actifs=n_actifs, correlation=correl)
        else:
            c = st.columns(4)
            T = c[0].slider("Maturité (années)", 1.0, 10.0, 5.0, 0.5, key="as_cp_maturite",
                            bind="query-params",
                            help="Plus c'est long, moins la banque doit mettre de côté pour "
                                 "garantir : le budget option grossit.")
            prot = c[1].slider("Protection (%)", 70, 100, _v("as_cp_protection", 100), 1,
                               disabled=gris("protection"), key="as_cp_protection",
                               bind="query-params", help=aide(
                                   "protection", "Le minimum rendu à l'échéance. Chaque point "
                                   "abandonné libère du budget pour la participation."))
            cap = c[2].slider("Cap (%)", 105, 300, _v("as_cp_cap", 300), 1,
                              disabled=gris("cap"), key="as_cp_cap", bind="query-params",
                              help=aide("cap", "Au-delà, le client ne profite plus de la "
                                               "hausse. 300 % = pas de cap."))
            retro = c[3].slider("Rétrocession (% par an)", 0.0, 2.5, 1.0, 0.1, key="as_retro",
                                bind="query-params", help=retro_aide)
            s = Structure(CP, T, protection=prot / 100, cap=cap / 100)
    return s, retro / 100


# =====================================================================
# CALCULS EN CACHE (sales.py, l'objet Marche est hachable)
# =====================================================================

@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _reponse(mk, s, cherche, cible):
    if cherche == "barriere":
        return S.brc_barriere(mk, cible, s.T, s.strike, s.continu)
    if cherche == "strike":
        return S.brc_strike(mk, cible, s.T, s.barriere, s.continu)
    if cherche == "maturite":
        return S.brc_maturite(mk, cible, s.barriere, s.strike, s.continu)
    if cherche == "barriere_capital":
        return S.autocall_barriere_capital(mk, cible, s.T, s.par_an, s.trigger, s.bar_coupon,
                                           s.memoire, s.n_actifs, s.correlation)
    if cherche == "trigger":
        return S.autocall_trigger(mk, cible, s.T, s.par_an, s.bar_coupon, s.bar_capital,
                                  s.memoire, s.n_actifs, s.correlation)
    if cherche == "protection":
        return S.cp_protection(mk, cible, s.T, s.cap)
    return S.cp_cap(mk, cible, s.T, s.protection)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _prix(mk, s):
    """Coupon annuel (BRC, autocall) ou participation : les memes calculs que les pages."""
    if s.produit == BRC:
        return S.coupon_brc(mk, s.T, s.barriere, s.strike, s.continu)
    if s.produit == AC:
        return s.par_an * S.coupon_autocall(mk, s.T, s.par_an, s.trigger, s.bar_coupon,
                                            s.bar_capital, s.memoire, s.n_actifs, s.correlation)
    return S.participation_cp(mk, s.T, s.protection, s.cap)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _cascade(mk, s):
    if s.produit == BRC:
        return S.cascade_brc(mk, s.T, s.barriere, s.strike, s.continu)
    if s.produit == AC:
        return S.cascade_autocall(mk, s.T, s.par_an, s.trigger, s.bar_coupon, s.bar_capital,
                                  s.memoire, s.n_actifs, s.correlation)
    return S.cascade_cp(mk, s.T, s.protection, s.cap)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _vie_autocall(mk, s):
    """Duree de vie moyenne et probabilite de rappel a la premiere date (memes tirages)."""
    ch, _ = mk.panier(s.n_actifs, s.correlation, s.T, s.bar_capital)
    duree, _, _ = M.stats_autocall(ch, s.T, s.n_obs, s.trigger, s.bar_capital)
    i = M.idx_obs(ch.shape[1] - 1, s.n_obs)[0]
    rappel_1 = float((ch[:, i] >= s.trigger * M.S0).mean()) if s.n_obs > 1 else 0.0
    return float(duree), rappel_1


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _secondaire(mk_e, mk_j, s, t, niveau, touchee, observes, fourchette):
    if s.produit == BRC:
        return S.valeur_secondaire_brc(mk_e, mk_j, s.T, s.barriere, s.strike, s.continu, t,
                                       niveau, touchee, fourchette)
    if s.produit == AC:
        return S.valeur_secondaire_autocall(mk_e, mk_j, s.T, s.par_an, s.trigger, s.bar_coupon,
                                            s.bar_capital, s.memoire, t, niveau, list(observes),
                                            n_actifs=s.n_actifs, correlation=s.correlation,
                                            fourchette=fourchette)
    return S.valeur_secondaire_cp(mk_e, mk_j, s.T, s.protection, s.cap, t, niveau, fourchette)


# =====================================================================
# FORMATS
# =====================================================================

def _p(x, d=1):
    """Fraction -> '8.8 %'."""
    return f"{x * 100:,.{d}f} %"


def _duree(T):
    if T < 1:
        return f"{round(T * 12)} mois"
    if abs(T - round(T)) < 1e-9:
        return f"{int(round(T))} an" + ("s" if T >= 2 else "")
    return f"{T:g} ans"


def _sous_jacent(s):
    """Le sous-jacent tel qu'on le nomme a un particulier."""
    if s.produit == AC and s.n_actifs > 1:
        return f"le plus faible des {s.n_actifs} sous-jacents"
    return "l'indice"


# =====================================================================
# GRAPHE EN CASCADE (absent de graphiques.py : construit ici, en Altair)
# =====================================================================

def graphe_cascade(etapes, titre, sous_titre=None, nominal=None, zero=False, legende=None,
                   hauteur=380):
    """
    Cascade : chaque barre part de la fin de la precedente.

    etapes : liste de dict {court, libelle, valeur, couleur, pourquoi, total}.
    total=True : barre pleine (de l'axe jusqu'a la valeur) ; sinon variation.
    zero : l'axe contient 0 (cascade d'un budget) ; sinon il est coupe autour
    des valeurs, et les barres pleines partent du bas de l'axe.
    Au survol, la barre ressort et l'infobulle donne le montant, le cumul et
    la raison economique de la ligne.
    """
    lignes, cumul = [], 0.0
    for i, e in enumerate(etapes):
        if e.get("total"):
            debut, fin = None, e["valeur"]
        else:
            debut, fin = cumul, cumul + e["valeur"]
        cumul = fin
        lignes.append({"i": i, "debut": debut, "fin": fin, "couleur": e["couleur"],
                       "libelle": e["libelle"], "pourquoi": e.get("pourquoi", ""),
                       "montant": G.eur(e["valeur"]) if e.get("total")
                       else ("0 EUR" if abs(e["valeur"]) < 0.5 else G.eur(e["valeur"], signe=True)),
                       "cumul": G.eur(fin),
                       "part": "-" if not nominal else G.pct(e["valeur"] / nominal * 100, 2)})
    hauts = [l["fin"] for l in lignes] + [l["debut"] for l in lignes if l["debut"] is not None]
    if nominal and not zero:
        hauts.append(nominal)
    lo, hi = min(hauts), max(hauts)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    marge = (hi - lo) * 0.12 or abs(hi) * 0.05 or 1.0
    lo, hi = (lo - marge * (0.4 if zero else 1.0)), hi + marge
    for l in lignes:
        if l["debut"] is None:
            l["debut"] = lo
        l["haut"] = max(l["debut"], l["fin"])
        l["texte"] = l["montant"].replace(" EUR", "")
    df = pd.DataFrame(lignes)
    df["a"], df["b"] = df["i"] - 0.36, df["i"] + 0.36

    n = len(lignes)
    ech_x = alt.Scale(domain=[-0.6, n - 0.4], nice=False, zero=False)
    ech_y = alt.Scale(domain=[lo, hi], nice=False, zero=False)
    noms = json.dumps([e["court"] for e in etapes])
    axe_x = alt.Axis(title=None, values=list(range(n)), grid=False, ticks=False,
                     labelExpr=f"{noms}[datum.value]", labelAngle=0,
                     labelFontSize=11, labelLimit=140, labelPadding=6)
    survol = alt.selection_point(name="survol", fields=["i"], on="pointerover",
                                 clear="pointerout", empty=False)
    base = alt.Chart(df)
    couches = []
    if zero:
        couches.append(alt.Chart(pd.DataFrame([{"y": 0.0}])).mark_rule(
            color=G.NIVEAU, opacity=0.6).encode(y=alt.Y("y:Q", scale=ech_y)))
    if nominal and not zero:
        couches += G._niveaux_horizontaux([(nominal, f"nominal  {G.eur(nominal)}", G.NIVEAU)],
                                          n - 0.45)
    liens = df.iloc[:-1].assign(c=df["i"].iloc[:-1] + 0.36, d=df["i"].iloc[:-1] + 0.64)
    couches.append(alt.Chart(liens).mark_rule(color=G.NIVEAU, strokeDash=[3, 3],
                                              opacity=0.7).encode(
        x=alt.X("c:Q", scale=ech_x), x2="d:Q", y=alt.Y("fin:Q", scale=ech_y)))
    infobulle = [alt.Tooltip("libelle:N", title="Ligne"), alt.Tooltip("montant:N", title="Montant"),
                 alt.Tooltip("cumul:N", title="Cumul après cette ligne")]
    if nominal:
        infobulle.append(alt.Tooltip("part:N", title="En % du nominal"))
    infobulle.append(alt.Tooltip("pourquoi:N", title="Pourquoi"))
    couches.append(base.mark_rect(cornerRadius=2, clip=True).encode(
        x=alt.X("a:Q", scale=ech_x, axis=axe_x), x2="b:Q",
        y=alt.Y("debut:Q", scale=ech_y, axis=alt.Axis(title="EUR", labelExpr=G.AXE_EUR)),
        y2="fin:Q", color=alt.Color("couleur:N", scale=None),
        opacity=alt.condition(survol, alt.value(1.0), alt.value(0.82)),
        stroke=alt.condition(survol, alt.value(G.NIVEAU), alt.value(None)),
        tooltip=infobulle).add_params(survol))
    # etiquettes de la couleur de leur barre : lisibles en theme clair comme sombre
    couches.append(base.mark_text(baseline="bottom", dy=-4, fontSize=11, fontWeight="bold").encode(
        x=alt.X("i:Q", scale=ech_x), y=alt.Y("haut:Q", scale=ech_y), text="texte:N",
        color=alt.Color("couleur:N", scale=None)))
    if legende:
        dl = pd.DataFrame([{"nom": a, "x": 0.0, "y": lo} for a, _ in legende])
        couches.append(alt.Chart(dl).mark_point(filled=True, shape="square", opacity=0).encode(
            x=alt.X("x:Q", scale=ech_x), y=alt.Y("y:Q", scale=ech_y),
            color=alt.Color("nom:N", scale=alt.Scale(domain=[a for a, _ in legende],
                                                      range=[b for _, b in legende]),
                            legend=alt.Legend(orient="top", title=None, symbolType="square",
                                              symbolSize=160, labelFontSize=12,
                                              labelLimit=420))))
    return alt.layer(*couches).properties(title=G._titre(titre, sous_titre),
                                          height=hauteur).resolve_scale(color="independent")


# =====================================================================
# 1. LE CLIENT DEMANDE
# =====================================================================

def _saisie_demande(produit):
    """Ce que le client fixe. Affiche avant la structure, pour griser le parametre cherche."""
    choix = CHERCHE[produit]
    c1, c2 = st.columns([2, 1])
    demande = c1.selectbox(
        "Le client veut", list(choix), key=f"as_dem_{'brc' if produit == BRC else 'ac' if produit == AC else 'cp'}",
        bind="query-params",
        help="La contrainte que le client pose au téléphone. Le desk cherche le paramètre le "
             "plus favorable au client qui la respecte ; tout le reste de la structure vient "
             "du cadre en dessous.")
    if produit == CP:
        cible = c2.number_input(
            "Participation demandée (%)", 5.0, 300.0, 60.0, 5.0, key="as_dem_part",
            bind="query-params",
            help="La part de la hausse que le client veut recevoir. 100 % = la hausse entière.")
    else:
        cible = c2.number_input(
            "Coupon demandé (% par an)", 0.5, 40.0, 8.0 if produit == BRC else 5.0, 0.25,
            key="as_dem_cpn_brc" if produit == BRC else "as_dem_cpn_ac", bind="query-params",
            help="Le coupon annuel que le client veut toucher, net de la rétrocession. Pour "
                 "l'autocall : l'équivalent annuel des coupons par observation.")
    return choix[demande], cible / 100


def _phrase_client(s, cherche, rep, depasse):
    """La reponse traduite pour le client, sans jargon."""
    texte = _phrase_brute(s, cherche, rep, depasse)
    return texte[0].upper() + texte[1:]


def _phrase_brute(s, cherche, rep, depasse):
    cible, x, obtenu = rep.cible, rep.valeur, rep.obtenu
    T = _duree(x if cherche == "maturite" else s.T)
    if s.produit == CP:
        if not rep.possible:
            if cherche == "cap":
                return (f"{_p(cible, 0)} de participation n'est pas faisable aujourd'hui avec "
                        f"{_p(s.protection, 0)} du capital protégé : au mieux {_p(obtenu, 0)}. "
                        "Pour aller plus haut, il faut accepter une protection plus basse ou "
                        "une durée plus longue.")
            return (f"{_p(cible, 0)} de participation demanderait de protéger moins de "
                    f"{_p(x, 0)} du capital : au mieux {_p(obtenu, 0)} avec {_p(x, 0)} protégé. "
                    "Un cap sur la hausse ou une durée plus longue permettrait d'aller plus loin.")
        if depasse:
            return (f"Bonne nouvelle : même sans rien céder, vous recevez {_p(obtenu, 0)} de la "
                    f"hausse sur {T}, avec {_p(s.protection if cherche == 'cap' else x, 0)} du "
                    "capital protégé à l'échéance.")
        if cherche == "protection":
            return (f"Pour {_p(cible, 0)} de la hausse sur {T}, il faut accepter une protection "
                    f"à {_p(x, 0)} : au pire, vous perdez {_p(1 - x, 0)} de votre capital à "
                    "l'échéance.")
        return (f"Avec {_p(s.protection, 0)} du capital protégé sur {T}, {_p(cible, 0)} de la "
                f"hausse est possible si le gain est plafonné : au-delà d'une hausse de "
                f"{_p(x - 1, 0)}, vous ne profitez plus du marché, soit un gain maximum de "
                f"{_p(obtenu * (x - 1), 0)}.")
    if not rep.possible:
        quoi = {"barriere": f"avec une barrière à {_p(x, 0)}", "strike": f"avec un strike à {_p(x, 0)}",
                "maturite": f"sur {T}", "barriere_capital": f"avec une barrière de capital à {_p(x, 0)}",
                "trigger": f"avec un rappel à {_p(x, 0)}"}[cherche]
        piste = {"barriere": "une durée plus longue ou un strike plus haut",
                 "strike": "une barrière plus haute",
                 "maturite": "une barrière ou un strike plus haut",
                 "barriere_capital": "une barrière de coupon plus haute, un panier de plusieurs "
                                     "sous-jacents ou une durée plus longue",
                 "trigger": "une barrière de capital plus haute"}[cherche]
        return (f"{_p(cible, 2)} par an n'est pas faisable aujourd'hui : le maximum est "
                f"{_p(obtenu, 2)} par an, {quoi}. Pour aller plus haut, il faudrait accepter "
                f"{piste}.")
    debut = "Bonne nouvelle : " if depasse else ""
    if cherche == "barriere":
        quand = ("à un moment quelconque de la vie du produit" if s.continu
                 else "à l'échéance")
        return (f"{debut}pour {_p(obtenu, 2)} par an sur {T}, votre capital n'est en risque que "
                f"si l'indice perd plus de {_p(1 - x, 0)} {quand} et finit sous {_p(s.strike, 0)} "
                "de son niveau de départ.")
    if cherche == "strike":
        return (f"{debut}pour {_p(obtenu, 2)} par an, la perte ne commence que sous "
                f"{_p(x, 0)} du niveau de départ, si la barrière à {_p(s.barriere, 0)} a été "
                "touchée, et elle est plus faible que celle de l'indice.")
    if cherche == "maturite":
        return (f"""{debut}{_p(obtenu, 2)} par an est faisable des {T}, avec une barrière à {_p(s.barriere, 0)}.""")
    if cherche == "barriere_capital":
        return (f"{debut}pour {_p(obtenu, 2)} par an, votre capital est rendu en entier à "
                f"l'échéance tant que {_sous_jacent(s)} n'a pas perdu plus de {_p(1 - x, 0)}.")
    return (f"{debut}pour {_p(obtenu, 2)} par an, le produit est remboursé par anticipation "
            f"dès que {_sous_jacent(s)} est au-dessus de {_p(x, 0)} de son niveau de départ à une "
            "date d'observation.")


def _retenir(cle, valeur):
    st.session_state[cle] = valeur


def arrondi_client(s, cherche, valeur):
    """
    La valeur ronde que le Sales propose (au point près), dans le sens qui
    garde la demande satisfaite : barriere, strike et trigger vers le haut,
    protection et cap vers le bas. Renvoie (valeur du curseur, structure).
    """
    cle, sens = CURSEUR[cherche]
    if cherche == "maturite":
        return float(valeur), replace(s, T=float(valeur))
    pts = int(sens(round(valeur * 100, 6)))
    champ = {"barriere_capital": "bar_capital"}.get(cherche, cherche)
    return pts, replace(s, **{champ: pts / 100})


def _section_demande(mk, s, cherche, cible):
    rep = _reponse(mk, s, cherche, cible)
    depasse = rep.possible and abs(rep.valeur - rep.courbe[0][0]) < 1e-9
    est_part = s.produit == CP
    unite = "%" if est_part else "% p.a."
    d = 0 if est_part else 2
    curseur, s_client = arrondi_client(s, cherche, rep.valeur)
    obtenu_client = _prix(mk, s_client)

    if cherche == "maturite":
        exact, rond = _duree(rep.valeur), _duree(rep.valeur)
    else:
        exact, rond = f"{rep.valeur * 100:,.1f}%", f"{curseur}%"
    k = st.columns(4)
    k[0].metric(NOM_PARAMETRE[cherche] + (" trouvée" if cherche in ("barriere", "maturite",
                "barriere_capital", "protection") else " trouve"), exact,
                help="Le paramètre le plus favorable au client qui respecte sa demande, au "
                     "dixième près. Hors d'atteinte : la meilleure valeur possible.")
    k[1].metric("Obtenu", f"{rep.obtenu * 100:,.{d}f}{unite}",
                help="Ce que paie la structure trouvée, net de la rétrocession.")
    k[2].metric("Arrondi pour le client", rond,
                help="La valeur ronde que le Sales propose, arrondie dans le sens qui garde la "
                     "demande satisfaite. « Retenir » l'inscrit dans la structure.")
    k[3].metric("Obtenu à l'arrondi", f"{obtenu_client * 100:,.{d}f}{unite}",
                help="Recalculé sur la structure arrondie, avec les mêmes tirages.")

    if rep.possible:
        st.success("**Réponse du desk.** " + rep.message)
    else:
        st.error("**Hors d'atteinte.** " + rep.message)
    rep_client = replace(rep, valeur=s_client.T if cherche == "maturite" else curseur / 100,
                         obtenu=obtenu_client)
    with st.container(border=True):
        st.markdown("**À dire au client.** " + _phrase_client(s_client, cherche, rep_client,
                                                               depasse))

    # --- graphe : l'arbitrage, avec la demande et la reponse
    pts = sorted(rep.courbe)
    x = np.array([a for a, _ in pts])
    y = np.array([b for _, b in pts]) * 100
    if cherche == "maturite":
        fmt_x, xlabel, x_aff = (lambda v: f"{v:,.2f} an"), "Maturité (années)", x
    else:
        fmt_x, xlabel, x_aff = (lambda v: f"{v:,.0f} %"), NOM_PARAMETRE[cherche] + " (%)", x * 100
    sol_x = rep.valeur if cherche == "maturite" else rep.valeur * 100
    nom_y = "Participation offerte" if est_part else "Coupon offert (p.a.)"
    ylabel = "Participation offerte (%)" if est_part else "Coupon offert (% p.a.)"
    lo, hi = min(y.min(), rep.cible * 100), max(y.max(), rep.cible * 100)
    y_domaine = (lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.10)
    fmt_y = (lambda v: f"{v:,.0f} %") if est_part else (lambda v: f"{v:,.2f} %")
    atteint = ["oui" if v >= rep.cible * 100 - 1e-9 else "non" for v in y]
    couleur_v = G.PERTE if cherche in ("barriere", "barriere_capital") else G.NIVEAU
    etiquette = f"{NOM_PARAMETRE[cherche].lower()}  {fmt_x(sol_x)}"
    if rep.possible:
        titre = (f"{NOM_PARAMETRE[cherche]} à {fmt_x(sol_x)} : {EXTREME[cherche]} qui paie "
                 f"{fmt_y(rep.cible * 100)}")
    else:
        titre = (f"{fmt_y(rep.cible * 100)} hors d'atteinte : au mieux {fmt_y(rep.obtenu * 100)}"
                 f" ({NOM_PARAMETRE[cherche].lower()} {fmt_x(sol_x)})")
    sous = {"barriere": "Plus la barrière est haute, plus le put vendu vaut cher, plus le coupon "
                        "monte. Au-dessus du strike, elle ne sert plus à rien.",
            "strike": "Plus le strike est haut, plus le client perd tôt et fort : le put vaut "
                      "plus, le coupon monte.",
            "maturite": "Le coupon est annuel : allonger donne plus de temps à la barrière, "
                        "mais la courbe finit par plafonner.",
            "barriere_capital": "La barrière de capital est le vrai risque : la remonter paie "
                                "le coupon.",
            "trigger": "Un trigger bas rappelle plus tôt et plus souvent : chaque rappel coupe "
                       "les coupons suivants.",
            "protection": "Chaque point de protection abandonné libère du budget pour la "
                          "participation.",
            "cap": "Capper vend un call de strike élevé : sa prime paie de la participation. "
                   "300 % = pas de cap."}[cherche]
    graphe = G.courbes(
        x_aff, [{"nom": nom_y, "y": y, "couleur": G.PRODUIT, "epaisseur": 2.8}],
        titre, sous, xlabel, ylabel, fmt_x=fmt_x, fmt_y=fmt_y,
        nom_x=NOM_PARAMETRE[cherche], y_domaine=y_domaine,
        niveaux_h=[(rep.cible * 100, f"demande du client  {fmt_y(rep.cible * 100)}", G.NIVEAU)],
        niveaux_v=[(sol_x, etiquette, couleur_v)],
        marqueurs=[(sol_x, rep.obtenu * 100, G.PRODUIT)],
        lignes_infobulle=[("Respecte la demande", atteint)], legende=False, hauteur=360)
    G.affiche(graphe)
    if not mk.bates and cherche in ("barriere", "barriere_capital"):
        st.caption("Sous Black-Scholes + skew, la vol du niveau de barrière s'applique à toute "
                   "la trajectoire : la courbe peut redescendre quand la barrière monte. Le "
                   "desk balaie toute la grille au lieu de la supposer croissante. Sous Bates, "
                   "elle est monotone.")

    st.button(f"Retenir : {NOM_PARAMETRE[cherche].lower()} {rond.replace('%', ' %')}",
              key="as_retenir", type="primary", on_click=_retenir,
              args=(CURSEUR[cherche][0], curseur),
              help="Inscrit la réponse dans le cadre de la structure, arrondie au point dans le "
                   "sens qui garde la demande du client satisfaite. Les autres sections "
                   "(où va l'argent, revente, pitch) partent alors de cette structure.")


# =====================================================================
# 2. OU VA L'ARGENT
# =====================================================================

def _budget(cascade):
    """
    Sources et emplois du budget, a partir des lignes de sales.py.
    Ce que l'obligation libere (nominal - zero-coupon) et ce que le client
    cede (option vendue, risque porte) financent ; coupons, options achetees
    et frais consomment. La somme retombe a zero.
    """
    N = cascade.nominal
    sources, emplois = [], []
    for l in cascade.lignes:
        if l.cle == "zc":
            sources.append({"cle": "zc", "court": "Taux",
                            "libelle": "Ce que l'obligation libère (taux)",
                            "valeur": N - l.montant, "couleur": G.ACCENT,
                            "pourquoi": f"{G.eur(l.montant)} placés en obligation zéro-coupon "
                                        f"rendent le nominal à l'échéance ; le reste est le "
                                        "budget que donnent les taux."})
        elif l.cle in ("marge", "retro"):
            emplois.append({"cle": l.cle, "court": COURT[l.cle], "libelle": l.libelle,
                            "valeur": -l.montant,
                            "couleur": G.NIVEAU, "pourquoi": l.explication})
        elif l.montant < 0 or (l.cle == "credit"):
            sources.append({"cle": l.cle, "court": COURT.get(l.cle, l.libelle),
                            "libelle": l.libelle,
                            "valeur": -l.montant, "couleur": G.ACCENT, "pourquoi": l.explication})
        else:
            emplois.append({"cle": l.cle, "court": COURT.get(l.cle, l.libelle),
                            "libelle": l.libelle,
                            "valeur": -l.montant, "couleur": G.PRODUIT, "pourquoi": l.explication})
    return [e for e in sources + emplois if abs(e["valeur"]) >= 0.5]   # sans les lignes nulles


PROVENANCE = {"zc": "des taux", "put": "du put vendu par le client",
              "credit": "du risque émetteur porté par le client", "cap": "du call au cap vendu"}
LEGENDE_BUDGET = [("Finance la structure", G.ACCENT), ("Revient au client", G.PRODUIT),
                  ("Frais : banque et distributeur", G.NIVEAU)]


def _metriques_resultat(s, cascade, col, libelle):
    if s.produit == AC:
        col.metric(libelle, f"{cascade.infos['coupon_pa'] * 100:,.2f}% p.a.",
                   help=f"Soit {cascade.resultat * 100:,.2f}% par observation.")
    elif s.produit == BRC:
        col.metric(libelle, f"{cascade.resultat * 100:,.2f}% p.a.")
    else:
        col.metric(libelle, f"{cascade.resultat * 100:,.0f}%")


def _resultat_pa(s, cascade):
    """Coupon annuel (BRC, autocall) ou participation, en fraction."""
    return cascade.infos["coupon_pa"] if s.produit == AC else cascade.resultat


def _section_argent(mk, s):
    cascade = _cascade(mk, s)
    N, T = cascade.nominal, s.T
    quoi = "participation" if s.produit == CP else "coupon"
    cout = cascade.infos["cout_retro_par_pt"]

    frais = {l.cle: l.montant for l in cascade.lignes}
    # autocall : frais pris sur la duree de vie attendue, pas sur la maturite maximale
    vie = (f"la durée de vie attendue ({cascade.infos['duree_vie']:,.2f} an)" if s.produit == AC
           else _duree(T))
    Tf = cascade.infos.get("duree_vie", T)
    k = st.columns(4)
    _metriques_resultat(s, cascade, k[0], "Participation offerte" if s.produit == CP
                        else "Coupon offert")
    k[1].metric("Marge banque", G.eur(frais["marge"]),
                help=f"{mk.marge * 100:,.1f}% par an sur {vie}, prélevée d'avance.")
    k[2].metric("Rétrocession", G.eur(frais["retro"]),
                help=f"{mk.retro * 100:,.1f}% par an sur {vie}, payée d'avance au "
                     "distributeur.")
    k[3].metric("Coût de 1 % de rétro", f"{cout * 100:,.{0 if s.produit == CP else 2}f} pt"
                + ("" if s.produit == CP else " p.a."),
                help=f"Ce que le client perd en {quoi} pour chaque point de rétrocession par an.")

    etapes = _budget(cascade)
    sources = sum(e["valeur"] for e in etapes if e["valeur"] > 0)
    principale = max((e for e in etapes if e["valeur"] > 0), key=lambda e: e["valeur"])
    zc = next(l.montant for l in cascade.lignes if l.cle == "zc")
    provenance = PROVENANCE.get(principale["cle"], "de " + principale["court"].lower())
    if abs(principale["valeur"] - sources) < 0.5:
        titre = f"{G.eur(sources)} financent la structure, tous viennent {provenance}"
    else:
        titre = (f"{G.eur(sources)} financent la structure, dont {G.eur(principale['valeur'])} "
                 f"{provenance}")
    graphe = graphe_cascade(
        etapes, titre,
        f"Sur {G.eur(N)}, {G.eur(zc)} sont placés en obligation pour rendre le nominal. Les "
        "barres montantes financent la structure, les descendantes la consomment : on retombe "
        "à zéro.", nominal=N, zero=True, legende=LEGENDE_BUDGET)
    G.affiche(graphe)

    lignes = [{"Composante": l.libelle, "EUR": G.eur(l.montant),
               "% du nominal": f"{l.montant / N * 100:,.2f}%", "Pourquoi": l.explication}
              for l in cascade.lignes]
    lignes.append({"Composante": "Total = nominal payé par le client", "EUR": G.eur(cascade.somme),
                   "% du nominal": f"{cascade.somme / N * 100:,.2f}%", "Pourquoi": ""})
    st.table(lignes, hide_index=True)

    # --- deux reseaux
    st.subheader("Deux réseaux, deux coupons",
                 help="Le même produit, chez le même émetteur, le même jour, placé par deux "
                      "distributeurs qui ne prennent pas la même rétrocession. Le client ne paie "
                      "pas de commission visible : il la paie en coupon.")
    c = st.columns(2)
    ra = c[0].slider("Réseau A : rétro (% par an)", 0.0, 2.5, 0.5, 0.1, key="as_retro_a",
                     bind="query-params",
                     help="Par exemple une banque privée ou un mandat de gestion, qui facture "
                          "ses honoraires à part.")
    rb = c[1].slider("Réseau B : rétro (% par an)", 0.0, 2.5, 1.5, 0.1, key="as_retro_b",
                     bind="query-params",
                     help="Par exemple un réseau d'agences grand public, rémunéré par la "
                          "rétrocession.")
    ca, cb = _cascade(mk.avec(retro=ra / 100), s), _cascade(mk.avec(retro=rb / 100), s)
    va, vb = _resultat_pa(s, ca), _resultat_pa(s, cb)
    d = 0 if s.produit == CP else 2
    k = st.columns(3)
    _metriques_resultat(s, ca, k[0], f"Réseau A, rétro {ra:,.1f}%")
    _metriques_resultat(s, cb, k[1], f"Réseau B, rétro {rb:,.1f}%")
    k[2].metric("Écart pour le client", f"{(vb - va) * 100:+,.{d}f} pt",
                help=f"{abs(rb - ra):,.1f} pt de rétro par an sur {vie} = "
                     f"{G.eur(abs(rb - ra) / 100 * Tf * N)} pris sur le budget option.")

    # le resultat est affine dans les frais : une droite, exacte, ancree sans retro
    c0 = _resultat_pa(s, _cascade(mk.avec(retro=0.0), s))
    xr = np.linspace(0.0, 2.5, 26)
    yr = np.maximum(c0 - xr * cout, 0.0) * 100
    fmt = (lambda v: f"{v:,.0f} %") if s.produit == CP else (lambda v: f"{v:,.2f} %")
    nom = "Participation offerte" if s.produit == CP else "Coupon offert (p.a.)"
    graphe = G.courbes(
        xr, [{"nom": nom, "y": yr, "couleur": G.PRODUIT, "epaisseur": 2.8}],
        f"Le même produit : {fmt(va * 100)} chez A, {fmt(vb * 100)} chez B",
        f"Chaque point de rétro par an coûte {cout * 100:,.{d}f} pt de {quoi} au client"
        + ("" if s.produit == CP else " par an"),
        "Rétrocession du distributeur (% par an)",
        "Participation offerte (%)" if s.produit == CP else "Coupon offert (% p.a.)",
        fmt_x=lambda v: f"{v:,.1f} % par an", fmt_y=fmt, nom_x="Rétrocession",
        niveaux_v=[(ra, f"réseau A  {ra:,.1f} %", G.NIVEAU), (rb, f"réseau B  {rb:,.1f} %",
                                                             G.NIVEAU)],
        marqueurs=[(ra, va * 100, G.PRODUIT), (rb, vb * 100, G.PRODUIT)],
        lignes_infobulle=[("Rétro payée sur la durée",
                           [G.eur(v / 100 * Tf * N) for v in xr])],
        legende=False, hauteur=320)
    G.affiche(graphe)

    comp = [{"Composante": la.libelle, "Réseau A": G.eur(la.montant),
             "Réseau B": G.eur(lb.montant), "Écart": G.eur(lb.montant - la.montant, signe=True)}
            for la, lb in zip(ca.lignes, cb.lignes)]
    st.table(comp, hide_index=True)
    st.info(f"**Pourquoi deux {quoi}s différents.** La rétro est payée d'avance sur toute la "
            f"durée : {abs(rb - ra):,.1f} pt par an sur {vie}, c'est "
            f"{G.eur(abs(rb - ra) / 100 * Tf * N)} de moins dans le budget option pour le "
            f"réseau B. La banque achète donc moins d'optionalité pour le client : le {quoi} "
            "baisse, la marge de la banque ne bouge pas.\n\n"
            "**À dire à un client qui compare deux offres** : à structure égale, l'écart de "
            f"{quoi} mesure l'écart de frais de distribution, pas la qualité de l'émetteur.")


# =====================================================================
# 3. REVENDRE EN COURS DE VIE
# =====================================================================

def _section_revente(mk, s):
    T, N = s.T, mk.nominal
    mois_max = max(int(round(T * 12)) - 1, 0)
    c = st.columns(3)
    mois = c[0].slider("Temps écoulé (mois)", 0, mois_max, min(mois_max, int(T * 12) // 2), 1,
                       key="as_sec_mois", bind="query-params",
                       help="Depuis l'émission. À 0, on mesure ce que coûte une revente le "
                            "lendemain du lancement.")
    niveau = c[1].slider("Sous-jacent aujourd'hui (% du niveau initial)", 30, 160, 90, 1,
                         key="as_sec_niveau", bind="query-params",
                         help="Le niveau du jour" + (" (du plus mauvais du panier)"
                                                    if s.produit == AC and s.n_actifs > 1 else "")
                              + ". C'est lui qui pèse le plus sur la valeur.")
    fourchette = c[2].slider("Fourchette de rachat (%)", 0.0, 3.0, 1.0, 0.25,
                             key="as_sec_fourchette", bind="query-params",
                             help="Écart entre le prix d'achat et le prix de vente de la "
                                  "banque. Le client qui revend reçoit la valeur moins la "
                                  "moitié de la fourchette.")
    t = min(mois, mois_max) / 12

    touchee, observes = False, ()
    if s.produit == BRC:
        if not s.continu:
            st.caption("Barrière européenne : seul le niveau final compte, l'historique ne "
                       "change rien.")
        elif niveau <= s.barriere * 100:
            st.caption(f"Le sous-jacent est sous la barrière ({s.barriere * 100:,.0f} %) : elle "
                       "est forcément touchée.")
            touchee = True
        else:
            touchee = st.checkbox("La barrière a déjà été touchée", key="as_sec_touchee",
                                  bind="query-params",
                                  help="Une fois touchée, la protection a disparu pour de bon : "
                                       "le produit se comporte comme une obligation plus un put "
                                       "vanille vendu, même si le marché est remonté depuis.")
    elif s.produit == AC:
        dt_obs = T / s.n_obs
        passees = sum(1 for k in range(s.n_obs) if (k + 1) * dt_obs <= t + 1e-9)
        if passees:
            st.markdown(f"**Niveaux aux {passees} observation(s) déjà passée(s)** (% du niveau "
                        "initial" + (", plus mauvais du panier" if s.n_actifs > 1 else "") + ")")
            cols = st.columns(min(passees, 6))
            vals = []
            for k in range(passees):
                vals.append(cols[k % len(cols)].number_input(
                    f"Obs {k + 1} ({(k + 1) * dt_obs * 12:,.0f} mois)", 20, 200, 95, 1,
                    key=f"as_sec_obs{k + 1}", bind="query-params",
                    help="Au-dessus du trigger, le produit a été rappelé à cette date. "
                         "Au-dessus de la barrière de coupon, le coupon (et ceux en mémoire) "
                         "a été versé.") / 100)
            observes = tuple(vals)
            etat = S.etat_autocall(observes, s.trigger, s.bar_coupon, s.memoire)
            if etat["vivant"]:
                st.caption(f"{etat['coupons_verses']} coupon(s) déjà versé(s), "
                           f"{etat['manques']} en attente" + (" (mémoire)" if s.memoire else "")
                           + ".")

    st.markdown("**Le marché depuis l'émission**")
    c = st.columns(3)
    dvol = c[0].slider("Volatilité (pts)", -10, 20, 0, 1, key="as_sec_dvol", bind="query-params",
                       help="Hausse de la vol depuis l'émission. Le client est vendeur "
                            "d'options (BRC, autocall) : la vol qui monte fait baisser la "
                            "valeur. Il est acheteur sur un Capital Protection.")
    dtaux = c[1].slider("Taux (bp)", -200, 200, 0, 25, key="as_sec_dtaux", bind="query-params",
                        help="Hausse des taux : l'obligation sous-jacente vaut moins, surtout "
                             "sur les produits longs.")
    dspread = c[2].slider("Spread émetteur (bp)", -100, 400, 0, 25, key="as_sec_dspread",
                          bind="query-params",
                          help="Si la signature de la banque se dégrade, le marché exige plus "
                               "pour porter son risque : la valeur du produit baisse.")
    choc = {"vol_atm": max(mk.vol_atm + dvol / 100, 0.04), "r": mk.r + dtaux / 10000,
            "spread": max(mk.spread + dspread, 0)}
    if mk.bates:
        choc["theta_v"] = max(mk.theta_v + dvol / 100, 0.04)
    mk_j = mk.avec(**choc)

    v = _secondaire(mk, mk_j, s, t, niveau / 100, touchee, observes, fourchette / 100)
    if not v.decomposition:          # autocall deja rappele
        st.success(v.message)
        st.metric("Coupons encaissés", G.eur(v.coupons_encaisses),
                  help="Plus le nominal rendu à la date de rappel.")
        return

    k = st.columns(4)
    k[0].metric("Valeur juste", G.eur(v.valeur), f"{v.valeur / N * 100:,.1f}% du nominal",
                delta_color="off", help="Valeur des flux restants, sur les mêmes tirages que "
                                        "le pricing à l'émission.")
    k[1].metric("Prix de rachat", G.eur(v.prix_rachat),
                help="Ce que la banque paie au client qui revend : valeur moins la "
                     "demi-fourchette.")
    k[2].metric("Bilan si revente", G.eur(v.bilan_client, signe=True),
                help="Prix de rachat plus coupons déjà encaissés, moins le nominal investi.")
    if s.produit == CP:
        k[3].metric("Dont obligation", G.eur(v.infos["obligation"]),
                    help="La part de la valeur qui garantit la protection à l'échéance. Elle "
                         "baisse quand les taux ou le spread montent.")
    else:
        k[3].metric("Proba de perte à l'échéance", f"{v.infos['proba_perte'] * 100:,.1f}%",
                    help="Vue d'aujourd'hui, sous le modèle de la barre latérale.")

    etapes = [{"court": "À l'émission", "libelle": "Valeur juste à l'émission",
               "valeur": v.valeur_emission, "total": True, "couleur": G.PRODUIT,
               "pourquoi": "Nominal moins marge et rétro, payées d'avance."}]
    pourquoi = {"Temps écoulé": "Le temps qui passe : coupons qui courent, options vendues qui "
                                "perdent de leur valeur, obligation qui se rapproche du pair.",
                "Sous-jacent": "Le niveau du sous-jacent (et l'historique de la barrière ou "
                               "des observations).",
                "Volatilité": "Le changement de vol depuis l'émission.",
                "Taux et dividende": "Le changement de taux depuis l'émission.",
                "Spread émetteur": "Le changement de spread de crédit de l'émetteur."}
    for nom, effet in v.decomposition:
        etapes.append({"court": nom.replace(" et dividende", "").replace(" émetteur", ""),
                       "libelle": nom, "valeur": effet,
                       "couleur": G.GAIN if effet >= 0 else G.PERTE, "pourquoi": pourquoi[nom]})
    etapes.append({"court": "Aujourd'hui", "libelle": "Valeur juste aujourd'hui", "total": True,
                   "valeur": v.valeur, "couleur": G.PRODUIT, "pourquoi": "Somme des lignes."})
    etapes.append({"court": "Fourchette", "libelle": "Demi-fourchette de rachat",
                   "valeur": v.prix_rachat - v.valeur, "couleur": G.PERTE,
                   "pourquoi": "Ce que la banque garde pour racheter le produit."})
    etapes.append({"court": "Rachat", "libelle": "Prix de rachat", "total": True,
                   "valeur": v.prix_rachat, "couleur": G.PRODUIT,
                   "pourquoi": "Ce que le client reçoit s'il revend aujourd'hui."})
    if v.coupons_encaisses:
        etapes.append({"court": "Coupons reçus", "libelle": "Coupons déjà encaissés",
                       "valeur": v.coupons_encaisses, "couleur": G.GAIN,
                       "pourquoi": "Versés aux observations passées."})
        etapes.append({"court": "Total client", "libelle": "Total pour le client", "total": True,
                       "valeur": v.prix_rachat + v.coupons_encaisses, "couleur": G.PRODUIT,
                       "pourquoi": "Prix de rachat plus coupons encaissés."})
    graphe = graphe_cascade(
        etapes, f"Revendre aujourd'hui : {G.eur(v.prix_rachat)}, soit "
                f"{G.eur(v.bilan_client, signe=True)} pour le client",
        "Ce qui a fait bouger la valeur depuis l'émission, facteur par facteur, sur les mêmes "
        "tirages. Axe coupé.", nominal=N,
        legende=[("Fait monter la valeur", G.GAIN), ("Fait baisser la valeur", G.PERTE),
                 ("Valeur du produit", G.PRODUIT)])
    G.affiche(graphe)
    st.caption(v.message)

    # autocall : frais pris sur la duree de vie attendue (= valeur a l'emission)
    Tf = _vie_autocall(mk, s)[0] if s.produit == AC else T
    frais = (mk.marge + mk.retro) * Tf * N
    if s.produit == CP:
        st.info(f"**La protection ne joue qu'à l'échéance.** Avant, le produit vaut une "
                f"obligation ({G.eur(v.infos['obligation'])} aujourd'hui) plus des options : "
                "une hausse des taux ou du spread le fait passer sous le niveau garanti. Le "
                "client « garanti » qui revend avant l'échéance peut perdre.")
    else:
        st.info(f"**Revendre le lendemain de l'émission coûte déjà {G.eur(frais + fourchette / 200 * N)}** : "
                f"marge ({G.eur(mk.marge * Tf * N)}) et rétro ({G.eur(mk.retro * Tf * N)}) "
                f"payées d'avance, plus la demi-fourchette ({G.eur(fourchette / 200 * N)}). "
                "Un produit structuré se garde jusqu'à l'échéance ou au rappel.")


# =====================================================================
# 4. PITCH CLIENT
# =====================================================================

def faits(mk, s):
    """Les chiffres dont le pitch a besoin, tires des memes calculs que les autres sections."""
    cascade = _cascade(mk, s)
    lignes = {l.cle: l.montant for l in cascade.lignes}
    f = {"N": mk.nominal, "T": s.T, "retro": mk.retro, "marge": mk.marge, "spread": mk.spread,
         "recovery": mk.recovery, "r": mk.r, "q": mk.q, "credit": abs(lignes.get("credit", 0.0)),
         "taux": mk.nominal - lignes["zc"], "zc": lignes["zc"],
         "cout_retro": cascade.infos["cout_retro_par_pt"],
         "resultat": _resultat_pa(s, cascade),
         # frais payes d'avance, tels que la cascade les preleve (autocall : sur la vie attendue)
         "marge_eur": lignes["marge"], "retro_eur": lignes["retro"],
         "frais": lignes["marge"] + lignes["retro"]}
    if s.produit == BRC:
        f.update(put=-lignes["put"], p_perte=cascade.infos["proba_perte"],
                 p_touche=cascade.infos["proba_touche"])
    elif s.produit == AC:
        duree, rappel_1 = _vie_autocall(mk, s)
        f.update(put=-lignes["put"], p_perte=cascade.infos["proba_perte"], duree=duree,
                 rappel_1=rappel_1, coupon_obs=cascade.resultat)
    else:
        f.update(call=lignes["call"], call_cap=-lignes["cap"])
    return f


def _adequation(s, vue, f, sj="le sous-jacent", recupere="le client récupère"):
    """(niveau d'adequation, reponse a la vue du client), selon le produit."""
    c = f["resultat"]
    if s.produit == BRC:
        return {
            "Forte hausse": ("non", f"ce produit n'est pas le bon, le gain est plafonné au "
                                    f"coupon ({_p(c * s.T)} au total) ; une participation à la "
                                    "hausse, Tracker ou Capital Protection, colle mieux à "
                                    "cette vue"),
            "Hausse modérée": ("oui", f"le coupon de {_p(c, 2)} par an est acquis même si la "
                                      "hausse n'arrive pas"),
            "Stabilité": ("oui", "c'est exactement le scénario que ce produit rémunère"),
            "Baisse modérée": ("oui" if s.barriere <= 0.75 else "partiel",
                               f"une baisse jusqu'à {_p(1 - s.barriere, 0)} ne coûte rien, le "
                               "coupon est versé quand même" + (
                                   "" if s.barriere <= 0.75 else
                                   ", mais une barrière a " + _p(s.barriere, 0) +
                                   " laisse peu de marge")),
            "Forte baisse": ("non", "c'est précisément le scénario où ce produit perd ; mieux "
                                    "vaut rester liquide ou acheter de la protection"),
        }[vue]
    if s.produit == AC:
        return {
            "Forte hausse": ("partiel", f"le produit sera sans doute rappelé dès la première "
                                        f"date avec un coupon de {_p(f['coupon_obs'], 2)} : le "
                                        "capital revient vite, mais le gain s'arrête la"),
            "Hausse modérée": ("oui", f"au-dessus de {_p(s.trigger, 0)} à une date "
                                      "d'observation, le produit rembourse avec son coupon, "
                                      f"{_p(f['rappel_1'], 0)} de chances dès la première date"),
            "Stabilité": ("oui" if s.trigger <= 1.0 else "partiel",
                          f"le coupon tombe tant que {sj} reste au-dessus de "
                          f"{_p(s.bar_coupon, 0)}" + (
                              ", et un marché stable suffit à déclencher le rappel"
                              if s.trigger <= 1.0 else
                              ", mais un trigger à " + _p(s.trigger, 0) + " demande de la hausse "
                              "pour être rappelé")),
            "Baisse modérée": ("oui" if s.bar_coupon <= 0.80 else "partiel",
                               f"tant que {sj} reste au-dessus de "
                               f"{_p(s.bar_coupon, 0)} aux dates d'observation, le coupon est "
                               "versé" + (", et la mémoire rattrape les coupons manqués"
                                          if s.memoire else "")),
            "Forte baisse": ("non", f"sous {_p(s.bar_capital, 0)} à l'échéance, toute la "
                                    "baisse est subie ; ce n'est pas le produit pour cette vue"),
        }[vue]
    gain = (", sans plafond" if s.cap >= 2.995 else
            f", jusqu'à un gain maximum de {_p(c * (s.cap - 1), 0)}")
    return {
        "Forte hausse": ("oui" if s.cap >= 1.5 else "partiel",
                         f"le produit capte {_p(c, 0)} de la hausse{gain}"),
        "Hausse modérée": ("oui", f"le capital est protégé à {_p(s.protection, 0)} et la hausse "
                                  f"est captée à {_p(c, 0)}"),
        "Stabilité": ("non", f"sans mouvement de marché, {recupere} {_p(s.protection, 0)} "
                             f"sans rien gagner pendant {_duree(s.T)} ; un dépôt à terme fait "
                             "mieux"),
        "Baisse modérée": ("partiel", f"la protection à {_p(s.protection, 0)} joue à "
                                      "l'échéance, mais le placement ne rapporte rien si la "
                                      "baisse se confirme"),
        "Forte baisse": ("partiel", "la protection limite la casse à l'échéance, mais sans "
                                    "aucun rendement : se protéger coûte la renonciation aux "
                                    "intérêts"),
    }[vue]


def pitch(s, profil, vue, f):
    """
    Trois a cinq phrases, construites sur des modeles a partir du produit,
    de ses chiffres (f, voir faits), du profil du client et de sa vue de
    marche. Renvoie {"phrases": [(role, texte)], "adequation": oui/partiel/non}.
    Roles toujours presents : "vue", "gain", "perte", "emetteur".
    """
    N, T, c = f["N"], s.T, f["resultat"]
    duree = _duree(T)
    sj = _sous_jacent(s) if profil == PARTICULIER else (
        f"le plus faible des {s.n_actifs} sous-jacents" if s.produit == AC and s.n_actifs > 1
        else "le sous-jacent")
    niveau_adequation, reponse = _adequation(
        s, vue, f, sj, "vous récupérez" if profil == PARTICULIER else "le client récupère")
    vue_txt = VUES[vue]
    if profil == PARTICULIER:
        accroche = f"Vous pensez que {vue_txt} : {reponse}."
    elif profil == CGP:
        accroche = f"Pour un client qui pense que {vue_txt}, {reponse}."
    else:
        accroche = f"Sur une vue « {vue.lower()} », {reponse}."
    accroche = accroche[0].upper() + accroche[1:]
    phrases = [("vue", accroche)]
    rec = _p(f["recovery"], 0)

    if s.produit == BRC:
        L = max(s.barriere - 0.10, 0.20)
        rendu = N * L / s.strike + c * T * N
        if s.continu:
            rendu_entier = (f"s'il ne perd jamais plus de {_p(1 - s.barriere, 0)} en cours de "
                            f"route, ou s'il finit au-dessus de {_p(s.strike, 0)} de son niveau "
                            "de départ")
        else:
            rendu_entier = (f"s'il ne finit pas plus de {_p(1 - s.barriere, 0)} sous son niveau "
                            "de départ")
        if profil == PARTICULIER:
            phrases += [
                ("gain", f"Vous touchez {_p(c, 2)} par an pendant {duree}, soit "
                         f"{G.eur(c * T * N)} pour {G.eur(N)} investis, quoi que fasse "
                         f"l'indice ; votre capital est rendu en entier {rendu_entier}."),
                ("perte", f"Le revers : si l'indice passe sous {_p(s.barriere, 0)} de son niveau "
                          f"de départ et finit sous {_p(s.strike, 0)}, vous perdez comme lui, "
                          f"par exemple {G.eur(rendu)} rendus, coupon compris, s'il finit à "
                          f"{_p(L, 0)} ; dans nos simulations, cela arrive dans "
                          f"{_p(f['p_perte'], 0)} des cas."),
            ]
        elif profil == CGP:
            phrases += [
                ("gain", f"Coupon garanti de {_p(c, 2)} par an sur {duree}, barrière "
                         f"{_p(s.barriere, 0)} {'continue' if s.continu else 'europeenne'}, "
                         f"strike {_p(s.strike, 0)}."),
                ("perte", f"Scénario de perte à présenter clairement : barrière franchie et "
                          f"sous-jacent sous le strike à l'échéance, le client perd comme le "
                          f"sous-jacent ({G.eur(rendu)} rendus pour un fixing à {_p(L, 0)}) ; "
                          f"probabilité de perte en capital de {_p(f['p_perte'], 0)} dans le "
                          "modèle."),
            ]
        else:
            credit = (f" et le risque de crédit ({G.eur(f['credit'])}) s'ajoutent"
                      if f["credit"] >= 0.5 else " s'ajoute")
            phrases += [
                ("gain", f"Coupon de {_p(c, 2)} p.a. sur {duree} : la prime du put "
                         f"down-and-in {_p(s.barriere, 0)} "
                         f"{'continu' if s.continu else 'europeen'} strike {_p(s.strike, 0)} "
                         f"({G.eur(f['put'])}){credit} aux {G.eur(f['taux'])} que donnent les "
                         "taux."),
                ("perte", f"Perte en capital si knock-in et fixing final sous "
                          f"{_p(s.strike, 0)} : probabilité risque-neutre {_p(f['p_perte'])}, "
                          f"knock-in {_p(f['p_touche'])}, soit {G.eur(rendu)} rendus coupon "
                          f"compris pour un fixing à {_p(L, 0)}."),
            ]
    elif s.produit == AC:
        L = max(s.bar_capital - 0.10, 0.20)
        sj = _sous_jacent(s)
        per = PERIODE[s.par_an]
        if profil == PARTICULIER:
            phrases += [
                ("gain", f"Chaque {per}, si {sj} est au-dessus de {_p(s.bar_coupon, 0)} de son "
                         f"niveau de départ, vous touchez {_p(f['coupon_obs'], 2)} "
                         f"({_p(c, 2)} par an)" + (", et les coupons manqués sont rattrapés"
                                                   if s.memoire else "")
                         + f" ; au-dessus de {_p(s.trigger, 0)}, le produit s'arrête et vous "
                           f"récupérez votre capital, en moyenne après {f['duree']:,.1f} an(s) "
                           f"sur {duree} possibles."),
                ("perte", f"Le revers : si à l'échéance {sj} a perdu plus de "
                          f"{_p(1 - s.bar_capital, 0)}, vous subissez toute la baisse, par "
                          f"exemple {G.eur(N * L)} rendus pour {G.eur(N)} s'il finit à "
                          f"{_p(L, 0)} ; dans nos simulations, cela arrive dans "
                          f"{_p(f['p_perte'], 0)} des cas."),
            ]
        elif profil == CGP:
            phrases += [
                ("gain", f"Coupon conditionnel de {_p(f['coupon_obs'], 2)} par {per} "
                         f"({_p(c, 2)} par an), barrière de coupon {_p(s.bar_coupon, 0)}"
                         + (" avec mémoire" if s.memoire else " sans mémoire")
                         + f", rappel à {_p(s.trigger, 0)}, durée de vie moyenne "
                           f"{f['duree']:,.1f} an(s) sur {duree}."),
                ("perte", f"Scénario de perte à présenter clairement : sous "
                          f"{_p(s.bar_capital, 0)} à l'échéance, le client subit toute la "
                          f"baisse de {sj} ({G.eur(N * L)} rendus pour un fixing à "
                          f"{_p(L, 0)}) ; probabilité de perte en capital de "
                          f"{_p(f['p_perte'], 0)} dans le modèle."),
            ]
        else:
            panier = (f" sur un worst-of de {s.n_actifs} sous-jacents (corrélation "
                      f"{s.correlation:.2f})" if s.n_actifs > 1 else "")
            phrases += [
                ("gain", f"Coupon de {_p(f['coupon_obs'], 2)} par observation ({_p(c, 2)} "
                         f"p.a.){panier}, trigger {_p(s.trigger, 0)}, barrière de coupon "
                         f"{_p(s.bar_coupon, 0)}" + (" mémoire" if s.memoire else "")
                         + f" ; finance par un put down-and-in européen {_p(s.bar_capital, 0)} "
                           f"({G.eur(f['put'])}) ; durée de vie moyenne {f['duree']:,.2f} an."),
                ("perte", f"Perte en capital si le fixing final {'du worst-of ' if s.n_actifs > 1 else ''}"
                          f"est sous {_p(s.bar_capital, 0)} : probabilité risque-neutre "
                          f"{_p(f['p_perte'])}, perte linéaire dès la barrière "
                          f"({G.eur(N * L)} rendus pour un fixing à {_p(L, 0)})."),
            ]
    else:
        gain_max = ("" if s.cap >= 2.995 else
                    f", dans la limite d'un gain de {_p(c * (s.cap - 1), 0)}")
        if profil == PARTICULIER:
            phrases.append(
                ("gain", f"A l'échéance, dans {duree}, vous récupérez au moins "
                         f"{_p(s.protection, 0)} de votre capital, plus {_p(c, 0)} de la hausse "
                         f"de l'indice{gain_max}."))
            if s.protection < 0.9995:
                phrases.append(
                    ("perte", f"Le revers : si l'indice baisse, vous perdez jusqu'à "
                              f"{_p(1 - s.protection, 0)} de votre capital, soit "
                              f"{G.eur(N * (1 - s.protection))}, sans avoir touche d'intérêts "
                              f"pendant {duree}."))
            else:
                phrases.append(
                    ("perte", f"Le scénario de perte : si l'indice finit sous son niveau de "
                              f"départ, vous récupérez votre capital sans rien gagner, alors "
                              f"qu'un placement sans risque à {_p(f['r'], 1)} aurait rapporté "
                              f"{G.eur(N * (math.exp(f['r'] * T) - 1))} ; et une revente avant "
                              "l'échéance se fait au prix du marché, qui peut être sous le "
                              "capital."))
        elif profil == CGP:
            phrases.append(
                ("gain", f"Protection de {_p(s.protection, 0)} du capital à l'échéance et "
                         f"participation de {_p(c, 0)} à la hausse{gain_max}, sur {duree}."))
            if s.protection < 0.9995:
                perte = (f"Scénario de perte : jusqu'à {_p(1 - s.protection, 0)} du capital si "
                         "le sous-jacent finit en baisse, et une valeur de revente qui peut "
                         "passer sous la protection si les taux montent.")
            else:
                perte = (f"Scénario de perte : aucun à l'échéance hors défaut, mais un coût "
                         f"d'opportunité si le marché baisse (aucun intérêt pendant {duree}), "
                         "et une valeur de revente qui peut passer sous le pair si les taux "
                         "montent.")
            phrases.append(("perte", perte))
        else:
            phrases.append(
                ("gain", f"Participation de {_p(c, 0)} ({_p(s.protection, 0)} de protection"
                         + ("" if s.cap >= 2.995 else f", cap {_p(s.cap, 0)}")
                         + f") : zéro-coupon au taux de financement de l'émetteur plus call "
                           f"spread acheté pour {G.eur(f['call'] - f['call_cap'])}."))
            phrases.append(
                ("perte", (f"Perte maximale de {_p(1 - s.protection, 0)} du nominal à "
                           "l'échéance, atteinte dès que le sous-jacent finit sous son niveau "
                           "initial" if s.protection < 0.9995 else
                           "Pas de perte en capital à l'échéance hors défaut ; le scénario de "
                           "perte est le mark-to-market")
                          + f" : une hausse des taux ou du spread fait passer la valeur sous le "
                            f"pair en cours de vie (duration de l'ordre de {s.T:g} ans)."))

    # risque emetteur : toujours nomme
    finance = "le coupon" if s.produit != CP else "la participation"
    if profil == PARTICULIER:
        txt = ("Votre remboursement dépend aussi de l'émetteur, la banque qui émet le produit : "
               "si elle fait faillite, " + ("même la protection ne joue plus et " if s.produit == CP
                                            else "")
               + f"vous ne récupérez qu'une partie de votre mise, de l'ordre de {rec}")
        txt += (f" ; c'est d'ailleurs ce risque qui finance une partie de {finance}."
                if f["spread"] > 0 else ".")
    elif profil == CGP:
        txt = (f"Risque émetteur : le client est créancier non garanti de la banque, avec un "
               f"recouvrement de l'ordre de {rec} en cas de défaut")
        txt += (f" ; les {f['spread']:,.0f} bp de spread apportent {G.eur(f['credit'])} au "
                "budget de la structure." if f["spread"] > 0 else ".")
    else:
        if f["spread"] > 0:
            txt = (f"Exposition au crédit de l'émetteur, senior non sécurisé : spread "
                   f"{f['spread']:,.0f} bp, recouvrement {rec}, soit {G.eur(f['credit'])} de "
                   f"valeur sur {G.eur(N)}.")
        else:
            txt = (f"""Exposition au crédit de l'émetteur, senior non sécurisé, recouvrement supposé {rec} ; le spread est à 0 bp dans cette vue de marché, chaque point de spread se retrouverait dans {finance}.""")
    phrases.append(("emetteur", txt))

    # une cinquieme phrase selon le profil
    if profil == CGP:
        d = 0 if s.produit == CP else 2
        if f["retro"] > 0:
            sans = c + f["retro"] * 100 * f["cout_retro"]
            phrases.append(("retro", f"La rétrocession de {_p(f['retro'])} par an est incluse "
                                     f"dans le prix : sans elle, {finance} serait de "
                                     f"{_p(sans, d)}" + ("" if s.produit == CP else " par an")
                                     + "."))
        else:
            phrases.append(("retro", f"""Aucune rétrocession n'est prévue : {finance} affiché est brut de frais de distribution."""))
    elif profil == INSTIT:
        frais = f["frais"]
        phrases.append(("liquidite", f"Valeur juste à l'émission {G.eur(N - frais)} pour "
                                     f"{G.eur(N)} ({G.eur(frais)} de marge et rétro payées "
                                     "d'avance) ; rachat quotidien par l'émetteur, fourchette "
                                     "de l'ordre de 1 %."))
    return {"phrases": phrases, "adequation": niveau_adequation}


def _section_pitch(mk, s):
    c = st.columns([3, 2])
    profil = _choix_profil(c[0])
    vue = c[1].selectbox("Vue de marché du client", list(VUES), index=2, key="as_vue",
                         bind="query-params",
                         help="Ce que le client attend du marché. Le pitch dit si le produit y "
                              "répond, et le dit aussi quand il n'y répond pas.")
    f = faits(mk, s)
    p = pitch(s, profil, vue, f)
    texte = " ".join(t for _, t in p["phrases"])
    with st.container(border=True):
        st.markdown(texte)
    if p["adequation"] == "oui":
        st.success("Produit adapté à la vue du client.")
    elif p["adequation"] == "partiel":
        st.warning("Produit partiellement adapté : le pitch dit ce que le client abandonne.")
    else:
        st.error("Produit mal adapté à cette vue : le pitch le dit et oriente vers autre chose.")
    st.caption(f"{len(p['phrases'])} phrases, construites sur des modèles à partir des chiffres "
               "des autres sections. Toujours présents : le scénario de perte chiffré et le "
               "risque émetteur.")
    with st.expander("Texte à copier"):
        st.code(texte, language=None, wrap_lines=True)


# =====================================================================
# 5. OBJECTIONS CLIENTS
# =====================================================================

# Euro Stoxx 50 en 2008 (data/eurostoxx50.csv) : -44,4 % du 31/12/2007 au
# 31/12/2008, plus bas a 49,2 % du niveau de depart le 21/11/2008
BAISSE_2008, PLUS_BAS_2008 = 0.444, 0.492

QUESTIONS = {
    "piege": "Pourquoi ce coupon est-il si élevé, où est le piège ?",
    "piege_cp": "Capital garanti et la hausse en plus : où est le piège ?",
    "faillite": "Et si la banque fait faillite ?",
    "etf": "Pourquoi pas simplement un ETF ?",
    "marge": "Combien vous gagnez dessus ?",
    "sortie": "Je peux sortir avant l'échéance ?",
    "krach": "Et si le marché s'effondre comme en 2008 ?",
    "hausse": "Pourquoi je ne profite pas de toute la hausse ?",
    "concurrent": "Une autre banque me propose un meilleur coupon.",
    "concurrent_cp": "Une autre banque me propose une meilleure participation.",
    "barriere": "Avec la barrière, ma perte est limitée, non ?",
    "rappel": "Pourquoi le produit s'arrête-t-il tout seul ?",
    "garanti": "Capital garanti, donc je ne peux rien perdre ?",
}
LISTE_OBJECTIONS = {
    BRC: ["piege", "faillite", "etf", "marge", "sortie", "krach", "hausse", "concurrent",
          "barriere"],
    AC: ["piege", "rappel", "faillite", "etf", "marge", "sortie", "krach", "hausse",
         "concurrent", "barriere"],
    CP: ["piege_cp", "garanti", "faillite", "etf", "marge", "sortie", "krach", "hausse",
         "concurrent_cp"],
}


def _ans(x):
    """0.92 -> '0.9 an', 2.5 -> '2.5 ans'."""
    return f"{x:,.1f} an" + ("s" if x >= 2 else "")


def objection(s, profil, cle, f):
    """
    La reponse d'un bon Sales a une objection, chiffree avec les valeurs du
    produit (f, voir faits). Particulier et conseiller : des mots simples
    (le conseiller les redit a son client), plus une phrase pour lui quand
    c'est utile ; institutionnel : le mecanisme et les grandeurs du desk.
    Renvoie {"reponse", "chiffre", "erreur"}.
    """
    N, T, c = f["N"], s.T, f["resultat"]
    inst, cgp = profil == INSTIT, profil == CGP
    duree = _duree(T)
    sj = _sous_jacent(s) if profil == PARTICULIER else (
        f"le plus faible des {s.n_actifs} sous-jacents" if s.produit == AC and s.n_actifs > 1
        else "le sous-jacent")
    rec, r = _p(f["recovery"], 0), _p(f["r"], 1)
    quoi = "la participation" if s.produit == CP else "le coupon"
    val = _p(c, 0) if s.produit == CP else _p(c, 2)
    par_an = "" if s.produit == CP else " par an"
    frais = f["frais"]
    vie = f["duree"] if s.produit == AC else T
    sur_vie = (f"sur la durée de vie attendue de {_ans(vie)}" if s.produit == AC
               else f"sur {duree}")
    valeur0 = N - frais
    rachat0 = valeur0 - 0.005 * N          # fourchette de 1 %, comme la section Revendre
    sans = c + (f["marge"] + f["retro"]) * 100 * f["cout_retro"]
    interets = N * (math.exp(f["r"] * T) - 1)
    d = 0 if s.produit == CP else 2

    if cle == "piege":
        if s.produit == BRC:
            if inst:
                rep = (f"Le coupon est la prime d'un put down-and-in {_p(s.barriere, 0)} strike "
                       f"{_p(s.strike, 0)} vendu par le client ({G.eur(f['put'])} sur {G.eur(N)}), "
                       f"ajoutée aux {G.eur(f['taux'])} que donnent les taux. Pas de levier "
                       f"caché : probabilité risque-neutre de perte en capital {_p(f['p_perte'])}, "
                       f"de knock-in {_p(f['p_touche'])}.")
                chiffre = (f"Put vendu : {G.eur(f['put'])} ; perte en capital dans "
                           f"{_p(f['p_perte'])} des scénarios.")
            else:
                rep = (f"Le piège n'est pas caché, c'est le risque que vous prenez : {val} par an "
                       "rémunère l'assurance que vous vendez à la banque contre une forte baisse "
                       f"de l'indice. Sans risque, on obtient {r} par an aujourd'hui ; l'écart "
                       f"paie ce risque. Si l'indice passe sous {_p(s.barriere, 0)} de son niveau "
                       f"de départ et finit sous {_p(s.strike, 0)}, vous perdez comme lui : dans "
                       f"nos simulations, cela arrive dans {_p(f['p_perte'], 0)} des cas.")
                chiffre = (f"{val} par an contre {r} sans risque ; perte en capital dans "
                           f"{_p(f['p_perte'], 0)} des cas.")
            erreur = ("Répondre « il n'y a pas de piège », ou parler du coupon sans nommer le "
                      "scénario de perte : le coupon élevé est exactement le prix du risque.")
        else:
            if inst:
                rep = (f"Le coupon de {_p(f['coupon_obs'], 2)} par observation ({val} par an) est "
                       f"financé par un put down-and-in européen {_p(s.bar_capital, 0)} vendu par "
                       f"le client ({G.eur(f['put'])}) et par le rappel, qui raccourcit la durée "
                       f"de vie attendue à {_ans(f['duree'])}. Probabilité risque-neutre de perte "
                       f"en capital : {_p(f['p_perte'])}.")
                chiffre = (f"Put vendu : {G.eur(f['put'])} ; perte en capital dans "
                           f"{_p(f['p_perte'])} des scénarios.")
            else:
                rep = (f"Le piège n'est pas caché, c'est le risque que vous prenez : {val} par an "
                       "rémunère l'assurance que vous vendez à la banque contre une forte baisse "
                       f"à l'échéance, et le coupon n'est versé que si {sj} reste au-dessus de "
                       f"{_p(s.bar_coupon, 0)}. Sans risque, on obtient {r} par an aujourd'hui. "
                       f"Si {sj} a perdu plus de {_p(1 - s.bar_capital, 0)} à l'échéance, vous "
                       f"subissez toute la baisse : dans nos simulations, cela arrive dans "
                       f"{_p(f['p_perte'], 0)} des cas.")
                chiffre = (f"{val} par an, conditionnel, contre {r} sans risque ; perte en capital "
                           f"dans {_p(f['p_perte'], 0)} des cas.")
            erreur = ("Présenter le coupon comme acquis : il est conditionnel, et sous la "
                      "barrière de capital la perte part du niveau de départ, pas de la barrière.")
    elif cle == "piege_cp":
        if inst:
            rep = (f"Zéro-coupon plus call spread : {G.eur(f['zc'])} placés pour rendre "
                   f"{_p(s.protection, 0)} du nominal, {G.eur(f['call'] - f['call_cap'])} "
                   f"d'options, {G.eur(frais)} de frais, pour {val} de participation. Le client "
                   f"finance son call avec le portage qu'il abandonne : {G.eur(interets)} au taux "
                   f"sans risque sur {duree}.")
        else:
            rep = (f"Le prix de la garantie, c'est ce que vous ne touchez pas : pendant {duree}, "
                   f"pas d'intérêts, alors qu'un placement sans risque à {r} rapporterait "
                   f"{G.eur(interets)} sur {G.eur(N)}. Vous recevez {val} de la hausse de "
                   "l'indice, sans les dividendes. Et la garantie est celle de la banque "
                   "émettrice : elle ne vaut que si la banque est là à l'échéance.")
        chiffre = (f"{G.eur(interets)} d'intérêts abandonnés sur {duree} ; participation {val}.")
        erreur = ("Dire « vous ne pouvez rien perdre » : le client renonce au rendement sans "
                  "risque, et le capital dépend de la survie de l'émetteur.")
    elif cle == "garanti":
        if s.protection < 0.9995:
            debut = (f"Non : la protection est de {_p(s.protection, 0)}, vous pouvez donc perdre "
                     f"jusqu'à {G.eur(N * (1 - s.protection))} à l'échéance si l'indice baisse.")
        else:
            debut = (f"À l'échéance, vous récupérez au moins {G.eur(N)}, à deux conditions : "
                     "garder le produit jusqu'au bout, et que la banque émettrice ne fasse pas "
                     "faillite.")
        if inst:
            rep = (f"Protection {_p(s.protection, 0)} à l'échéance, hors défaut de l'émetteur "
                   f"(recouvrement {rec}). En cours de vie, le produit vaut un zéro-coupon plus "
                   f"des options : {G.eur(valeur0)} le lendemain de l'émission, sous le pair, et "
                   "sensible aux taux et au spread.")
        else:
            rep = (debut + " Avant l'échéance, le prix suit les taux et le marché : le lendemain "
                   f"de l'émission, il vaut déjà environ {G.eur(valeur0)}.")
            if s.protection < 0.9995:
                rep += " La garantie, elle, dépend de la banque émettrice."
        chiffre = (f"Garanti à l'échéance : {G.eur(N * s.protection)} ; valeur le lendemain de "
                   f"l'émission : {G.eur(valeur0)}.")
        erreur = ("Dire « garanti » sans préciser « à l'échéance, et par l'émetteur » : c'est ce "
                  "qui revient le plus souvent en réclamation.")
    elif cle == "faillite":
        finance = "la participation" if s.produit == CP else "le coupon"
        if inst:
            if f["spread"] > 0:
                rep = (f"Exposition senior non sécurisée à l'émetteur : spread {f['spread']:,.0f} "
                       f"bp, recouvrement {rec}, soit {G.eur(f['credit'])} de valeur sur "
                       f"{G.eur(N)}, déjà dans {finance}. Pour la réduire : un émetteur mieux "
                       "noté, ou un produit collatéralisé, au prix d'un rendement plus bas.")
            else:
                rep = ("Exposition senior non sécurisée à l'émetteur, recouvrement supposé "
                       f"{rec} ; le spread est à 0 bp dans cette vue de marché, le risque n'est "
                       f"donc pas rémunéré dans {finance}. Pour la réduire : diversifier les "
                       "émetteurs, ou un produit collatéralisé.")
        else:
            rep = ("C'est un vrai risque, et il faut le dire : ce produit est une dette de la "
                   "banque émettrice. Si elle fait faillite, "
                   + ("la garantie ne joue plus et " if s.produit == CP else "")
                   + f"vous ne récupérez qu'une partie de votre mise, de l'ordre de {rec}, soit "
                   f"environ {G.eur(N * f['recovery'])} pour {G.eur(N)}.")
            if f["spread"] > 0:
                rep += (f" Ce risque est payé : il apporte {G.eur(f['credit'])} au budget du "
                        "produit.")
            else:
                rep += (" Avec un spread de 0 bp dans cette vue de marché, il n'est pas payé : "
                        "c'est une raison de répartir sur plusieurs émetteurs.")
            if cgp:
                rep += (" Pour votre allocation : regardez la notation et le spread de "
                        f"l'émetteur, pas seulement {finance}.")
        chiffre = (f"Recouvrement de l'ordre de {rec} : {G.eur(N * f['recovery'])} pour {G.eur(N)} ; "
                   + (f"spread {f['spread']:,.0f} bp, soit {G.eur(f['credit'])} dans le budget."
                      if f["spread"] > 0 else "spread à 0 bp, risque non rémunéré."))
        erreur = ("Dire « la banque ne peut pas faire faillite » : en 2008, Lehman Brothers "
                  "émettait aussi des produits à capital garanti. Le risque émetteur se nomme "
                  "toujours.")
    elif cle == "etf":
        div = f["q"] * vie * N
        if s.produit == CP:
            if inst:
                rep = (f"L'ETF est delta 1. Ici, le client est long {val} d'un call à la monnaie"
                       + ("" if s.cap >= 2.995 else f" plafonné à {_p(s.cap, 0)}")
                       + f", avec un plancher à {_p(s.protection, 0)} : il échange les dividendes "
                       f"({_p(f['q'])} par an) et une partie de la hausse contre la convexité.")
            else:
                rep = ("Un ETF vous donne toute la hausse de l'indice, mais aussi toute la baisse. "
                       f"Ici, vous récupérez au moins {_p(s.protection, 0)} de votre capital dans "
                       f"{duree}, et vous recevez {val} de la hausse ; vous renoncez aux dividendes "
                       f"({_p(f['q'])} par an) et à une partie de la hausse. C'est le choix de "
                       "quelqu'un qui ne peut pas se permettre de perdre.")
            chiffre = (f"Participation {val} contre 100 % pour l'ETF, avec {_p(s.protection, 0)} "
                       "du capital protégé à l'échéance.")
        else:
            seuil = s.barriere if s.produit == BRC else s.bar_capital
            qui = sj if s.produit == AC else "l'indice"
            if inst:
                rep = ("L'ETF est delta 1. Le produit vend de la volatilité à la baisse et renonce "
                       f"aux dividendes ({_p(f['q'])} par an, environ {G.eur(div)} {sur_vie}) "
                       f"contre {val} par an : il se justifie quand la vol implicite paraît chère. "
                       f"Frais : {G.eur(frais)} {sur_vie}.")
            else:
                rep = ("Un ETF vous donne toute la hausse et toute la baisse de l'indice, avec ses "
                       f"dividendes ({_p(f['q'])} par an), pour des frais faibles. Ce produit fait "
                       f"un autre pari : {val} par an même si l'indice ne monte pas, et votre "
                       f"capital rendu tant que {qui} ne "
                       f"perd pas plus de {_p(1 - seuil, 0)}. En "
                       f"échange, le gain est plafonné aux coupons et les frais sont plus élevés "
                       f"({G.eur(frais)} {sur_vie}).")
                if cgp:
                    rep += (" Pour un client qui attend une forte hausse, l'ETF reste le "
                            "meilleur choix.")
            chiffre = (f"Dividendes abandonnés : {_p(f['q'])} par an ; frais du produit : "
                       f"{G.eur(frais)} {sur_vie}.")
        erreur = ("Dénigrer l'ETF : pour un client qui attend une forte hausse, c'est souvent le "
                  "meilleur choix. Le produit structuré répond à une autre vue de marché.")
    elif cle == "marge":
        if inst:
            rep = (f"Marge banque {G.eur(f['marge_eur'])}, rétrocession "
                   f"{G.eur(f['retro_eur'])}, prélevées d'avance {sur_vie} : valeur juste à "
                   f"l'émission {G.eur(valeur0)} pour {G.eur(N)}. Sans frais, {quoi} serait de "
                   f"{_p(sans, d)}{par_an}. La marge paie aussi le risque que le desk garde : "
                   "gap, corrélation, vol de la vol.")
        else:
            rep = (f"La banque prend une marge de {G.eur(f['marge_eur'])} sur {G.eur(N)} "
                   f"({_p(f['marge'])} par an) et le distributeur une rétrocession de "
                   f"{G.eur(f['retro_eur'])} : {G.eur(frais)} au total {sur_vie}, prélevés au "
                   f"départ et déjà compris dans le prix. Sans ces frais, {quoi} serait de "
                   f"{_p(sans, d)}{par_an} au lieu de {val}.")
            if cgp:
                rep += (f" La rétrocession de {_p(f['retro'])} par an est la rémunération de "
                        f"votre cabinet : chaque point coûte {f['cout_retro'] * 100:,.{d}f} pt "
                        f"de {quoi.split()[-1]} à votre client.")
        chiffre = (f"{G.eur(frais)} de frais sur {G.eur(N)} (marge {G.eur(f['marge_eur'])}, "
                   f"rétro {G.eur(f['retro_eur'])}) ; sans frais, {_p(sans, d)}{par_an}.")
        erreur = ("Esquiver ou répondre « presque rien » : les frais figurent dans le document "
                  "d'informations clés, le client les retrouvera. Donner le chiffre, et dire ce "
                  "qu'il paie.")
    elif cle == "sortie":
        fin = " ou au rappel" if s.produit == AC else ""
        if inst:
            rep = (f"Rachat quotidien par l'émetteur, fourchette de l'ordre de 1 % : valeur juste "
                   f"le lendemain de l'émission {G.eur(valeur0)}, prix de rachat "
                   f"{G.eur(rachat0)}. Ensuite, le prix suit le sous-jacent, la vol, les taux et "
                   "le spread de l'émetteur : la section « Revendre en cours de vie » le chiffre.")
        else:
            rep = ("Oui : la banque rachète le produit chaque jour, mais au prix du marché, pas "
                   f"au nominal. Le lendemain de l'émission, il vaut environ {G.eur(valeur0)} "
                   f"pour {G.eur(N)}, car les frais sont payés d'avance, et la banque le rachète "
                   f"autour de {G.eur(rachat0)}. Ce produit est fait pour être gardé jusqu'à "
                   f"l'échéance{fin}"
                   + (" : la protection ne joue qu'à ce moment-là." if s.produit == CP else "."))
        chiffre = (f"Le lendemain de l'émission : valeur {G.eur(valeur0)}, rachat "
                   f"{G.eur(rachat0)} pour {G.eur(N)}.")
        erreur = ("Promettre le nominal en cas de sortie : la banque garantit la liquidité, pas "
                  "le prix" + (", et la protection n'existe qu'à l'échéance."
                               if s.produit == CP else "."))
    elif cle == "krach":
        fin = 1 - BAISSE_2008
        annee = (f"Sur l'année 2008, l'Euro Stoxx 50 a perdu {_p(BAISSE_2008, 0)}, avec un plus "
                 f"bas à {_p(PLUS_BAS_2008, 0)} de son niveau de début d'année, en novembre.")
        if s.produit == BRC:
            touchee = s.barriere >= (PLUS_BAS_2008 if s.continu else fin)
            perte = touchee and fin < s.strike
            rendu = N * (fin / s.strike if perte else 1.0) + c * T * N
            if perte:
                suite = (f" Dans ce scénario, la barrière de {_p(s.barriere, 0)} est franchie et "
                         f"vous perdez comme l'indice : environ {G.eur(rendu)} rendus pour "
                         f"{G.eur(N)}, coupon compris.")
            else:
                suite = (f" Avec une barrière à {_p(s.barriere, 0)}, ce scénario précis ne la "
                         f"franchit pas : {G.eur(rendu)} rendus, coupon compris ; une crise plus "
                         "longue, comme 2000-2003, l'aurait franchie.")
        elif s.produit == AC:
            perte = fin < s.bar_capital
            rendu = N * fin if perte else N
            if perte:
                suite = (f" Si {sj} finit l'échéance à {_p(fin, 0)} de son niveau de départ, "
                         f"comme l'indice fin 2008, sous la barrière de capital de "
                         f"{_p(s.bar_capital, 0)}, vous subissez toute la baisse : environ "
                         f"{G.eur(rendu)} rendus pour {G.eur(N)}, plus les coupons déjà touchés.")
            else:
                suite = (f" Avec une barrière de capital à {_p(s.bar_capital, 0)}, finir à "
                         f"{_p(fin, 0)} "
                         f"rend le capital ({G.eur(rendu)}) ; les coupons, eux, s'arrêtent tant "
                         f"que {sj} reste sous {_p(s.bar_coupon, 0)}.")
        else:
            rendu = N * s.protection
            suite = (f" À l'échéance, vous récupérez {G.eur(rendu)} : c'est exactement ce pour "
                     "quoi ce produit existe, à condition que la banque émettrice survive. En "
                     "2008, les porteurs de produits garantis par Lehman Brothers n'ont récupéré "
                     "qu'une partie de leur mise, des années plus tard.")
        if inst:
            rep = (f"Euro Stoxx 50 en 2008 : {_p(-BAISSE_2008)} sur l'année, plus bas à "
                   f"{_p(PLUS_BAS_2008)} du niveau initial. Rejoué sur ce produit : "
                   f"{G.eur(rendu)} pour {G.eur(N)}"
                   + ("" if s.produit == CP else
                      f" ; probabilité risque-neutre de perte en capital {_p(f['p_perte'])}")
                   + ". Le stress test se montre au client, il ne se résume pas à une "
                   "probabilité.")
        else:
            rep = annee + suite
            if s.produit != CP and len(suite) < 260:
                rep += (" Dans nos simulations, une perte arrive dans "
                        f"{_p(f['p_perte'], 0)} des cas.")
        chiffre = (f"Euro Stoxx 50 en 2008 : {_p(-BAISSE_2008, 0)} ; dans ce scénario, "
                   f"{G.eur(rendu)} rendus pour {G.eur(N)}.")
        if s.produit == CP:
            erreur = ("Oublier le risque émetteur : en 2008, c'est la faillite de Lehman "
                      "Brothers, pas la baisse des marchés, qui a fait perdre les porteurs de "
                      "produits garantis.")
        else:
            erreur = ("Promettre que la barrière tiendra : en 2008, l'indice est descendu à "
                      f"{_p(PLUS_BAS_2008, 0)} de son niveau de début d'année. Le scénario de "
                      "perte se chiffre, il ne se minimise pas.")
    elif cle == "hausse":
        if s.produit == BRC:
            if inst:
                rep = (f"Le client est vendeur d'un put, pas acheteur d'un call : rendement "
                       f"plafonné au coupon ({_p(c * T)} sur {duree}), profil concave. Pour une "
                       "vue haussière, un call spread ou un Capital Protection.")
            else:
                rep = ("Parce que ce produit ne vend pas de la hausse, il vend un revenu : vous "
                       f"touchez {val} par an quoi que fasse l'indice, soit au plus "
                       f"{G.eur(c * T * N)} sur {duree}. Même si l'indice monte de 30 %, vous "
                       "gagnez le coupon, pas plus. Si vous attendez une forte hausse, un produit "
                       "à participation ou l'indice lui-même sont plus adaptés.")
            chiffre = f"Gain maximum : {G.eur(c * T * N)} sur {duree}, soit {_p(c * T)}."
            erreur = ("Laisser croire que le client profite de la hausse : son gain est plafonné "
                      "au coupon, quoi que fasse le marché.")
        elif s.produit == AC:
            if inst:
                rep = (f"Le client est court de l'option de rappel : au-dessus de "
                       f"{_p(s.trigger, 0)}, le produit s'arrête avec un coupon de "
                       f"{_p(f['coupon_obs'], 2)}. Rappel dès la première date dans "
                       f"{_p(f['rappel_1'])} des scénarios, durée de vie attendue "
                       f"{_ans(f['duree'])} : le gain est plafonné, le risque de baisse reste "
                       "entier.")
            else:
                rep = (f"Le produit s'arrête dès que {sj} dépasse {_p(s.trigger, 0)} de son "
                       "niveau de départ à une date d'observation : vous récupérez votre capital "
                       f"et le coupon de {_p(f['coupon_obs'], 2)}, et c'est tout. En échange, vous "
                       "êtes payé même dans un marché stable. Dans nos simulations, il est "
                       f"rappelé dès la première date dans {_p(f['rappel_1'], 0)} des cas.")
            chiffre = (f"Rappel dès la première date dans {_p(f['rappel_1'], 0)} des cas ; durée "
                       f"de vie moyenne {_ans(f['duree'])} sur {duree}.")
            erreur = ("Promettre la hausse du marché : le gain s'arrête au coupon, et le rappel "
                      "coupe le produit au moment où il devenait intéressant.")
        else:
            options = f["call"] - f["call_cap"]       # ce qui reste pour la hausse
            gain = ("" if s.cap >= 2.995 else
                    f", dans la limite d'un gain de {_p(c * (s.cap - 1), 0)}")
            if c >= 1 and s.cap < 2.995:
                # participation au-dela de 100 % : c'est le cap qui la paie
                if inst:
                    rep = (f"Participation {val} financée par la vente du call au cap "
                           f"{_p(s.cap, 0)} ({G.eur(f['call_cap'])}) : le client échange les "
                           f"très fortes hausses contre un levier sur les hausses modérées. Gain "
                           f"maximum {_p(c * (s.cap - 1), 0)}.")
                else:
                    rep = (f"Vous recevez déjà {val} de la hausse, plus que l'indice lui-même, "
                           f"mais jusqu'à un plafond : au-delà de {_p(s.cap - 1, 0)} de hausse, "
                           f"votre gain s'arrête à {_p(c * (s.cap - 1), 0)}. C'est ce plafond, "
                           "revendu par la banque, qui paie la participation au-delà de 100 %.")
            elif inst:
                rep = (f"Après le zéro-coupon ({G.eur(f['zc'])}), les frais ({G.eur(frais)}) et "
                       f"le risque émetteur, il reste {G.eur(options)} d'options : le call à la "
                       f"monnaie coûte plus, d'où {val} de participation{gain}. Leviers : "
                       "protection plus basse, maturité plus longue, cap, ou moyenne des cours.")
            else:
                rep = (f"Parce que la garantie coûte : sur {G.eur(N)}, {G.eur(f['zc'])} servent à "
                       f"garantir {_p(s.protection, 0)} du capital à l'échéance, et après les "
                       f"frais, il ne reste que {G.eur(options)} pour acheter de la hausse. Cela "
                       f"paie {val} de la hausse{gain}. Une protection plus basse ou une durée "
                       "plus longue augmenterait la participation.")
            chiffre = (f"Participation {val}{gain} ; {G.eur(f['zc'])} sur {G.eur(N)} servent à "
                       "la garantie.")
            erreur = (f"Promettre « la hausse du marché » : le client n'en reçoit que {val}, et "
                      "pas les dividendes.")
    elif cle in ("concurrent", "concurrent_cp"):
        cout = f"{f['cout_retro'] * 100:,.{d}f} pt"
        de_quoi = f"de {quoi.split()[-1]}{par_an}"
        memes = ("même protection, même cap" if s.produit == CP else "mêmes barrières")
        if inst:
            rep = ("À structure identique, l'écart vient du spread de financement de l'émetteur, "
                   f"des frais et de la couverture. Un point de frais par an vaut {cout} "
                   f"{de_quoi}. Comparez la valeur juste du document d'informations "
                   "clés et le spread CDS des deux émetteurs.")
        else:
            rep = ("C'est possible, et il faut comparer à structure égale : même sous-jacent, "
                   f"même durée, {memes}. Ensuite, deux choses expliquent l'écart : les "
                   f"frais de distribution, car chaque point de rétrocession par an coûte {cout} "
                   f"{de_quoi}, et la solidité de l'émetteur, car une banque moins "
                   "bien notée paie plus parce que vous portez plus de risque.")
            if cgp:
                rep += (" Un rendement plus haut chez un émetteur plus fragile rémunère un risque, "
                        "ce n'est pas un meilleur produit.")
        chiffre = f"1 % de rétrocession par an = {cout} {de_quoi}."
        erreur = ("Dénigrer le concurrent ou baisser la marge à l'aveugle : vérifier d'abord que "
                  "la structure et la signature de l'émetteur sont les mêmes.")
    elif cle == "barriere":
        if s.produit == BRC:
            L = max(s.barriere - 0.10, 0.20)
            rendu = N * L / s.strike + c * T * N
            quand = "en cours de route" if s.continu else "à l'échéance"
            if inst:
                rep = (f"Put down-and-in : une fois la barrière de {_p(s.barriere, 0)} touchée, "
                       f"la perte est linéaire depuis le strike {_p(s.strike, 0)}, pas depuis la "
                       f"barrière. Probabilité de knock-in {_p(f['p_touche'])} ; à {_p(L, 0)} "
                       f"final, {G.eur(rendu)} rendus coupon compris.")
            else:
                rep = ("Non, et c'est le malentendu le plus fréquent : la barrière n'est pas un "
                       f"plancher. Tant que l'indice ne passe pas sous {_p(s.barriere, 0)} "
                       f"{quand}, vous êtes protégé ; si elle est franchie et que l'indice finit "
                       f"sous {_p(s.strike, 0)}, vous perdez comme lui, sans limite : à "
                       f"{_p(L, 0)}, vous récupérez environ {G.eur(rendu)}, coupon compris. La "
                       "barrière dit quand le risque commence, pas combien vous pouvez perdre.")
            chiffre = (f"Barrière franchie, indice à {_p(L, 0)} : {G.eur(rendu)} rendus pour "
                       f"{G.eur(N)} ; barrière touchée dans {_p(f['p_touche'], 0)} des cas.")
            erreur = (f"Dire « vous ne pouvez perdre que {_p(1 - s.barriere, 0)} » : la perte se "
                      "calcule depuis le strike, pas depuis la barrière.")
        else:
            juste = s.bar_capital - 0.01
            if inst:
                rep = (f"Put down-and-in européen {_p(s.bar_capital, 0)} : discontinuité à la "
                       f"barrière, le remboursement passe de {G.eur(N)} à "
                       f"{G.eur(N * s.bar_capital)} en un point. Perte linéaire depuis 100 %, "
                       f"probabilité risque-neutre {_p(f['p_perte'])}.")
            else:
                rep = ("Non, et c'est le malentendu le plus fréquent : la barrière de capital "
                       f"n'est pas un plancher. Si {sj} finit au-dessus de "
                       f"{_p(s.bar_capital, 0)}, votre capital est rendu ; juste en dessous, à "
                       f"{_p(juste, 0)}, vous récupérez {G.eur(N * juste)}, toute la baisse depuis "
                       "le départ. Un point de marché à l'échéance peut coûter "
                       f"{_p(1 - s.bar_capital, 0)} de votre capital.")
            chiffre = (f"Juste sous {_p(s.bar_capital, 0)} à l'échéance : {G.eur(N * juste)} "
                       f"rendus au lieu de {G.eur(N)}.")
            erreur = (f"Dire « vous ne perdez qu'au-delà de {_p(1 - s.bar_capital, 0)} de "
                      "baisse » : sous la barrière, la perte part du niveau de départ.")
    elif cle == "rappel":
        if inst:
            rep = (f"Le rappel au-dessus de {_p(s.trigger, 0)} coupe le coupon dans les "
                   "scénarios favorables au client : c'est ce qui le rend plus élevé. Durée de "
                   f"vie attendue {_ans(f['duree'])} sur {duree}, rappel dès la première date "
                   f"dans {_p(f['rappel_1'])} des scénarios ; le client porte le risque de "
                   "réinvestissement.")
        else:
            rep = (f"Parce que c'est ce qui finance le coupon : dès que {sj} est au-dessus de "
                   f"{_p(s.trigger, 0)} à une date d'observation, la banque vous rend votre "
                   "capital plus le coupon, et ne vous doit plus rien. Dans nos simulations, cela "
                   f"arrive dès la première date dans {_p(f['rappel_1'], 0)} des cas, et le "
                   f"produit dure en moyenne {_ans(f['duree'])} sur {duree} possibles. La "
                   "contrepartie : il faudra réinvestir, peut-être à des conditions moins bonnes.")
        chiffre = (f"Rappel à la première date dans {_p(f['rappel_1'], 0)} des cas ; durée de vie "
                   f"moyenne {_ans(f['duree'])} sur {duree}.")
        erreur = ("Présenter la durée maximale comme la durée probable : l'argent revient "
                  "probablement bien plus tôt, et il faudra le réinvestir.")
    else:
        raise ValueError(cle)
    return {"reponse": rep, "chiffre": chiffre, "erreur": erreur}


def _cache_reponse():
    st.session_state["as_obj_voir"] = False


def _hasard(cle, options):
    actuelle = st.session_state.get(cle)
    st.session_state[cle] = random.choice([o for o in options if o != actuelle] or options)
    st.session_state["as_obj_voir"] = False


def _choix_profil(col):
    """Le profil du client, commun au pitch et aux objections."""
    return col.segmented_control(
        "Profil du client", PROFILS, default=PARTICULIER, required=True, key="as_profil",
        bind="query-params",
        help="Le même produit ne se présente pas de la même façon. Particulier : des euros et "
             "des mots simples. Conseiller en gestion de patrimoine : le mécanisme, la "
             "rétrocession, l'adéquation. Institutionnel : les options, les probabilités "
             "risque-neutres, la valeur juste.")


def _section_objections(mk, s):
    suffixe = {BRC: "brc", AC: "ac", CP: "cp"}[s.produit]
    cle_choix = f"as_obj_{suffixe}"
    options = [QUESTIONS[k] for k in LISTE_OBJECTIONS[s.produit]]
    profil = _choix_profil(st)
    c = st.columns([3, 1], vertical_alignment="bottom")
    question = c[0].selectbox(
        "Objection du client", options, key=f"as_obj_{suffixe}", bind="query-params",
        on_change=_cache_reponse,
        help="Les objections qu'un Sales entend chaque semaine. La réponse reste cachée le temps "
             "de préparer la vôtre.")
    c[1].button("Objection au hasard", key="as_obj_hasard", on_click=_hasard,
                args=(cle_choix, options), width="stretch",
                help="Tire une autre objection et cache la réponse.")
    cle = next(k for k in LISTE_OBJECTIONS[s.produit] if QUESTIONS[k] == question)

    with st.container(border=True):
        st.markdown(f"**Le client :** « {question} »")
    voir = st.toggle("Voir la réponse du Sales", key="as_obj_voir", bind="query-params",
                     help="Préparez d'abord votre réponse : 2 à 4 phrases, honnêtes, avec un "
                          "chiffre. Puis comparez.")
    if not voir:
        st.caption("Réfléchissez à votre réponse, puis affichez celle du Sales.")
        return
    o = objection(s, profil, cle, faits(mk, s))
    with st.container(border=True):
        st.markdown("**À dire au client.** " + o["reponse"])
    c = st.columns(2)
    c[0].info("**Le chiffre à citer.** " + o["chiffre"])
    c[1].warning("**L'erreur à éviter.** " + o["erreur"])
    st.caption("Réponses construites sur des modèles, avec les chiffres des autres sections : "
               "coupon, frais, probabilité de perte, risque émetteur.")


# =====================================================================
# PAGE
# =====================================================================

def affiche(m):
    st.title("Atelier Sales", help="**Ce que c'est.** Les cinq gestes d'un Sales face à un "
             "client, sur un même produit : trouver la structure qui répond à sa demande, "
             "expliquer où va son argent, lui racheter le produit en cours de vie, le lui "
             "présenter, et répondre à ses objections.\n\n**Comment c'est construit.** Tout part de la vue de marché de la "
             "barre latérale et du cadre « Structure », plus la rétrocession du distributeur. "
             "Les calculs sont ceux des pages produits (mêmes trajectoires, même graine) : avec "
             "une rétro nulle, les coupons sont exactement les mêmes.\n\n**Pourquoi c'est le "
             "métier.** Un client n'achète pas un payoff, il pose une contrainte : « je veux 8 % "
             "», « je ne veux rien perdre ». Le Sales la traduit en structure, rapporte une "
             "réponse honnête quand c'est hors d'atteinte, et sait dire où part chaque euro, "
             "rétro comprise.\n\n**Ce qu'il faut regarder.** Dans « Le client demande », "
             "l'arbitrage sur le graphe ; puis « Retenir cette structure » et suivre le même "
             "produit dans les autres sections. Le pitch nomme toujours le scénario de "
             "perte et le risque émetteur.")
    st.caption("Le client pose une contrainte. Le Sales rapporte une structure, son prix et "
               "son risque.")
    m.bandeau_modele()

    produit = st.segmented_control(
        "Produit", PRODUITS, default=BRC, required=True, key="as_produit", bind="query-params",
        help="Les trois produits que sales.py sait structurer, décomposer et racheter.")
    section = st.segmented_control(
        "Section", SECTIONS, default=DEMANDE, required=True, key="as_section",
        bind="query-params",
        help="Seule la section choisie est calculée. Le cadre Structure est commun à toutes.")

    cherche = cible = None
    if section == DEMANDE:
        cherche, cible = _saisie_demande(produit)
    s, retro = _structure(produit, cherche)
    mk = S.Marche.depuis_barre(m, retro=retro)
    if produit == AC and s.n_actifs > 1 and mk.bates:
        st.caption("Worst-of simulé en Black-Scholes multi-actifs, comme la page Autocall.")

    if section == DEMANDE:
        st.subheader("La réponse du desk",
                     help="Le desk balaie le paramètre du plus favorable au client au moins "
                          "favorable, sur les mêmes tirages, et s'arrête au premier qui respecte "
                          "la demande (affiné par dichotomie). Rien n'est supposé monotone.")
        _section_demande(mk, s, cherche, cible)
    elif section == ARGENT:
        st.subheader("Où va l'argent",
                     help="Décomposition du nominal en obligation, options, risque émetteur, "
                          "marge et rétrocession. La somme retombe exactement sur le nominal. "
                          "Le graphe montre le budget : ce qui le finance, ce qui le consomme.")
        _section_argent(mk, s)
    elif section == REVENTE:
        st.subheader("Revendre en cours de vie",
                     help="Valeur des flux restants à une date donnée, avec l'historique "
                          "(barrière touchée, observations passées) et le marché du jour. La "
                          "décomposition change un facteur à la fois, sur les mêmes tirages, "
                          "et retombe exactement sur la valeur du jour.")
        _section_revente(mk, s)
    elif section == OBJECTIONS:
        st.subheader("Objections clients",
                     help="Le client fictif pose une objection classique ; la réponse d'un bon "
                          "Sales arrive quand vous l'avez préparée. Chiffrée avec le produit à "
                          "l'écran : coupon, frais sur la durée de vie, probabilité de perte, "
                          "spread émetteur. Honnête : on ne cache jamais le risque.")
        _section_objections(mk, s)
    else:
        st.subheader("Pitch client",
                     help="Trois à cinq phrases construites sur des modèles à partir du produit, "
                          "de ses chiffres, du profil du client et de sa vue de marché. Pas d'IA : "
                          "les mêmes chiffres donnent toujours le même pitch.")
        _section_pitch(mk, s)
