"""
Atelier Sales : les outils du Sales face au client (sales.py), plus un pitch.

Quatre sections, un seul produit a la fois (BRC, Autocall, Capital
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

Seule la section choisie est calculee, et chaque appel a sales.py est en
cache (l'objet Marche est hachable) : les curseurs restent rapides.
"""

import json
import math
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
DEMANDE, ARGENT, REVENTE, PITCH = ("Le client demande", "Où va l'argent",
                                   "Revendre en cours de vie", "Pitch client")
SECTIONS = [DEMANDE, ARGENT, REVENTE, PITCH]
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
         "credit": "Risque émetteur", "call": "Call achète", "cap": "Call au cap vendu",
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
            "Coupon demande (% par an)", 0.5, 40.0, 8.0 if produit == BRC else 5.0, 0.25,
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
        quoi = {"barriere": f"avec une barrière a {_p(x, 0)}", "strike": f"avec un strike à {_p(x, 0)}",
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
                f"{_p(x, 0)} du niveau de départ, si la barrière a {_p(s.barriere, 0)} a été "
                "touchée, et elle est plus faible que celle de l'indice.")
    if cherche == "maturite":
        return (f"{debut}{_p(obtenu, 2)} par an est faisable des {T}, avec une barrière a "
                f"{_p(s.barriere, 0)}.")
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
    La valeur ronde que le Sales propose (au point pres), dans le sens qui
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
                     "dixième pres. Hors d'atteinte : la meilleure valeur possible.")
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
              "credit": "du risque émetteur porte par le client", "cap": "du call au cap vendu"}
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

    k = st.columns(4)
    _metriques_resultat(s, cascade, k[0], "Participation offerte" if s.produit == CP
                        else "Coupon offert")
    k[1].metric("Marge banque", G.eur(mk.marge * T * N),
                help=f"{mk.marge * 100:,.1f}% par an sur {_duree(T)}, prélevée d'avance.")
    k[2].metric("Rétrocession", G.eur(mk.retro * T * N),
                help=f"{mk.retro * 100:,.1f}% par an sur {_duree(T)}, payée d'avance au "
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
                help=f"{abs(rb - ra):,.1f} pt de rétro par an sur {_duree(T)} = "
                     f"{G.eur(abs(rb - ra) / 100 * T * N)} pris sur le budget option.")

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
                           [G.eur(v / 100 * T * N) for v in xr])],
        legende=False, hauteur=320)
    G.affiche(graphe)

    comp = [{"Composante": la.libelle, "Réseau A": G.eur(la.montant),
             "Réseau B": G.eur(lb.montant), "Écart": G.eur(lb.montant - la.montant, signe=True)}
            for la, lb in zip(ca.lignes, cb.lignes)]
    st.table(comp, hide_index=True)
    st.info(f"**Pourquoi deux {quoi}s différents.** La rétro est payée d'avance sur toute la "
            f"durée : {abs(rb - ra):,.1f} pt par an sur {_duree(T)}, c'est "
            f"{G.eur(abs(rb - ra) / 100 * T * N)} de moins dans le budget option pour le "
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
                                  help="Une fois touchée, la protection à disparu pour de bon : "
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
        "tirages. Axe coupe.", nominal=N,
        legende=[("Fait monter la valeur", G.GAIN), ("Fait baisser la valeur", G.PERTE),
                 ("Valeur du produit", G.PRODUIT)])
    G.affiche(graphe)
    st.caption(v.message)

    frais = (mk.marge + mk.retro) * T * N
    if s.produit == CP:
        st.info(f"**La protection ne joue qu'à l'échéance.** Avant, le produit vaut une "
                f"obligation ({G.eur(v.infos['obligation'])} aujourd'hui) plus des options : "
                "une hausse des taux ou du spread le fait passer sous le niveau garanti. Le "
                "client « garanti » qui revend avant l'échéance peut perdre.")
    else:
        st.info(f"**Revendre le lendemain de l'émission coûte déjà {G.eur(frais + fourchette / 200 * N)}** : "
                f"marge ({G.eur(mk.marge * T * N)}) et rétro ({G.eur(mk.retro * T * N)}) "
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
         "recovery": mk.recovery, "r": mk.r, "credit": abs(lignes.get("credit", 0.0)),
         "taux": mk.nominal - lignes["zc"], "cout_retro": cascade.infos["cout_retro_par_pt"],
         "resultat": _resultat_pa(s, cascade)}
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
                              f"qu'un placement sans risque à {_p(f['r'], 1)} aurait rapporte "
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
            txt = (f"Exposition au crédit de l'émetteur, senior non sécurisé, recouvrement "
                   f"suppose {rec} ; le spread est à 0 bp dans cette vue de marché, chaque "
                   f"point de spread se retrouverait dans {finance}.")
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
            phrases.append(("retro", f"Aucune rétrocession n'est prévue : {finance} affiche "
                                     "est brut de frais de distribution."))
    elif profil == INSTIT:
        frais = (f["marge"] + f["retro"]) * T * N
        phrases.append(("liquidite", f"Valeur juste à l'émission {G.eur(N - frais)} pour "
                                     f"{G.eur(N)} ({G.eur(frais)} de marge et rétro payées "
                                     "d'avance) ; rachat quotidien par l'émetteur, fourchette "
                                     "de l'ordre de 1 %."))
    return {"phrases": phrases, "adequation": niveau_adequation}


def _section_pitch(mk, s):
    c = st.columns([3, 2])
    profil = c[0].segmented_control(
        "Profil du client", PROFILS, default=PARTICULIER, required=True, key="as_profil",
        bind="query-params",
        help="Le même produit ne se présente pas de la même façon. Particulier : des euros et "
             "des mots simples. Conseiller en gestion de patrimoine : le mécanisme, la "
             "rétrocession, l'adéquation. Institutionnel : les options, les probabilités "
             "risque-neutres, la valeur juste.")
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
               "des autres sections. Toujours présents : le scénario de perte chiffre et le "
               "risque émetteur.")
    with st.expander("Texte à copier"):
        st.code(texte, language=None, wrap_lines=True)


# =====================================================================
# PAGE
# =====================================================================

def affiche(m):
    st.title("Atelier Sales", help="**Ce que c'est.** Les quatre gestes d'un Sales face à un "
             "client, sur un même produit : trouver la structure qui répond à sa demande, "
             "expliquer où va son argent, lui racheter le produit en cours de vie, et le lui "
             "présenter.\n\n**Comment c'est construit.** Tout part de la vue de marché de la "
             "barre latérale et du cadre « Structure », plus la rétrocession du distributeur. "
             "Les calculs sont ceux des pages produits (mêmes trajectoires, même graine) : avec "
             "une rétro nulle, les coupons sont exactement les mêmes.\n\n**Pourquoi c'est le "
             "métier.** Un client n'achète pas un payoff, il pose une contrainte : « je veux 8 % "
             "», « je ne veux rien perdre ». Le Sales la traduit en structure, rapporte une "
             "réponse honnête quand c'est hors d'atteinte, et sait dire où part chaque euro, "
             "rétro comprise.\n\n**Ce qu'il faut regarder.** Dans « Le client demande », "
             "l'arbitrage sur le graphe ; puis « Retenir cette structure » et suivre le même "
             "produit dans les trois autres sections. Le pitch nomme toujours le scénario de "
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
        help="Seule la section choisie est calculée. Le cadre Structure est commun aux quatre.")

    cherche = cible = None
    if section == DEMANDE:
        cherche, cible = _saisie_demande(produit)
    s, retro = _structure(produit, cherche)
    mk = S.Marche.depuis_barre(m, retro=retro)
    if produit == AC and s.n_actifs > 1 and mk.bates:
        st.caption("Worst-of simule en Black-Scholes multi-actifs, comme la page Autocall.")

    if section == DEMANDE:
        st.subheader("La réponse du desk",
                     help="Le desk balaie le paramètre du plus favorable au client au moins "
                          "favorable, sur les mêmes tirages, et s'arrête au premier qui respecte "
                          "la demande (affine par dichotomie). Rien n'est suppose monotone.")
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
    else:
        st.subheader("Pitch client",
                     help="Trois à cinq phrases construites sur des modèles à partir du produit, "
                          "de ses chiffres, du profil du client et de sa vue de marché. Pas d'IA : "
                          "les mêmes chiffres donnent toujours le même pitch.")
        _section_pitch(mk, s)
