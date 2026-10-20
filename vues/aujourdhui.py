"""
Que vendre aujourd'hui ? Le marche du jour (data/marche_du_jour.json), ce
que chaque produit peut offrir avec lui, compare a un marche habituel.

Les conditions sont calculees comme dans les pages produits : memes
fonctions du moteur, memes reglages par defaut, meme graine, et le modele,
la marge, le recouvrement, le nominal et le nombre de trajectoires de la
barre laterale. Apres le scenario « Aujourd'hui », chaque page produit
affiche donc le meme chiffre que cette page (controle par
tests/verifie_aujourdhui.py). Si une page produit change sa facon d'appeler
le moteur, adapter conditions().
"""

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import donnees_du_jour as DJ
import graphiques as G
import moteur as M
import sales as S
from barre import AUJOURDHUI, scenario_du_jour
from communs import MAX_CACHE
from moteur import S0
from vues.autocall import CLASSIQUE

CONTINUE = "Continue (américaine)"
CP, BRC, AC, BONUS, TW, DC = "cp", "brc", "ac", "bonus", "tw", "dc"
ORDRE = [CP, BRC, AC, BONUS, TW, DC]

NOMS = {CP: "Capital Protection", BRC: "Barrier Reverse Convertible",
        AC: "Autocall Phoenix", BONUS: "Bonus Certificate", TW: "Twin-Win",
        DC: "Discount Certificate"}
PAGES_PRODUITS = {CP: "2 - Capital Protection", BRC: "3 - Barrier Reverse Convertible",
                  AC: "4 - Autocall Phoenix", BONUS: "5 - Bonus Certificate",
                  TW: "6 - Twin-Win", DC: "7 - Discount Certificate"}
REGLAGES = {CP: "5 ans, capital protégé à 100 %, sans cap",
            BRC: "1 an, barrière 70 % continue, strike 100 %",
            AC: "3 ans, trimestriel, trigger 100 %, coupon 70 %, capital 60 %, mémoire",
            BONUS: "2 ans, barrière 70 % continue",
            TW: "2 ans, barrière 65 % continue",
            DC: "1 an, cap 100 %"}
GRANDEURS = {CP: "Participation à la hausse", BRC: "Coupon par an",
             AC: "Coupon annuel équivalent", BONUS: "Gain minimum si la barrière tient",
             TW: "Participation à la baisse", DC: "Décote"}
# Les curseurs des pages produits, a leurs valeurs par defaut
DEFAUTS_PAGES = {
    CP: {"cp_maturite": 5.0, "cp_protection": 100, "cp_cap": 300},
    BRC: {"brc_maturite": 1.0, "brc_barriere": 70, "brc_strike": 100,
          "brc_observation": CONTINUE},
    AC: {"ac_maturite": 3.0, "ac_frequence": "Trimestrielle", "ac_trigger": 100,
         "ac_bar_coupon": 70, "ac_bar_capital": 60, "ac_memoire": True,
         "ac_step_down": False, "ac_indice": CLASSIQUE, "ac_sous_jacents": 1},
    BONUS: {"bonus_maturite": 2.0, "bonus_barriere": 70},
    TW: {"tw_maturite": 2.0, "tw_barriere": 65},
    DC: {"dc_maturite": 1.0, "dc_cap": 100},
}
ATTRACTIF, NORME, MOINS = "Plus attractif que d'habitude", "Dans la norme", \
    "Moins attractif que d'habitude"
SEUIL = 0.10           # +/- 10 % du niveau habituel


# =====================================================================
# CALCULS (memes appels que les pages produits)
# =====================================================================

def marche(m, valeurs):
    """Vue de marche des calculs : curseurs du scenario + reglages de la barre laterale."""
    d = S.Marche()

    def ou(x, defaut):
        return defaut if x is None else x

    return S.Marche(modele="bates" if m.BATES else "bs", r=valeurs["taux"] / 100,
                    q=valeurs["dividende"] / 100, vol_atm=valeurs["vol"] / 100,
                    skew=valeurs["skew"], theta_v=valeurs["vol_lt"] / 100,
                    kappa=ou(m.kappa, d.kappa), xi=ou(m.xi, d.xi), rho=ou(m.rho, d.rho),
                    lam=ou(m.lam, d.lam), mu_j=ou(m.mu_j, d.mu_j), sig_j=ou(m.sig_j, d.sig_j),
                    spread=valeurs["spread"], recovery=m.recovery, marge=m.marge, retro=0.0,
                    nominal=m.nominal, n_paths=m.n_paths)


def _bissection(pv, lo, hi, cible):
    """Meme recherche que les pages Bonus et Twin-Win (40 iterations)."""
    for _ in range(40):
        mid = (lo + hi) / 2
        hi, lo = (mid, lo) if pv(mid) > cible else (hi, mid)
    return lo


def conditions(mk):
    """
    Ce que chaque produit offre sur ce marche, aux reglages par defaut de sa
    page. valeur = la grandeur comparee (fraction), plus les risques a citer.
    """
    out, r, nominal, marge = {}, mk.r, mk.nominal, mk.marge

    out[CP] = {"valeur": S.participation_cp(mk, 5.0, 1.0, 3.0)}

    # BRC 1 an et Discount 1 an : memes trajectoires
    ch, sg = mk.trajectoires(1.0, 0.70)
    tau = mk.defauts(len(ch), 1.0)
    cpn = M.coupon_brc(ch, 1.0, r, nominal, 0.70, 1.0, True, marge, sg, tau, mk.recovery)
    ST = ch[:, -1].astype(np.float64)
    p_no = M.proba_non_touche(ch, 0.70 * S0, sg, 1.0)
    out[BRC] = {"valeur": float(cpn), "p_perte": float(((1 - p_no) * (ST < S0)).mean()),
                "p_touche": float(1 - p_no.mean())}
    if mk.bates:
        dc = M.discount_mc(ch, 1.0, r, 1.0, marge)
    else:
        dc = M.discount_certificate(S0, S0, 1.0, r, M.vol_au_strike(1.0, mk.vol_atm, mk.skew),
                                    mk.q, marge)
    out[DC] = {"valeur": float(dc["decote"]), "p_perte": float((ST < dc["prix"]).mean()),
               "rendement_max": float(dc["rendement_max"])}
    del ch, sg

    # Bonus et Twin-Win 2 ans : memes trajectoires, barrieres 70 % et 65 %
    ch, sg = mk.trajectoires(2.0, 0.70)
    ST = ch[:, -1].astype(np.float64)
    perf = ST / S0 - 1
    cible = nominal * (1.0 - marge * 2.0)
    actu = math.exp(-r * 2.0)

    p70 = M.proba_non_touche(ch, 0.70 * S0, sg, 2.0)
    bonus = _bissection(lambda b: actu * (p70 * np.maximum(nominal * ST / S0, nominal * b)
                                          + (1 - p70) * nominal * ST / S0).mean(),
                        1.0, 2.5, cible)
    out[BONUS] = {"valeur": bonus - 1.0, "p_touche": float(1 - p70.mean())}

    p65 = M.proba_non_touche(ch, 0.65 * S0, sg, 2.0)

    def pv_twin(pb):
        haut = np.where(perf >= 0, nominal * (1 + perf), nominal * (1 + pb * (-perf)))
        return actu * (p65 * haut + (1 - p65) * nominal * ST / S0).mean()

    out[TW] = {"valeur": _bissection(pv_twin, 0.0, 3.0, cible), "p_touche": float(1 - p65.mean())}
    del ch, sg

    # Autocall Phoenix 3 ans trimestriel 100 / 70 / 60, memoire
    ch, _ = mk.trajectoires(3.0, 0.60)
    tau = mk.defauts(len(ch), 3.0)
    cpn = M.coupon_autocall(ch, 3.0, r, nominal, 12, 1.0, 0.70, 0.60, True, marge, tau,
                            mk.recovery)
    duree, _, p_perte = M.stats_autocall(ch, 3.0, 12, 1.0, 0.60)
    out[AC] = {"valeur": float(cpn) * 4, "duree": float(duree), "p_perte": float(p_perte)}
    return out


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _conditions(mk):
    return conditions(mk)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def analyse(mk_jour, mk_habituel):
    """
    Conditions du jour, conditions habituelles, et l'effet de chaque parametre
    pris seul : on part du marche habituel et on ne change que lui.
    """
    jour, hab = _conditions(mk_jour), _conditions(mk_habituel)
    seuls = {"taux": mk_habituel.avec(r=mk_jour.r),
             "vol": mk_habituel.avec(vol_atm=mk_jour.vol_atm, theta_v=mk_jour.theta_v),
             "dividende": mk_habituel.avec(q=mk_jour.q)}
    effets = {}
    for nom, mk in seuls.items():
        c = conditions(mk) if mk != mk_habituel else hab
        effets[nom] = {p: c[p]["valeur"] - hab[p]["valeur"] for p in ORDRE}
    return {"jour": jour, "habituel": hab, "effets": effets}


def verdict(jour, habituel):
    """Rapport au niveau habituel et verdict : +/- 10 % = dans la norme."""
    rapport = jour / habituel if habituel > 1e-9 else (math.inf if jour > 0 else 1.0)
    if rapport >= 1 + SEUIL:
        return rapport, ATTRACTIF
    if rapport <= 1 - SEUIL:
        return rapport, MOINS
    return rapport, NORME


# =====================================================================
# FORMATS (ceux des pages produits)
# =====================================================================

def format_valeur(p, v):
    if p in (CP, TW):
        return f"{v*100:,.0f}%"
    if p in (BRC, AC):
        return f"{v*100:,.2f}% p.a."
    if p == BONUS:
        return f"{v*100:,.1f}%"
    return f"{v*100:,.2f}%"


def format_ecart(p, e):
    """Ecart en points de % (participation sans decimale)."""
    return f"{e*100:+,.0f} pts" if p in (CP, TW) else f"{e*100:+,.2f} pts"


def _rang(rg):
    return "" if rg is None else f", plus haut que {rg*100:.0f} % des séances depuis 2004"


def _niveau(rg):
    """Position dans l'historique : 'bas' sous le 30e centile, 'élevé' au-dessus du 70e."""
    if rg is None:
        return "inconnu"
    return "élevé" if rg >= 0.70 else ("bas" if rg <= 0.30 else "dans la norme")


# =====================================================================
# ACTIONS : marche du jour dans la barre laterale, ouverture d'une page
# =====================================================================

def _charge_jour():
    """Avant le rendu des curseurs : la barre laterale prend le scenario « Aujourd'hui »."""
    sc = scenario_du_jour()
    for c, v in sc["valeurs"].items():
        st.session_state[c] = v
    st.session_state["scenario_applique"] = AUJOURDHUI


def _ouvre(p):
    """Marche du jour + reglages par defaut de la page produit, puis la page."""
    _charge_jour()
    for c, v in DEFAUTS_PAGES[p].items():
        st.session_state[c] = v
    st.session_state["page"] = PAGES_PRODUITS[p]


# =====================================================================
# TEXTES
# =====================================================================

def raisons_marche(d, jour, hab):
    """Les trois faits de marche, en clair, avec leur conséquence pour la vente."""
    vj, vh = jour["exact"], hab["exact"]
    rg_taux = DJ.habituel(d, "taux_3_ans")[1] if jour["origine"]["taux"] else None
    rg_vol = DJ.habituel(d, hab["origine"]["vol"])[1] if jour["origine"]["vol"] else None
    rg_div = DJ.habituel(d, "dividende")[1] if jour["origine"]["dividende"] else None
    lignes = []

    n = _niveau(rg_taux)
    suite = {"élevé": "la garantie du capital coûte moins cher, il reste plus de budget pour "
                      "les options : le capital garanti redevient vendable, et tous les coupons "
                      "en profitent un peu.",
             "bas": "garantir le capital coûte cher, il reste peu de budget : le capital "
                    "garanti offre peu, le marché bascule vers les BRC et les autocalls.",
             "dans la norme": "le budget des produits garantis est dans sa moyenne.",
             "inconnu": "le taux du jour n'est pas disponible, la page prend la valeur par "
                        "défaut."}[n]
    lignes.append(f"**Taux : {vj['taux']:.2f} %** sur 3 ans, contre {vh['taux']:.2f} % "
                  f"habituellement{_rang(rg_taux)}. Donc {suite}")

    n = _niveau(rg_vol)
    suite = {"élevé": "les options vendues par le client valent cher : les BRC, les "
                      "Discount et les autocalls paient bien.",
             "bas": "les options vendues par le client rapportent peu : BRC, Discount et "
                    "autocalls paient moins, les produits qui achètent de l'option (capital "
                    "garanti, Twin-Win) coûtent moins cher.",
             "dans la norme": "ni chère ni bon marché : les produits qui vendent de la vol "
                              "(BRC, Discount, autocall) paient leur niveau habituel, ni plus "
                              "ni moins.",
             "inconnu": "la vol du jour n'est pas disponible, la page prend la valeur par "
                        "défaut."}[n]
    lignes.append(f"**Vol : {vj['vol']:.1f} %**, contre {vh['vol']:.1f} % "
                  f"habituellement{_rang(rg_vol)}. Donc {suite}")

    n = _niveau(rg_div)
    suite = {"élevé": "le client abandonne plus de dividendes à la banque : Discount, Bonus "
                      "et BRC en profitent.",
             "bas": "le client abandonne moins de dividendes à la banque : un peu moins de "
                    "budget pour le Discount, le Bonus et le BRC.",
             "dans la norme": "effet neutre.",
             "inconnu": "le dividende du jour n'est pas disponible, la page prend la valeur "
                        "par défaut."}[n]
    lignes.append(f"**Dividende : {vj['dividende']:.2f} %** net sur 12 mois, contre "
                  f"{vh['dividende']:.2f} % habituellement{_rang(rg_div)}. Donc {suite}")
    return lignes


GRACE = {"taux": "grâce aux taux", "vol": "grâce à la vol", "dividende": "grâce au dividende"}
CAUSE = {"taux": "à cause des taux", "vol": "à cause de la vol",
         "dividende": "à cause du dividende"}
PARAMETRES = {"taux": "Taux", "vol": "Vol", "dividende": "Dividende"}


def effets_seuls(p, effets):
    """'taux +48 pts, vol -1 pts, dividende -3 pts' : chaque parametre pris seul."""
    return ", ".join(f"{PARAMETRES[k]} {format_ecart(p, v[p])}" for k, v in effets.items())


def pourquoi(p, effets, ecart):
    """
    Le parametre qui explique l'ecart au niveau habituel, s'il y en a un :
    le plus gros effet pris seul, s'il va dans le sens de l'ecart et que les
    effets contraires n'en effacent pas plus de la moitie. Sinon, les effets
    se compensent.
    """
    e = {k: v[p] for k, v in effets.items()}
    cle = max(e, key=lambda k: abs(e[k]))
    if abs(e[cle]) * 100 < 0.05 and abs(ecart) * 100 < 0.05:
        return "Marché proche de l'habituel"
    contraires = sum(abs(x) for x in e.values() if x * ecart < 0)
    if e[cle] * ecart > 0 and contraires <= 0.5 * abs(e[cle]):
        return f"Surtout {(GRACE if e[cle] > 0 else CAUSE)[cle]}"
    return "Des effets qui se compensent"


def phrase_offre(p, j, h):
    """La proposition, chiffree, comparee au niveau habituel."""
    v, vh = format_valeur(p, j[p]["valeur"]), format_valeur(p, h[p]["valeur"])
    return {
        CP: f"Je mettrais en avant le capital garanti : sur 5 ans, 100 % du capital rendu à "
            f"l'échéance et {v} de la hausse de l'Euro Stoxx 50, contre {vh} dans un marché "
            "habituel.",
        BRC: f"Je mettrais en avant un BRC 1 an barrière 70 % : un coupon de {v} payé quoi "
             f"qu'il arrive, contre {vh} dans un marché habituel.",
        AC: f"Je mettrais en avant un autocall Phoenix 3 ans sur l'Euro Stoxx 50 : {v}, contre "
            f"{vh} dans un marché habituel, avec une durée de vie attendue de "
            f"{j[AC]['duree']:.1f} an.",
        BONUS: f"Je mettrais en avant un Bonus Certificate 2 ans : au moins {v} de gain si "
               f"l'indice ne touche jamais 70 %, contre {vh} d'habitude, sans plafond à la "
               "hausse.",
        TW: f"Je mettrais en avant un Twin-Win 2 ans : le client gagne {v} de chaque point de "
            f"baisse tant que la barrière à 65 % tient, contre {vh} d'habitude.",
        DC: f"Je mettrais en avant un Discount Certificate 1 an : l'Euro Stoxx 50 avec {v} de "
            f"remise, contre {vh} dans un marché habituel.",
    }[p]


def risque(p, j):
    """Le risque a dire au client, chiffre (debut de phrase en minuscule)."""
    x = j[p]
    return {
        CP: "la garantie ne vaut qu'à l'échéance et tant que l'émetteur ne "
            "fait pas défaut, et le client ne touche rien si l'indice baisse.",
        BRC: f"si l'indice passe un jour sous 70 % et finit sous son niveau "
             f"de départ, le client perd comme lui, ce qui arrive dans "
             f"{x.get('p_perte', 0)*100:.0f} % des scénarios simulés.",
        AC: f"si l'indice finit sous 60 % au bout de 3 ans, le client perd "
            f"comme lui, dans {x.get('p_perte', 0)*100:.0f} % des scénarios simulés, et le "
            "produit est souvent rappelé tôt, ce qui coupe les coupons suivants.",
        BONUS: f"si l'indice touche 70 % une seule fois, le bonus disparaît "
               f"et le client suit toute la baisse ; la barrière casse dans "
               f"{x.get('p_touche', 0)*100:.0f} % des scénarios simulés.",
        TW: f"si l'indice touche 65 % une seule fois, le gain à la baisse "
            f"disparaît et le client suit toute la baisse ; la barrière casse dans "
            f"{x.get('p_touche', 0)*100:.0f} % des scénarios simulés.",
        DC: f"le gain est plafonné à {x.get('rendement_max', 0)*100:.1f} %, "
            f"et sous le point mort le client perd comme un actionnaire, dans "
            f"{x.get('p_perte', 0)*100:.0f} % des scénarios simulés.",
    }[p]


REVENU = {BRC: "le BRC 1 an", AC: "l'autocall 3 ans"}


def reponse_entretien(jour, hab, res, classement):
    """« Que vendrais-tu aujourd'hui ? » : 4 phrases, chiffrees, honnetes sur le risque."""
    vj, vh = jour["exact"], hab["exact"]
    j, h = res["jour"], res["habituel"]
    meilleur = classement[0]
    debut = (f"Aujourd'hui, les taux à 3 ans sont à {vj['taux']:.1f} % contre "
             f"{vh['taux']:.1f} % en médiane depuis 2004, et la vol de l'Euro Stoxx 50 est à "
             f"{vj['vol']:.0f} % contre {vh['vol']:.0f} % d'habitude.")
    if res["rapports"][meilleur] < 1 + SEUIL:
        debut += " Aucun produit n'est particulièrement bon marché : je partirais de la vue du "\
                 "client plutôt que du marché."
    # deuxieme idee : le capital garanti si le meilleur est un produit de revenu, sinon le
    # produit de revenu (BRC ou autocall) le mieux place
    autre = CP if meilleur in (BRC, AC) else next(p for p in classement if p in (BRC, AC))
    if autre == CP:
        fin = (f"Pour un client prudent, le capital garanti 5 ans offre "
               f"{format_valeur(CP, j[CP]['valeur'])} de la hausse, sans risque sur le capital "
               "à l'échéance, hors défaut de l'émetteur.")
    else:
        fin = (f"Pour un client qui cherche du revenu, l'autre idée est {REVENU[autre]} à "
               f"{format_valeur(autre, j[autre]['valeur'])}, en lui disant clairement qu'il "
               f"perd son capital dans {j[autre]['p_perte']*100:.0f} % des scénarios simulés.")
    return [debut, phrase_offre(meilleur, j, h), "Le risque à dire : " + risque(meilleur, j), fin]


# =====================================================================
# GRAPHE
# =====================================================================

def graphe_attrait(lignes, titre, sous_titre):
    """Barres horizontales : niveau du jour en % du niveau habituel (100 = habituel)."""
    df = pd.DataFrame(lignes)
    haut = max(160.0, float(df["indice"].max()) * 1.15)
    y = alt.Y("produit:N", sort=alt.SortField("ordre"), title=None,
              axis=alt.Axis(labelLimit=220))
    x = alt.X("indice:Q", title="Niveau du jour, en % du niveau habituel",
              scale=alt.Scale(domain=[0, haut]))
    infobulle = [alt.Tooltip("produit:N", title="Produit"),
                 alt.Tooltip("jour:N", title="Aujourd'hui"),
                 alt.Tooltip("habituel:N", title="Habituel"),
                 alt.Tooltip("verdict:N", title="Verdict")]
    base = alt.Chart(df)
    barres = base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3,
                           height=22).encode(x, y, color=alt.Color("couleur:N", scale=None),
                                             tooltip=infobulle)
    texte = base.mark_text(align="left", dx=4, fontSize=11, fontWeight="bold").encode(
        x, y, text="etiquette:N", color=alt.Color("couleur:N", scale=None), tooltip=infobulle)
    repere = alt.Chart(pd.DataFrame({"x": [100.0], "t": ["niveau habituel"]}))
    trait = repere.mark_rule(strokeDash=[4, 4], strokeWidth=1.4, color=G.NIVEAU).encode(x="x:Q")
    note = repere.mark_text(align="left", baseline="top", dx=4, dy=-14, fontSize=11,
                            color=G.NIVEAU).encode(x="x:Q", y=alt.value(0), text="t:N")
    return alt.layer(barres, texte, trait, note).properties(
        title=G._titre(titre, sous_titre), height=44 * len(df) + 20)


# =====================================================================
# PAGE
# =====================================================================

def affiche(m):
    st.title("Que vendre aujourd'hui ?", help="**Ce que c'est.** Le marché du jour (taux, vol, "
             "dividende), et ce que chaque produit peut offrir avec lui, comparé à un marché "
             "habituel.\n\n**D'où viennent les chiffres.** Un fichier mis à jour chaque jour "
             "ouvré par GitHub, le lendemain de la séance : Euro Stoxx 50 et VSTOXX (fichiers "
             "officiels STOXX), €STR, Euribor et taux AAA de la zone euro (BCE), dividende "
             "net des 12 derniers mois (STOXX). L'appli lit le dernier fichier disponible.\n\n"
             "**Comment c'est calculé.** Les mêmes fonctions et les mêmes réglages par défaut "
             "que les pages produits, sous le modèle choisi dans la barre latérale. Le niveau "
             "habituel est celui d'un marché médian depuis 2004 (taux, vol et dividende "
             "médians), avec le même skew et le même spread émetteur.\n\n**Pourquoi c'est "
             "utile.** « Que vendrais-tu aujourd'hui ? » est une question classique "
             "d'entretien de Sales. La bonne réponse part du marché : taux hauts = capital "
             "garanti, vol haute = BRC et Discount.\n\n**Ce qu'il faut regarder.** Le "
             "graphe : au-dessus de 100, le produit paie mieux que d'habitude.")
    st.caption("Le marché décide de ce qui se vend : la bonne idée du jour est celle que le "
               "budget permet.")

    d = DJ.charge()
    jour = DJ.reglages_du_jour(d)
    hab = DJ.reglages_habituels(d, jour["origine"]["vol"])
    date = DJ.date_fr(jour["date"])
    age = DJ.anciennete(d)
    if date is None:
        st.warning("Fichier du marché du jour introuvable ou illisible : la page utilise les "
                   "réglages par défaut de la barre latérale et les médianes enregistrées. "
                   "Les chiffres restent justes pour ces réglages.")
    elif age is not None and age > 5:
        st.warning(f"Données du {date} : la mise à jour quotidienne n'a pas tourné depuis "
                   f"{age} jours. Les chiffres sont ceux de cette date.")
    manquants = [k for k, v in jour["origine"].items() if v is None]
    if date is not None and manquants:
        st.warning("Donnée absente du fichier, remplacée par la valeur par défaut : "
                   + ", ".join(PARAMETRES[k] for k in manquants) + ".")
    if jour["origine"]["vol"] in ("vol_realisee_3m", "vol_realisee_1m"):
        st.warning("VSTOXX indisponible : la vol du jour est une vol réalisée, plus basse que "
                   "la vol implicite en temps normal. Elle est comparée à la vol réalisée "
                   "habituelle, pas au VSTOXX.")

    # --- le marche du jour
    st.subheader("Le marché du jour" + (f" ({date})" if date else ""))
    niv = DJ.valeur(d, "eurostoxx50")
    mt = st.columns(4)
    mt[0].metric("Euro Stoxx 50", "—" if niv is None else f"{niv:,.2f}".replace(",", " "),
                 help="Clôture officielle STOXX (SX5E), contrôlée contre Yahoo Finance.")
    for col, cle, nom, aide in (
            (mt[1], "vol", "Vol ATM du jour", "VSTOXX : vol implicite 30 jours de l'Euro "
             "Stoxx 50, publiée par STOXX. À défaut, vol réalisée 3 mois."),
            (mt[2], "taux", "Taux 3 ans", "Taux zéro-coupon à 3 ans des emprunts d'État "
             "notés AAA de la zone euro (BCE), milieu de la gamme de maturités des produits."),
            (mt[3], "dividende", "Dividende", "Rendement net du dividende sur 12 mois : écart "
             "entre l'Euro Stoxx 50 dividendes nets réinvestis (SX5T) et l'indice de prix.")):
        col.metric(nom, f"{jour['exact'][cle]:.2f} %",
                   f"{jour['exact'][cle] - hab['exact'][cle]:+.2f} pt contre l'habituel",
                   delta_color="off", help=aide)

    with st.expander("Toutes les données et leurs sources"):
        champs = [("eurostoxx50", "Euro Stoxx 50 (points)", None),
                  ("vstoxx", "VSTOXX, vol implicite 30 jours", "vstoxx"),
                  ("vol_realisee_1m", "Vol réalisée 1 mois", None),
                  ("vol_realisee_3m", "Vol réalisée 3 mois", "vol_realisee_3m"),
                  ("estr", "€STR, taux au jour le jour", None),
                  ("euribor_3m", "Euribor 3 mois (moyenne du mois)", None),
                  ("taux_depot_bce", "Taux de dépôt de la BCE", None),
                  ("taux_1_an", "Taux AAA 1 an", None),
                  ("taux_2_ans", "Taux AAA 2 ans", None),
                  ("taux_3_ans", "Taux AAA 3 ans", "taux_3_ans"),
                  ("taux_5_ans", "Taux AAA 5 ans", None),
                  ("dividende", "Dividende net 12 mois", "dividende")]
        sources = {"eurostoxx50": "STOXX", "vstoxx": "STOXX", "vol_realisee_1m": "STOXX, calcul",
                   "vol_realisee_3m": "STOXX, calcul", "estr": "BCE", "euribor_3m": "BCE",
                   "taux_depot_bce": "BCE", "taux_1_an": "BCE", "taux_2_ans": "BCE",
                   "taux_3_ans": "BCE", "taux_5_ans": "BCE", "dividende": "STOXX, calcul"}
        tableau = []
        for cle, nom, h in champs:
            v = DJ.valeur(d, cle)
            med = DJ.habituel(d, h)[0] if h else None
            tableau.append({
                "Donnée": nom,
                "Valeur": "—" if v is None else (f"{v:,.2f}".replace(",", " ")
                                                 if cle == "eurostoxx50" else f"{v:.2f} %"),
                "Date": DJ.date_fr(DJ.date_champ(d, cle)) or "—",
                "Médiane depuis 2004": "—" if med is None else f"{med:.2f} %",
                "Source": sources[cle] + (" (ancienne valeur, source en panne)"
                                          if cle in (d.get("anciens") or []) else "")})
        st.table(tableau)
        ctrl = ((d.get("champs") or {}).get("eurostoxx50") or {}).get("controle_yahoo")
        st.caption(
            "Sources : fichiers historiques officiels STOXX (h_3msx5e, h_3msx5t, h_v2tx) et API "
            "de la BCE (data-api.ecb.europa.eu). Contrôles : le niveau de l'indice est "
            "comparé à Yahoo Finance"
            + (f" ({ctrl:,.2f})".replace(",", " ") if ctrl else "")
            + ", l'€STR au taux de dépôt de la BCE ; le 18/10/2026, le dividende net (2,1 %) "
            "était dans la fourchette du dividende implicite des options Eurex (2,1 à 2,4 %). "
            "Limites : le VSTOXX est une vol 30 jours, les produits durent de 1 à 5 ans ; le "
            "skew (2,5) et le spread émetteur (50 bp) ne sont pas mis à jour chaque jour.")

    st.button("Charger le marché du jour dans la barre latérale", key="aj_charger",
              on_click=_charge_jour,
              help="Écrit le scénario « Aujourd'hui » dans la barre latérale : toutes les pages "
                   "calculent alors avec le marché du jour.")

    # --- ce que chaque produit offre
    mk_j, mk_h = marche(m, jour["valeurs"]), marche(m, hab["valeurs"])
    with st.spinner("Calcul des six produits..."):
        res = analyse(mk_j, mk_h)
    j, h, effets = res["jour"], res["habituel"], res["effets"]
    res["rapports"], verdicts = {}, {}
    for p in ORDRE:
        res["rapports"][p], verdicts[p] = verdict(j[p]["valeur"], h[p]["valeur"])
    classement = sorted(ORDRE, key=lambda p: -res["rapports"][p])
    meilleur = classement[0]

    G.en_30_secondes(
        f"{NOMS[meilleur]} : {format_valeur(meilleur, j[meilleur]['valeur'])} "
        f"({GRANDEURS[meilleur].lower()}), contre {format_valeur(meilleur, h[meilleur]['valeur'])} "
        "dans un marché habituel. C'est le produit le mieux placé du jour.",
        (lambda t: t[:1].upper() + t[1:])(risque(meilleur, j)),
        {CP: "Un client prudent, qui croit à la hausse mais refuse de perdre son capital.",
         BRC: "Un client qui veut un revenu et pense que le marché va peu bouger.",
         AC: "Un client qui veut un revenu régulier et pense que le marché va stagner ou "
             "monter un peu.",
         BONUS: "Un client qui pense que le marché va stagner ou monter, sans plafonner son "
                "gain.",
         TW: "Un client sans vue, qui pense que le marché va peu bouger dans un sens ou dans "
             "l'autre.",
         DC: "Un client qui voulait acheter l'indice et pense qu'il ne va pas beaucoup "
             "monter."}[meilleur])

    st.subheader("Ce que chaque produit offre aujourd'hui")
    st.caption(f"Réglages par défaut de chaque page produit. Marché du jour : taux "
               f"{jour['valeurs']['taux']:.2f} %, dividende {jour['valeurs']['dividende']:.2f} %, "
               f"vol {jour['valeurs']['vol']:.0f} %. Marché habituel : taux "
               f"{hab['valeurs']['taux']:.2f} %, dividende {hab['valeurs']['dividende']:.2f} %, "
               f"vol {hab['valeurs']['vol']:.0f} %. Les deux avec skew {DJ.SKEW:.1f}, spread "
               f"{DJ.SPREAD} bp et les autres réglages de la barre latérale.")
    # pas le bandeau des pages produits : il cite la vol de la barre laterale, pas celle du jour
    st.caption("Pricing sous Bates : vol de départ du jour, autres paramètres de la barre "
               "latérale." if m.BATES else "Pricing sous Black-Scholes + skew, en volatilité "
               "locale.")

    lignes = []
    for i, p in enumerate(ORDRE):
        rapport = res["rapports"][p]
        indice = min(rapport, 9.99) * 100
        lignes.append({
            "produit": NOMS[p], "ordre": i, "indice": indice,
            "jour": format_valeur(p, j[p]["valeur"]), "habituel": format_valeur(p, h[p]["valeur"]),
            "verdict": verdicts[p],
            "etiquette": f"{format_valeur(p, j[p]['valeur'])} contre {format_valeur(p, h[p]['valeur'])}",
            "couleur": G.PRODUIT if verdicts[p] != MOINS else G.DIRECT})
    r_m = res["rapports"][meilleur]
    titre = (f"{NOMS[meilleur]} : {r_m:.1f} fois son niveau habituel" if r_m >= 1.5 else
             f"{NOMS[meilleur]} : {(r_m - 1) * 100:+.0f} % par rapport à d'habitude")
    G.affiche(graphe_attrait(lignes, titre, "100 = ce que le produit offre dans un marché "
                                            "médian depuis 2004. Survole une barre pour le détail."))

    st.table([{"Produit": NOMS[p], "Réglages": REGLAGES[p], "Ce qui est offert": GRANDEURS[p],
               "Aujourd'hui": format_valeur(p, j[p]["valeur"]),
               "Habituel": format_valeur(p, h[p]["valeur"]),
               "Verdict": verdicts[p],
               "Pourquoi": pourquoi(p, effets, j[p]["valeur"] - h[p]["valeur"]),
               "Effet de chaque paramètre pris seul": effets_seuls(p, effets)}
              for p in ORDRE])
    st.caption("Effet pris seul : on part du marché habituel et on ne change qu'un paramètre "
               "(taux, vol ou dividende). Les effets ne s'additionnent pas exactement, les "
               "paramètres interagissent. Dans la norme = moins de 10 % d'écart avec "
               "l'habituel.")
    if j[CP]["valeur"] < 0.35:
        st.caption("Capital garanti sous 35 % de participation : invendable en l'état (voir "
                   "sa page).")

    st.subheader("Pourquoi")
    for ligne in raisons_marche(d, jour, hab):
        st.markdown("- " + ligne)

    st.subheader("Que vendrais-tu aujourd'hui ?")
    reponse = " ".join(reponse_entretien(jour, hab, res, classement))
    st.info(f"**À dire en entretien** : « {reponse} »")
    st.caption("Réponse construite à partir des chiffres de la page : elle change avec le "
               "marché du jour et avec le modèle choisi.")

    st.subheader("Ouvrir un produit avec le marché du jour")
    st.caption("Charge le scénario « Aujourd'hui » dans la barre latérale et ouvre la page "
               "avec ses réglages par défaut : elle affiche le même chiffre qu'ici.")
    cols = st.columns(3)
    for i, p in enumerate(ORDRE):
        cols[i % 3].button(NOMS[p], key=f"aj_ouvre_{p}", on_click=_ouvre, args=(p,),
                           width="stretch")
