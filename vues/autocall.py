"""4 - Autocall Phoenix."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0
from communs import MAX_CACHE, sim_panier

CLASSIQUE, DECREMENT = "Indice classique", "Indice à décrément"

# Garde-fou memoire. Simuler un worst-of occupe ~16 octets par sous-jacent,
# par trajectoire et par pas (mesure) : 4 sous-jacents sur 6 ans avec 40 000
# trajectoires montent a 800 Mo pour la seule simulation, et le serveur
# gratuit (~1 Go) tombe. On borne la taille du calcul a 12,5 millions de
# points (~200 Mo) et a 20 000 trajectoires.
POINTS_PANIER = 12_500_000
MAX_PANIER = 20000
# Les comparaisons d'indices simulent trois fois : au plus 20 000 trajectoires.
MAX_COMPARAISON = 20000


def trajectoires_panier(n_paths, n_actifs, T):
    """Nombre de trajectoires d'un worst-of, plafonne selon la taille du calcul."""
    if n_actifs <= 1:
        return n_paths
    plafond = POINTS_PANIER // (n_actifs * M.nb_pas(T)) // 1000 * 1000
    return int(min(n_paths, MAX_PANIER, max(plafond, 5000)))


# =====================================================================
# SIMULATIONS ET CALCULS MIS EN CACHE
# =====================================================================

@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_decrement(T, r, vol_atm, skew, n, decrement, bates=None):
    """
    Indice a decrement : dividendes reinvestis (q = 0), D points retires par an.
    Black-Scholes : volatilite locale, comme les autres pages (moteur.simule_lv).
    """
    if bates is None:
        return M.simule_lv(S0, T, r, 0.0, vol_atm, skew, n, decrement=decrement)
    return M.simule_bates(S0, T, r, 0.0, n, decrement=decrement, **bates)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def sim_stress(n_actifs, corr, corr_stress, seuil, T, r, q, sigma, n):
    """Worst-of avec correlation de stress, et part du temps passee en krach."""
    worst, diag = M._panier_stress(n_actifs, corr, corr_stress, seuil, S0, T, r, q, sigma, n)
    return worst, diag["part_stress"]


def resume(ch, T, r, produit, tau=None, recovery=0.4):
    """Coupon par observation, duree de vie, proba de rappel et de perte."""
    nominal, n_obs, trig, bar_cp, bar_cap, memoire, marge = produit
    cpn = M.coupon_autocall(ch, T, r, nominal, n_obs, trig, bar_cp, bar_cap, memoire,
                            marge, tau, recovery)
    duree, p_rappel, p_perte = M.stats_autocall(ch, T, n_obs, trig, bar_cap)
    return {"coupon": cpn, "duree": duree, "p_rappel": p_rappel, "p_perte": p_perte}


def defauts_internes(n, T, spread, recovery):
    """Memes defauts que la barre laterale (meme graine)."""
    return M.tire_defaut(n, T, spread, recovery) if spread > 0 else None


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def compare_indices(T, r, q, vol_atm, skew, bates, n, decrement, produit, spread, recovery):
    """
    Le meme autocall sur trois sous-jacents, a vue de marche egale (memes
    tirages, meme vol ou memes parametres Bates) :
      classique  indice de prix, dividende q
      pct        dividendes reinvestis, decrement fixe en % du niveau courant :
                 equivaut a un dividende certain (effet niveau seul)
      points     decrement fixe en points : pese plus quand l'indice baisse
    Seuls les resultats sont gardes en cache, pas les trajectoires.
    """
    tau = defauts_internes(n, T, spread, recovery)
    out = {}
    for nom, q_, dec in (("classique", q, 0.0), ("pct", decrement / S0, 0.0),
                         ("points", 0.0, decrement)):
        if bates is None:
            ch = M.simule_lv(S0, T, r, q_, vol_atm, skew, n, decrement=dec)[0]
        else:
            ch = M.simule_bates(S0, T, r, q_, n, decrement=dec, **bates)[0]
        out[nom] = resume(ch, T, r, produit, tau, recovery)
        out[nom]["forward"] = float(ch[:, -1].astype(np.float64).mean())
        del ch
    return out


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def compare_stress(n_actifs, corr, corr_stress, seuil, T, r, q, sigma, n, produit, spread,
                   recovery):
    """
    Worst-of en correlation constante, avec correlation de stress, et en
    correlation constante au niveau de stress. Memes tirages : la premiere
    ligne redonne exactement les chemins de simule_panier.
    """
    tau = defauts_internes(n, T, spread, recovery)
    out = {}
    for nom, (c_n, c_s) in (("constante", (corr, corr)), ("stress", (corr, corr_stress)),
                            ("haute", (corr_stress, corr_stress))):
        ch, diag = M._panier_stress(n_actifs, c_n, c_s, seuil, S0, T, r, q, sigma, n)
        out[nom] = resume(ch, T, r, produit, tau, recovery)
        out[nom]["part_stress"] = diag["part_stress"]
        del ch
    return out


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def grecques_desk(T, r, q, sigma, n_obs, trig, bar_cp, bar_cap, memoire, coupon, nominal,
                  n_actifs, correlation, correlation_stress, seuil_stress, bates, decrement):
    """Grecques du desk, 10 000 trajectoires, memes tirages a chaque choc."""
    return M.grecques_autocall(T, r, q, sigma, n_obs, trig, bar_cp, bar_cap, memoire, coupon,
                               nominal, n_actifs=n_actifs, correlation=correlation,
                               correlation_stress=correlation_stress,
                               seuil_stress=seuil_stress, bates=bates, decrement=decrement,
                               n_paths=10000)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def table_worst_of(T, r, q, v_p, n_paths, n_obs, trig, bar_cp, bar_cap, memoire, marge,
                   nominal, par_an):
    """Effet du nombre de sous-jacents. Seul le tableau est garde en cache."""
    lignes = []
    for na in [1, 2, 3, 4]:
        for co in ([0.0] if na == 1 else [0.80, 0.50, 0.25]):
            if na == 1:
                c_, _ = M.simule_bs(S0, T, r, q, v_p, min(n_paths, 20000))
            else:
                c_, _ = M.simule_panier(na, co, S0, T, r, q, v_p,
                                        trajectoires_panier(n_paths, na, T))
            cp = M.coupon_autocall(c_, T, r, nominal, n_obs, trig, bar_cp, bar_cap, memoire,
                                   marge)
            _, _, pp = M.stats_autocall(c_, T, n_obs, trig, bar_cap)
            del c_
            lignes.append({"Sous-jacents": str(na),
                           "Corrélation": "-" if na == 1 else f"{co:.2f}",
                           "Coupon annuel": f"{cp*par_an*100:,.2f}%",
                           "Proba de perte": f"{pp*100:,.1f}%"})
    return lignes


# =====================================================================
# AFFICHAGE
# =====================================================================

def escalier(graphe, dates, niveaux, texte, T):
    """
    Pose le trigger degressif sur le faisceau de trajectoires : un palier
    par date d'observation (le niveau teste a cette date). Les extremes ont
    deja ete passes au graphe comme reperes invisibles, l'echelle les
    contient.
    """
    df = pd.DataFrame({"t": [0.0] + list(dates), "y": list(niveaux) + [niveaux[-1]],
                       "txt": [f"{v:,.0f}" for v in list(niveaux) + [niveaux[-1]]]})
    ligne = alt.Chart(df).mark_line(interpolate="step-after", color=G.GAIN, strokeWidth=1.6,
                                    strokeDash=[4, 4], clip=True).encode(
        x="t:Q", y="y:Q", tooltip=[alt.Tooltip("txt:N", title="Trigger de rappel")])
    etiquette = alt.Chart(pd.DataFrame([{"t": T * 1.02, "y": niveaux[-1], "txt": texte}])).mark_text(
        align="right", baseline="bottom", dy=-3, fontSize=11, fontWeight="bold",
        color=G.GAIN).encode(x="t:Q", y="y:Q", text="txt:N")
    graphe.layer += [ligne, etiquette]
    return graphe


def entier(v):
    return f"{v:,.0f}".replace(",", " ")


def lecture(valeur, positif, negatif, nul=None):
    if nul is not None and abs(valeur) < 1e-12:
        return nul
    return positif if valeur > 0 else negatif


def affiche(m):
    BATES, bandeau_modele, defauts, marge, n_paths = m.BATES, m.bandeau_modele, m.defauts, m.marge, m.n_paths
    nominal, q, r, recovery, skew = m.nominal, m.q, m.r, m.recovery, m.skew
    spread, trajectoires, vol_atm = m.spread, m.trajectoires, m.vol_atm

    st.title("Autocall Phoenix", help="**Ce que c'est.** Le produit phare des desks. Coupons conditionnels, et remboursement\nanticipé automatique si le sous-jacent est au-dessus d'un niveau à une date\nd'observation.\n\n**Les trois niveaux à distinguer.**\n- **Trigger** : au-dessus, le produit s'arrête et le client est remboursé.\n- **Barrière de coupon** : au-dessus, le coupon est payé.\n- **Barrière de capital** : observée à l'échéance seulement ; en dessous, le client encaisse\n  toute la baisse.\n\n**Le piège commercial.** Le rappel anticipé coupe les coupons futurs. Regarde la durée de\nvie moyenne : le produit dure rarement jusqu'au bout.\n\n**Le worst-of.** Active plusieurs sous-jacents : le plus mauvais décide de tout. Le coupon\ngrimpe, et la probabilité de perte aussi. C'est la version réellement dominante sur les\ndesks.\n\n**Les variantes du marché.**\n- **Step-down** : le trigger baisse avec le temps, le rappel devient plus facile.\n- **Indice à décrément** : l'indice réinvestit ses dividendes et retire un montant fixe\n  chaque année. Coupon plus haut, risque plus haut, plus de risque de dividende pour la banque.\n- **Corrélation de stress** : sur un worst-of, la corrélation monte dans les krachs. Le\n  desk qui l'ignore surpaie le client.\n- **Grecques du desk** : ce que la banque doit couvrir une fois le produit vendu.")
    st.caption("Le produit phare des desks.")
    bandeau_modele()

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturité maximale (années)", 1.0, 6.0, 3.0, 0.5,
                  help="Durée MAXIMALE : le produit est presque toujours rappelé avant.",
                  key="ac_maturite", bind="query-params")
    freq = c1.selectbox("Fréquence d'observation",
                        ["Trimestrielle", "Semestrielle", "Annuelle"],
                        key="ac_frequence", bind="query-params")
    trigger = c2.slider("Trigger de rappel (%)", 80, 110, 100, 5,
                        key="ac_trigger", bind="query-params")
    bar_cp = c2.slider("Barrière de coupon (%)", 50, 100, 70, 5,
                       help="Le coupon n'est payé que si le sous-jacent est au-dessus à "
                            "la date d'observation. C'est ce qui distingue un Phoenix "
                            "d'un BRC.", key="ac_bar_coupon", bind="query-params")
    bar_cap = c3.slider("Barrière de capital (%)", 40, 90, 60, 5,
                        help="Observée uniquement A L'ÉCHÉANCE. Le vrai risque du produit.",
                        key="ac_bar_capital", bind="query-params")
    memoire = c3.checkbox("Effet mémoire", value=True, key="ac_memoire", bind="query-params")

    st.markdown("---")
    s1, s2 = st.columns(2)
    step_down = s1.checkbox(
        "Trigger dégressif (step-down)", key="ac_step_down", bind="query-params",
        help="Le seuil de rappel baisse chaque année. Le client sort plus souvent avec son "
             "gain même si le marché a un peu baissé, et va moins souvent jusqu'à "
             "l'échéance, là où se joue la perte en capital. Le prix : moins de coupons au "
             "total, puisque le produit s'arrête plus tôt ; le coupon par période bouge peu, "
             "car la banque prélève sa marge sur une vie plus courte.")
    if step_down:
        pas_sd = s1.slider("Baisse du trigger par an (points de %)", 1, 10, 5, 1,
                           help="5 points : 100 % la première année, 95 % la deuxième, "
                                "90 % la troisième...", key="ac_sd_pas", bind="query-params")
        plancher_sd = s1.slider("Plancher du trigger (%)", 60, 100, 80, 5,
                                help="Le trigger ne descend jamais sous ce niveau.",
                                key="ac_sd_plancher", bind="query-params")
    indice = s2.radio(
        "Sous-jacent", [CLASSIQUE, DECREMENT], horizontal=True,
        key="ac_indice", bind="query-params",
        help="Indice classique : un indice de prix, dont les dividendes vont à la banque. "
             "Indice à décrément : l'indice réinvestit ses dividendes et retire chaque "
             "année un montant fixe de points. Très répandu dans les autocalls vendus en "
             "France.")
    if indice == DECREMENT:
        pts_dec = s2.slider("Décrément (points d'indice par an)", 10, 100, 50, 5,
                            help="Montant fixe retire chaque année, en points d'indice. "
                                 "Fixe en points, pas en % : il pèse plus lourd quand "
                                 "l'indice baisse.", key="ac_dec_points", bind="query-params")
        niveau_dec = s2.select_slider("Niveau de l'indice à l'émission (points)",
                                      [1000, 2000, 3000, 4000, 5000], 1000,
                                      help="Convertit le décrément en % du niveau initial : "
                                           "50 points sur 1 000 = 5 % par an, sur 5 000 = "
                                           "1 % par an.", key="ac_dec_niveau",
                                      bind="query-params")

    st.markdown("---")
    w1, w2 = st.columns(2)
    n_actifs = w1.select_slider(
        "Nombre de sous-jacents (worst-of)", [1, 2, 3, 4], 1,
        help="Le plus mauvais des sous-jacents décide de tout. C'est la structure "
             "réellement dominante sur les desks.", key="ac_sous_jacents", bind="query-params")
    correl = w2.slider(
        "Corrélation entre sous-jacents", 0.0, 0.95, 0.60, 0.05,
        disabled=(n_actifs == 1),
        help="Le paramètre de pricing dominant sur un worst-of. Corrélation haute : les "
             "actifs bougent ensemble, le panier ressemble à un actif unique, coupon "
             "faible. Corrélation basse : il y a presque toujours un traînard, coupon "
             "élevé et risque élevé.", key="ac_correlation", bind="query-params")
    # pas de case grisee en mono sous-jacent : la case n'apparait qu'avec un panier
    stress = n_actifs > 1 and w1.checkbox(
        "Corrélation de stress (krach)", key="ac_stress", bind="query-params",
        help="Fait de marché : dans les ventes paniques, tout baisse en même temps et les "
             "corrélations montent. Ici, la corrélation passe au niveau de krach tant que "
             "le marché est à plus du seuil sous son plus haut.")
    if stress:
        correl_stress = w1.slider(
            "Corrélation en krach", 0.0, 0.95, 0.90, 0.05,
            help="Niveau atteint par la corrélation pendant les krachs. 90 % : tout chute "
                 "ensemble.", key="ac_correl_stress", bind="query-params")
        seuil_stress = w2.slider(
            "Seuil de krach (% sous le plus haut)", 5, 30, 10, 5,
            help="Le marché (moyenne du panier) est en krach quand il est à plus de ce "
                 "pourcentage sous son plus haut. À 10 %, un marché a 20-30 % de vol y passe "
                 "plus de la moitié du temps : monte le seuil pour ne garder que les vrais "
                 "krachs.", key="ac_seuil_stress", bind="query-params")

    decrement = indice == DECREMENT and n_actifs == 1
    D = M.decrement_base100(pts_dec, niveau_dec) if decrement else 0.0
    if indice == DECREMENT and n_actifs > 1:
        s2.caption("Indice à décrément : mono sous-jacent seulement. Le worst-of reste sur "
                   "des indices classiques.")
    elif decrement:
        s2.caption(f"{pts_dec} points sur un indice à {niveau_dec:,}".replace(",", " ")
                   + f" = **{D:,.2f} %** du niveau initial par an, contre {q*100:,.2f} % de "
                   "dividende sur l'indice classique.")

    par_an = {"Trimestrielle": 4, "Semestrielle": 2, "Annuelle": 1}[freq]
    n_obs = max(int(T * par_an), 1)
    trig = trigger / 100
    if step_down:
        trig = tuple(M.trigger_degressif(n_obs, trigger / 100, pas_sd / 100,
                                         min(plancher_sd, trigger) / 100,
                                         obs_par_palier=par_an))
    seuils = M.niveaux_rappel(trig, n_obs)
    escalier_visible = step_down and n_obs > 1
    bas_sd = min(seuils[:-1]) if n_obs > 1 else seuils[0]

    v_bs = M.vol_au_strike(bar_cap / 100, vol_atm, skew)
    bates = (dict(v0=vol_atm ** 2, theta=m.theta_v ** 2, kappa=m.kappa, xi=m.xi, rho=m.rho,
                  lam=m.lam, mu_j=m.mu_j, sig_j=m.sig_j) if BATES else None)
    n_panier = trajectoires_panier(n_paths, n_actifs, T)
    v_panier = vol_atm if BATES else v_bs
    part_stress = None
    if n_actifs > 1:
        if stress:
            ch, part_stress = sim_stress(n_actifs, correl, correl_stress, seuil_stress / 100,
                                         T, r, q, v_panier, n_panier)
        else:
            ch, _ = sim_panier(n_actifs, correl, S0, T, r, q, v_panier, n_panier)
        if BATES:
            st.warning("Le worst-of est simulé en Black-Scholes multi-actifs : un Bates "
                       "multi-dimensionnel demanderait une matrice de corrélation entre "
                       "les volatilités de chaque actif, ce qui dépasse le cadre de cet "
                       "outil. Le mécanisme de corrélation reste juste.")
    elif decrement:
        ch, _ = sim_decrement(T, r, vol_atm, skew, n_paths, D, bates)
    else:
        ch, _ = trajectoires(T, bar_cap / 100)

    tau = defauts(len(ch), T)
    cpn = M.coupon_autocall(ch, T, r, nominal, n_obs, trig, bar_cp / 100,
                            bar_cap / 100, memoire, marge, tau, recovery)
    duree, p_rappel, p_perte = M.stats_autocall(ch, T, n_obs, trig, bar_cap / 100)

    mt = st.columns(4)
    mt[0].metric("Coupon par observation", f"{cpn*100:,.2f}%")
    mt[1].metric("Équivalent annuel", f"{cpn*par_an*100:,.2f}% p.a.")
    mt[2].metric("Durée de vie moyenne", f"{duree:,.2f} ans")
    mt[3].metric("Proba de perte en capital", f"{p_perte*100:,.1f}%")
    if n_panier < n_paths:
        st.caption(f"Worst-of calculé sur {entier(n_panier)} trajectoires au lieu de "
                   f"{entier(n_paths)}, pour tenir dans la mémoire du serveur.")

    produit = (nominal, n_obs, trig, bar_cp / 100, bar_cap / 100, memoire, marge)

    # --- step-down : ce que change le trigger degressif, sur les memes trajectoires
    if step_down:
        fixe = resume(ch, T, r, produit[:2] + (trigger / 100,) + produit[3:], tau, recovery)
        deg = {"coupon": cpn, "duree": duree, "p_rappel": p_rappel, "p_perte": p_perte}
        st.table([{"Seuil de rappel": nom,
                   "Coupon annuel": f"{x['coupon']*par_an*100:,.2f}%",
                   "Rappel anticipé": f"{x['p_rappel']*100:,.1f}%",
                   "Durée de vie": f"{x['duree']:,.2f} ans",
                   "Proba de perte": f"{x['p_perte']*100:,.1f}%"}
                  for nom, x in ((f"Fixe à {trigger} %", fixe),
                                 (f"Step-down {trigger} % -> {bas_sd*100:,.0f} %", deg))])
        st.caption("Le step-down rend le rappel plus facile avec le temps : le client sort plus "
                   "souvent avec son gain et va moins souvent à l'échéance, là où se joue la "
                   f"perte. Effet sur le coupon : {fixe['coupon']*par_an*100:,.2f} % -> "
                   f"{cpn*par_an*100:,.2f} % par an.")

    # --- indice a decrement : classique contre decrement, a vue de marche egale
    if decrement:
        n_cmp = min(n_paths, MAX_COMPARAISON)
        res = compare_indices(T, r, q, vol_atm, skew, bates, n_cmp, D, produit, spread, recovery)
        cl, pc, po = res["classique"], res["pct"], res["points"]
        st.table([{"Sous-jacent": nom,
                   "Coupon annuel": f"{x['coupon']*par_an*100:,.2f}%",
                   "Proba de perte": f"{x['p_perte']*100:,.1f}%",
                   "Durée de vie": f"{x['duree']:,.2f} ans",
                   "Niveau moyen à l'échéance": f"{x['forward']:,.1f}"}
                  for nom, x in ((f"Indice classique (dividende {q*100:,.2f} %)", cl),
                                 (f"Décrément {D:,.2f} % du niveau courant", pc),
                                 (f"Décrément {pts_dec} points (= {D:,.2f} % au départ)", po))])
        st.caption("Mêmes tirages et même vue de marché pour les trois lignes"
                   + (f", sur {entier(n_cmp)} trajectoires" if n_cmp < n_paths else "")
                   + ". La ligne du milieu isole l'effet niveau (décrément contre dividende), "
                     "la dernière ajoute l'effet points.")
        if po["coupon"] > cl["coupon"]:
            phrase = ("« Le coupon est plus élevé parce que l'indice retire chaque année un "
                      "montant fixe supérieur aux dividendes ; c'est aussi pour ça que le risque "
                      "monte : ce montant fixe pèse plus lourd quand le marché baisse. »")
        else:
            phrase = ("« Ici le décrément retire moins que les dividendes : l'indice monte plus "
                      "vite que l'indice de prix et le coupon baisse. Le risque, lui, monte "
                      "quand même : le montant fixe pèse plus lourd quand le marché baisse. »")
        st.info(
            f"**En une phrase pour le client** : {phrase}\n\n"
            f"**Pourquoi le coupon bouge.** L'indice classique perd {q*100:,.2f} % par an de "
            f"dividendes, l'indice à décrément {D:,.2f} % au départ. Niveau moyen à "
            f"l'échéance : {cl['forward']:,.1f} contre {po['forward']:,.1f}. Plus ce niveau est "
            "bas, moins le produit est rappelé et plus le put que le client vend vaut cher : "
            f"coupon {cl['coupon']*par_an*100:,.2f} % -> {po['coupon']*par_an*100:,.2f} % par an. "
            "Et la banque n'a plus de risque de dividende à couvrir.\n\n"
            f"**Pourquoi la perte monte.** {pts_dec} points pèsent {D:,.1f} % du niveau de "
            f"départ, {2*D:,.1f} % quand l'indice a perdu la moitié de sa valeur : le "
            "décrément accélère la chute exactement dans les scénarios où se joue la perte "
            "en capital. À poids égal au départ, la proba de perte passe de "
            f"{pc['p_perte']*100:,.1f} % (décrément en %) à "
            f"{po['p_perte']*100:,.1f} % (décrément en points).")

    # --- correlation de stress : le message contre-intuitif
    if stress:
        res = compare_stress(n_actifs, correl, correl_stress, seuil_stress / 100, T, r, q,
                             v_panier, n_panier, produit, spread, recovery)
        co, sr, ha = res["constante"], res["stress"], res["haute"]
        st.table([{"Corrélation": nom,
                   "Coupon annuel": f"{x['coupon']*par_an*100:,.2f}%",
                   "Proba de perte": f"{x['p_perte']*100:,.1f}%",
                   "Durée de vie": f"{x['duree']:,.2f} ans"}
                  for nom, x in ((f"Constante {correl*100:,.0f} %", co),
                                 (f"{correl*100:,.0f} %, {correl_stress*100:,.0f} % en krach", sr),
                                 (f"Constante {correl_stress*100:,.0f} %", ha))])
        ecart = (co["coupon"] - sr["coupon"]) * par_an * 100
        if ecart > 0:
            st.info(
                "**Le sens est contre-intuitif : la corrélation de krach fait BAISSER le coupon "
                f"et la perte.** Coupon {co['coupon']*par_an*100:,.2f} % -> "
                f"{sr['coupon']*par_an*100:,.2f} % par an, proba de perte "
                f"{co['p_perte']*100:,.1f} % -> {sr['p_perte']*100:,.1f} %. Quand tout chute "
                "ensemble, le plus mauvais des sous-jacents est moins mauvais : moins de "
                "dispersion, donc le put worst-of que le client vend vaut moins.\n\n"
                f"**Pour le desk** : pricer en corrélation constante, c'est surpayer le client "
                f"de {ecart:,.2f} points par an, et perdre quand la corrélation monte en krach. "
                "C'est le skew de corrélation : un desk le facture, ou il le subit.")
        else:
            st.info(
                "**Ici la corrélation de krach est plus basse que la corrélation normale** : "
                "plus de dispersion dans les krachs, donc un put worst-of plus cher et un "
                f"coupon plus haut ({co['coupon']*par_an*100:,.2f} % -> "
                f"{sr['coupon']*par_an*100:,.2f} % par an). Le marché fait l'inverse : les "
                "corrélations montent dans les krachs.")
        st.caption(f"Marché en krach (plus de {seuil_stress} % sous son plus haut) "
                   f"{part_stress*100:,.0f} % du temps. Les vols individuelles restent fixes : "
                   "dans un vrai krach, la hausse de vol s'ajoute et fait mal.")

    indices = M.idx_obs(ch.shape[1] - 1, n_obs)
    tt = np.linspace(0, T, ch.shape[1])

    # --- graphe 1 : trajectoires et rappels
    couleurs, statuts, t_fin, rappels = [], [], [], []
    for i in range(60):
        chem = ch[i]
        rappel_a = next((k for k, j in enumerate(indices[:-1])
                         if chem[j] >= seuils[k] * S0), None)
        if rappel_a is not None:
            j = indices[rappel_a]
            couleurs.append(G.GAIN)
            statuts.append(f"rappelé à l'observation {rappel_a + 1} ({tt[j]:,.2f} an) : "
                           "nominal remboursé + coupon")
            t_fin.append(tt[j])
            rappels.append({"t": tt[j], "y": float(chem[j]), "couleur": G.GAIN})
        elif chem[-1] < bar_cap / 100 * S0:
            couleurs.append(G.PERTE)
            statuts.append(f"perte en capital : rembourse {nominal * chem[-1] / S0:,.0f}")
            t_fin.append(T)
        else:
            couleurs.append(G.NIVEAU)
            statuts.append("va au terme sans perte : nominal remboursé")
            t_fin.append(T)
    niveaux = [(bar_cp / 100 * S0, f"barrière coupon  {bar_cp/100*S0:,.0f}", G.ACCENT,
                T * 0.62),
               (bar_cap / 100 * S0, f"barrière capital  {bar_cap/100*S0:,.0f}", G.PERTE)]
    if escalier_visible:
        paliers = [s * S0 for s in seuils[:-1]]
        # reperes invisibles : l'echelle du graphe contient tout l'escalier
        niveaux += [(max(paliers), "", "transparent"), (min(paliers), "", "transparent")]
    else:
        niveaux.insert(0, (trigger / 100 * S0, f"trigger de rappel  {trigger/100*S0:,.0f}",
                           G.GAIN))
    sous_titre = ("Les traits verticaux sont les dates d'observation. Survole une trajectoire "
                  "pour voir son issue.")
    if escalier_visible:
        sous_titre = (f"Le trigger descend de {pas_sd} points par an jusqu'à "
                      f"{min(paliers):,.0f}. " + sous_titre)
    graphe = G.trajectoires(
        ch, T, niveaux,
        couleurs, f"Durée de vie moyenne : {duree:,.2f} ans sur {T:,.1f} possibles",
        sous_titre,
        ylabel="Worst-of du panier" if n_actifs > 1 else "Sous-jacent", n_max=60,
        zone_sous=bar_cap / 100 * S0, statuts=statuts, t_fin=t_fin, points=rappels,
        dates_v=[tt[j] for j in indices[:-1]],
        legende=[("Rappelé par anticipation", G.GAIN), ("Perte en capital", G.PERTE),
                 ("Va au terme sans perte", G.NIVEAU)])
    if escalier_visible:
        graphe = escalier(graphe, [tt[j] for j in indices[:-1]], paliers,
                          f"trigger dégressif  {paliers[0]:,.0f} -> {paliers[-1]:,.0f}", T)
    G.affiche(graphe)

    # --- graphe 2 : quand le produit se termine
    probas, vivant = [], np.ones(len(ch), dtype=bool)
    for k, j in enumerate(indices[:-1]):
        rappel = vivant & (ch[:, j] >= seuils[k] * S0)
        probas.append(rappel.mean())
        vivant = vivant & ~rappel
    probas.append(vivant.mean())
    labels = [f"Obs {k+1}" for k in range(len(probas) - 1)] + ["Terme"]
    couleurs = [G.GAIN] * (len(probas) - 1) + [G.NIVEAU]

    cumul = np.cumsum(probas) * 100
    infos = [("Date", [f"{tt[j]:,.2f} an" for j in indices[:-1]] + [f"{T:,.2f} an"]),
             ("Terminé à cette date ou avant", [G.pct(v, 0) for v in cumul])]
    if step_down:
        infos.insert(1, ("Trigger de rappel", [f"{s*100:,.0f} %" for s in seuils[:-1]] + ["-"]))
    graphe = G.barres(
        labels, np.array(probas) * 100, couleurs,
        f"{(1-probas[-1])*100:,.0f}% des scénarios se terminent avant l'échéance",
        "Chaque rappel anticipé coupe tous les coupons restants", "Probabilité (%)",
        nom_x="Observation", infos=infos)
    G.affiche(graphe)

    # --- grecques du desk
    if st.checkbox("Grecques du desk", key="ac_voir_grecques", bind="query-params",
                   help="Ce que la banque doit couvrir une fois le produit vendu : ses "
                        "sensibilités sont l'opposé de celles du client. Calcul par chocs "
                        "sur 10 000 trajectoires, mêmes tirages à chaque choc."):
        g = grecques_desk(T, r, 0.0 if decrement else q, v_panier if n_actifs > 1 else v_bs,
                          n_obs, trig, bar_cp / 100, bar_cap / 100, memoire, float(cpn),
                          nominal, n_actifs, correl if n_actifs > 1 else 0.0,
                          correl_stress if stress else None,
                          seuil_stress / 100 if stress else 0.10,
                          bates if n_actifs == 1 else None, D)
        lignes = [
            ("Delta", f"{g['delta']:+,.2f} EUR par point",
             f"Acheter {g['couverture']*100:,.0f} % du nominal en sous-jacent "
             f"({G.eur(g['couverture']*nominal)}), et ajuster chaque jour."
             if g["couverture"] > 0 else
             f"Vendre {-g['couverture']*100:,.0f} % du nominal en sous-jacent, et ajuster "
             "chaque jour."),
            ("Gamma", f"{g['gamma']:+,.3f} EUR par point, par point",
             lecture(g["gamma"], "Long gamma : il rachète quand le marché baisse et revend "
                                 "quand il monte. Il gagne si le marché bouge beaucoup.",
                     "Court gamma : il vend quand le marché baisse et rachète quand il monte. "
                     "Il perd si le marché bouge beaucoup.")),
            ("Vega", f"{g['vega']:+,.2f} EUR par point de vol",
             lecture(g["vega"], "Long vega : le client lui a vendu la protection sous la "
                                "barrière. Il revend de la vol longue sur le marché.",
                     "Court vega : il rachète de la vol sur le marché.")),
        ]
        if n_actifs > 1:
            lignes.append(("Corrélation", f"{g['correlation']:+,.2f} EUR par point de corrélation",
                           lecture(g["correlation"],
                                   "Long corrélation : il gagne si les actifs bougent ensemble.",
                                   "Court corrélation : le put worst-of qu'il détient vaut moins "
                                   "quand tout chute ensemble. Presque impossible à couvrir "
                                   "sur le marché.")))
        lignes += [
            ("Dividende", f"{g['dividende'] + 0.0:+,.2f} EUR par point de dividende",
             lecture(g["dividende"], "Long dividendes : moins de dividendes = forward plus haut "
                                     "= produit plus cher à servir. Il vend des dividendes "
                                     "futurs pour se couvrir.",
                     "Court dividendes : il rachète des dividendes futurs pour se couvrir.",
                     "Aucun risque : l'indice à décrément réinvestit ses dividendes. C'est sa "
                     "raison d'être pour la banque.")),
            ("Rho", f"{g['rho']:+,.2f} EUR par point de taux",
             lecture(g["rho"], "Gagne si les taux montent : la partie obligataire du produit "
                               "vaut moins pour le client.",
                     "Perd si les taux montent.")),
            ("Theta", f"{g['theta']:+,.2f} EUR par jour",
             lecture(g["theta"], "Gagne à marché inchangé, chaque jour qui passe.",
                     "Perd à marché inchangé, chaque jour qui passe : le prix de son gamma et "
                     "de son vega.")),
        ]
        st.table([{"Grecque": a, "Desk": b, "Ce que le desk fait": c} for a, b, c in lignes])
        texte = ("**Pourquoi le desk est long vega et long dividendes.** En vendant l'autocall, "
                 "la banque a acheté au client une protection contre la baisse (le put sous la "
                 "barrière de capital), payée par les coupons. Elle gagne si la vol monte, et "
                 "si les dividendes montent (forward plus bas, rappels plus rares à financer). "
                 "Pour se couvrir, les desks revendent de la vol longue et des dividendes "
                 "futurs : c'est pour ça que les dividendes futurs cotent sous leur valeur "
                 "attendue. En 2020, l'annulation des dividendes a coûté cher à tous les desks "
                 "d'autocalls.")
        if decrement:
            texte += ("\n\n**Sur l'indice à décrément**, la sensibilité au dividende disparaît : "
                      "la banque ne porte plus ce risque, et le client reçoit un coupon plus "
                      "haut en échange.")
        if n_actifs > 1:
            texte += ("\n\n**Pourquoi il est court corrélation.** Le client lui a vendu un put "
                      "sur le plus mauvais des sous-jacents, qui vaut le plus quand les actifs "
                      "se dispersent. Si la corrélation monte (krach), ce put perd de sa "
                      "valeur et le desk perd. Ce risque se couvre mal : il reste dans les "
                      "livres de la banque.")
        st.info(texte)
        st.caption(f"Valeur du produit pour le client : {G.eur(g['pv_client'])} pour "
                   f"{G.eur(nominal)} de nominal, hors risque émetteur. Point de vue du desk : "
                   "l'opposé des sensibilités du client.")

    if st.checkbox("Effet du nombre de sous-jacents", key="ac_voir_worst_of", bind="query-params"):
        v_p = vol_atm if BATES else v_bs
        st.table(table_worst_of(T, r, q, v_p, n_paths, n_obs, trig, bar_cp / 100,
                                bar_cap / 100, memoire, marge, nominal, par_an))
        st.caption("Lis les deux colonnes ensemble. Le coupon double entre un mono "
                   "sous-jacent et un worst-of décorrélé — et la probabilité de perte "
                   "aussi. Un client qui ne regarde que le coupon ne voit qu'une moitié "
                   "du produit.")

    st.info(
        f"""**Durée de vie moyenne : {duree:,.2f} ans sur {T:,.1f} possibles.** Le rappel anticipé coupe les coupons futurs. Un client qui compare le coupon affiché à un rendement obligataire se trompe : il ne le touchera peut-être qu'une ou deux fois."""
        + ("\n\n**Sur un worst-of, le coupon spectaculaire à une contrepartie simple** : "
           "il faut que TOUS les sous-jacents tiennent, pas un seul. Plus ils sont "
           "décorrélés, plus il est probable qu'au moins un décroche."
           if n_actifs > 1 else ""))
