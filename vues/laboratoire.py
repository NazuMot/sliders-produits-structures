"""Laboratoire modele."""

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

import moteur as M
import graphiques as G
import calibration as C
from moteur import S0
from communs import MAX_CACHE, sim_bates, sim_bs


MODELE_BATES = "Bates (vol stochastique + sauts)"
# reglage "curseurs" de data/bates_calibre.json -> cle du curseur dans barre.py
CLES_BARRE = {"vol_atm": "vol", "theta_v": "vol_lt", "kappa": "kappa", "xi": "vol_vol",
              "rho": "rho", "lam": "sauts", "mu_j": "taille_saut", "sig_j": "disp_saut"}
NOMS_SMILE = ["Marche Eurex (prix de reglement)", "Bates calibre", "Bates aux crans des curseurs",
              "Black-Scholes plat"]


@st.cache_data(show_spinner=False)
def _surface():
    """Calibrage stocke (data/bates_calibre.json) et points de marche Eurex : lus une fois."""
    return C.charge_parametres(), C.charge_options()


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE)
def _smiles(echeance):
    """
    Smiles d'une echeance : points de marche, Bates calibre et Bates aux crans
    des curseurs (formule fermee, 1 ms), avec le taux et le dividende
    implicites de l'echeance (forward et actualisation Eurex).
    """
    cal, o = _surface()
    sel = o["echeance"] == np.datetime64(echeance)
    ordre = np.argsort(o["strike_pct"][sel])
    T = float(o["T"][sel][0])
    r_e = -math.log(float(o["DF"][sel][0])) / T
    q_e = r_e - math.log(float(o["F"][sel][0]) / o["spot"]) / T
    ks = o["strike_pct"][sel][ordre]
    grille = np.linspace(ks.min() - 0.03, ks.max() + 0.03, 90)
    p, pc = cal["parametres"], C.depuis_curseurs(cal["curseurs"])
    return {"T": T, "r": r_e, "q": q_e, "ks": ks, "K": o["K"][sel][ordre],
            "vol": o["vol"][sel][ordre], "call": o["call"][sel][ordre],
            "oi": o["open_interest"][sel][ordre], "grille": grille,
            "cal_grille": C.vol_bates(p, T, grille, r_e, q_e),
            "curs_grille": C.vol_bates(pc, T, grille, r_e, q_e),
            "cal": C.vol_bates(p, T, ks, r_e, q_e), "curs": C.vol_bates(pc, T, ks, r_e, q_e)}


def _date(d):
    s = str(d)
    return f"{s[8:10]}/{s[5:7]}/{s[:4]}"


def _duree(T):
    return f"{round(T * 12)} mois" if T < 1 else f"{T:.1f} ans"


def _applique_calibrage(curseurs):
    """Avant le rendu des widgets : la barre laterale passe sous Bates, reglee sur le marche."""
    st.session_state["modele"] = MODELE_BATES
    for nom, cle in CLES_BARRE.items():
        valeur = float(curseurs[nom])
        st.session_state[cle] = valeur
        # Venant de Black-Scholes, les curseurs Bates ne sont pas encore affiches :
        # Streamlit ne mettrait pas leur valeur dans le lien. On l'y ecrit ici. Deja
        # affiches, ils sont lies au lien et Streamlit le met a jour lui-meme.
        try:
            st.query_params[cle] = f"{valeur:g}"
        except StreamlitAPIException:
            pass
    st.session_state["labo_calibrage_applique"] = True


def _graphe_smile(d, titre, sous_titre):
    """
    Vols implicites de marche (points) contre smiles des modeles (courbes).
    Au survol d'un strike cote : toutes les vols a ce strike et l'ecart au
    marche, dans une seule infobulle.
    """
    atm = float(np.interp(1.0, d["ks"], d["vol"]))
    lignes = []
    for nom, y in ((NOMS_SMILE[1], d["cal_grille"]), (NOMS_SMILE[2], d["curs_grille"]),
                   (NOMS_SMILE[3], np.full(len(d["grille"]), atm))):
        lignes.append(pd.DataFrame({"x": d["grille"] * 100, "y": y * 100, "serie": nom}))
    lignes = pd.concat(lignes, ignore_index=True)
    lignes = lignes[np.isfinite(lignes["y"])]
    pts = pd.DataFrame({
        "idx": np.arange(len(d["ks"])), "x": d["ks"] * 100, "y": d["vol"] * 100,
        "strike": [f"{k * 100:.1f} % du spot ({K:,.0f} points)".replace(",", " ")
                   for k, K in zip(d["ks"], d["K"])],
        "type": ["call" if c else "put" for c in d["call"]],
        "marche": [G.pct(v * 100, 2) for v in d["vol"]],
        "bates": [G.pct(v * 100, 2) for v in d["cal"]],
        "ecart": [f"{(b - v) * 100:+.2f} pt" for b, v in zip(d["cal"], d["vol"])],
        "curseurs": [G.pct(v * 100, 2) for v in d["curs"]],
        "ecart_c": [f"{(b - v) * 100:+.2f} pt" for b, v in zip(d["curs"], d["vol"])],
        "bs": [f"{atm * 100:.2f} % ({(atm - v) * 100:+.2f} pt)" for v in d["vol"]],
        "oi": [f"{n:,} contrats".replace(",", " ") for n in d["oi"]]})

    toutes = np.concatenate([lignes["y"].to_numpy(), pts["y"].to_numpy()])
    marge = (toutes.max() - toutes.min()) * 0.08
    ech_x = alt.Scale(domain=[float(d["grille"][0] * 100), float(d["grille"][-1] * 100)],
                      nice=False, zero=False)
    ech_y = alt.Scale(domain=[float(toutes.min() - marge), float(toutes.max() + marge)],
                      nice=False, zero=False)
    x = alt.X("x:Q", scale=ech_x, axis=alt.Axis(title="Strike (% du spot)"))
    y = alt.Y("y:Q", scale=ech_y, axis=alt.Axis(title="Volatilite implicite (%)"))
    couleur = alt.Scale(domain=NOMS_SMILE, range=[G.ACCENT, G.PRODUIT, G.PRODUIT, G.DIRECT])
    trait = alt.Scale(domain=NOMS_SMILE, range=[[1, 0], [1, 0], G.TRAITS["pointille"],
                                                G.TRAITS["tirete"]])
    survol = alt.selection_point(name="survol", nearest=True, on="pointerover", fields=["idx"],
                                 empty=False, clear="pointerout")
    infobulle = [alt.Tooltip("strike:N", title="Strike"), alt.Tooltip("type:N", title="Option"),
                 alt.Tooltip("marche:N", title="Vol de marche"),
                 alt.Tooltip("bates:N", title="Bates calibre"),
                 alt.Tooltip("ecart:N", title="Ecart Bates - marche"),
                 alt.Tooltip("curseurs:N", title="Bates aux crans des curseurs"),
                 alt.Tooltip("ecart_c:N", title="Ecart curseurs - marche"),
                 alt.Tooltip("bs:N", title="Black-Scholes plat"),
                 alt.Tooltip("oi:N", title="Positions ouvertes")]
    spot = pd.DataFrame([{"x": 100.0, "y": float(toutes.min() - marge * 0.6), "txt": "spot"}])
    base = alt.Chart(pts)
    return alt.layer(
        alt.Chart(spot).mark_rule(strokeDash=[2, 3], strokeWidth=1.4, color=G.NIVEAU).encode(
            alt.X("x:Q", scale=ech_x)),
        alt.Chart(spot).mark_text(align="left", baseline="bottom", dx=4, fontSize=11,
                                  fontWeight="bold", color=G.NIVEAU).encode(
            alt.X("x:Q", scale=ech_x), alt.Y("y:Q", scale=ech_y), text="txt:N"),
        # une couche par courbe : une echelle de taille nominale interpolerait les epaisseurs
        *[alt.Chart(lignes[lignes["serie"] == nom]).mark_line(strokeWidth=larg).encode(
            x, y, color=alt.Color("serie:N", scale=couleur,
                                  legend=G._legende() if nom == NOMS_SMILE[1] else None),
            strokeDash=alt.StrokeDash("serie:N", scale=trait, legend=None))
          for nom, larg in zip(NOMS_SMILE[1:], (2.8, 1.8, 1.6))],
        base.mark_point(filled=True, size=80, stroke="white", strokeWidth=1.2, opacity=1,
                        color=G.ACCENT).encode(x, y),
        base.mark_rule(color=G.NIVEAU, strokeWidth=1.2).encode(
            alt.X("x:Q", scale=ech_x), opacity=alt.condition(survol, alt.value(0.9), alt.value(0)),
            tooltip=infobulle).add_params(survol),
        base.mark_point(filled=True, size=190, stroke="white", strokeWidth=2, opacity=1,
                        color=G.ACCENT).encode(x, y).transform_filter(survol),
    ).properties(title=G._titre(titre, sous_titre), height=380)


def affiche(m):
    BATES, kappa, lam, marge, mu_j = m.BATES, m.kappa, m.lam, m.marge, m.mu_j
    n_paths, nominal, q, r, recovery = m.n_paths, m.nominal, m.q, m.r, m.recovery
    rho, sig_j, theta_v, vol_atm, xi = m.rho, m.sig_j, m.theta_v, m.vol_atm, m.xi

    st.title("Laboratoire modele", help="**Ce que tu regardes.** Le meme produit price par deux modeles differents : Black-Scholes a\nvolatilite constante, et Bates (volatilite stochastique + sauts).\n\n**Pourquoi c'est important.** Le choix du modele n'est pas un detail academique. Sur un BRC\na barriere 70%, il change le coupon de pres de 3 points. Black-Scholes sous-estime la\nprobabilite de toucher la barriere, donc sous-estime la valeur du put que vend le client,\ndonc affiche un coupon trop faible.\n\n**Les cinq blocs.**\n1. Le smile : Bates genere un skew realiste, Black-Scholes donne une droite plate.\n2. Les queues de distribution : Bates produit de vrais krachs.\n3. L'effet sur le prix du BRC.\n4. L'effet du spread emetteur sur le coupon.\n5. Le calibrage : les 8 parametres de Bates ajustes sur 122 prix d'options Eurex, et un\nbouton pour regler toute l'appli sur le marche.\n\n**A retenir.** Un coupon anormalement eleve doit faire poser deux questions : quel modele\nle desk utilise-t-il, et quelle est la signature de l'emetteur ?")
    st.caption("Pourquoi le choix du modele n'est pas un detail academique : "
               "il change le coupon de plusieurs points.")

    T = st.slider("Maturite (annees)", 0.5, 3.0, 1.0, 0.25,
                  key="labo_maturite", bind="query-params")
    bar = st.slider("Barriere du BRC de reference (%)", 50, 90, 70, 5,
                    key="labo_barriere", bind="query-params")

    ch_bs, s_bs = sim_bs(S0, T, r, q, vol_atm, n_paths)
    if BATES:
        ch_bt, s_bt = sim_bates(S0, T, r, q, n_paths, vol_atm ** 2, theta_v ** 2,
                                kappa, xi, rho, lam, mu_j, sig_j)
    else:
        ch_bt, s_bt = sim_bates(S0, T, r, q, n_paths, vol_atm ** 2, (vol_atm + 0.02) ** 2,
                                2.0, 0.5, -0.70, 0.3, -0.10, 0.15)
        st.warning("Le modele actif est Black-Scholes. Les colonnes Bates ci-dessous "
                   "utilisent des parametres typiques d'indice actions, a titre de "
                   "comparaison. Bascule le modele dans la barre laterale pour les regler.")

    st.subheader("1. Le smile genere par le modele")
    ks = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
    vols = M.vol_implicite_bates(ch_bt, T, r, q, ks)
    xs = [k * 100 for k in ks]
    ys = [v * 100 for v in vols]
    graphe = G.courbes(
        xs, [{"nom": "Volatilite implicite generee par Bates", "y": ys, "couleur": G.PRODUIT,
              "epaisseur": 2.8, "points": True},
             {"nom": "Volatilite unique de Black-Scholes", "y": [vol_atm * 100] * len(xs),
              "couleur": G.DIRECT, "trait": "tirete", "epaisseur": 1.6}],
        "Bates genere le skew, Black-Scholes l'ignore",
        "Les puts bas coutent plus cher en vol que les options a la monnaie",
        "Strike (% du spot)", "Volatilite implicite (%)", nom_x="Strike",
        fmt_x=lambda v: f"{v:.0f} % du spot", fmt_y=lambda v: G.pct(v, 1),
        x_domaine=(66, 124), ticks_x=xs,
        lignes_infobulle=[("Prime de skew sur Black-Scholes",
                           [G.pct(y - vol_atm * 100, 1, signe=True) for y in ys])],
        niveaux_v=[(100, "a la monnaie")],
        notes=[(xs[0], ys[0], f"{(vols[0]-vols[3])*100:+.1f} pts de skew", G.PRODUIT, 10, 4)],
        hauteur=340)
    G.affiche(graphe)

    pente = (vols[0] - vols[3]) * 100
    st.markdown(
        f"Pente 70 % / 100 % : **{pente:+.1f} points de vol**. "
        "Black-Scholes donne une droite plate — il price le put a 70 % au meme prix "
        "relatif que l'option a la monnaie, ce que le marche ne fait jamais.")

    st.subheader("2. Queues de distribution")
    st_bs = ch_bs[:, -1].astype(float)
    st_bt = ch_bt[:, -1].astype(float)
    c = st.columns(4)
    c[0].metric("P(S < 60) Black-Scholes", f"{(st_bs < 60).mean()*100:,.2f}%")
    c[1].metric("P(S < 60) Bates", f"{(st_bt < 60).mean()*100:,.2f}%")
    c[2].metric("1er centile BS", f"{np.percentile(st_bs, 1):,.1f}")
    c[3].metric("1er centile Bates", f"{np.percentile(st_bt, 1):,.1f}")

    bins = np.linspace(20, 200, 91)
    h_bs = np.histogram(st_bs, bins=bins)[0] / len(st_bs) * 100
    h_bt = np.histogram(st_bt, bins=bins)[0] / len(st_bt) * 100
    larg = bins[1] - bins[0]
    sous_bs = [(st_bs < b).mean() * 100 for b in bins[:-1]]
    sous_bt = [(st_bt < b).mean() * 100 for b in bins[:-1]]
    graphe = G.courbes(
        bins[:-1], [{"nom": "Black-Scholes", "y": h_bs, "couleur": G.DIRECT, "aire": True,
                     "epaisseur": 1.2, "interpolation": "step-after"},
                    {"nom": "Bates", "y": h_bt, "couleur": G.PRODUIT, "aire": True,
                     "epaisseur": 1.2}],
        "Bates produit des krachs, Black-Scholes presque pas",
        f"Sous la barriere : {(st_bs < bar).mean()*100:,.1f}% des scenarios en "
        f"Black-Scholes contre {(st_bt < bar).mean()*100:,.1f}% en Bates",
        "Sous-jacent a l'echeance", "Part des scenarios (%)", nom_x="Sous-jacent entre",
        fmt_x=lambda v: f"{v:,.0f} et {v + larg:,.0f}", fmt_y=lambda v: G.pct(v, 2),
        x_domaine=(20, 200), y_domaine=(0, max(h_bs.max(), h_bt.max()) * 1.1),
        zone_x=(20, bar, G.FOND_PERTE), niveaux_v=[(bar, f"barriere  {bar:,.0f}", G.PERTE)],
        lignes_infobulle=[("Black-Scholes : finir sous ce niveau", [G.pct(v, 1) for v in sous_bs]),
                          ("Bates : finir sous ce niveau", [G.pct(v, 1) for v in sous_bt])],
        hauteur=340)
    G.affiche(graphe)

    st.subheader("3. Effet sur le prix")
    lignes = []
    for nom, ch_, s_ in [("Black-Scholes, vol plate", ch_bs, vol_atm),
                         ("Bates (vol stochastique + sauts)", ch_bt, s_bt)]:
        pn = M.proba_non_touche(ch_, bar / 100 * S0, s_, T)
        cp = M.coupon_brc(ch_, T, r, nominal, bar / 100, 1.0, True, marge, s_)
        lignes.append({"Modele": nom,
                       "Proba de toucher": f"{(1-pn.mean())*100:,.1f}%",
                       "Coupon equitable": f"{cp*100:,.2f}% p.a."})
    st.table(lignes)

    st.error(
        "**C'est le point le plus important de cette appli.** Black-Scholes sous-estime "
        "la probabilite de franchir la barriere, donc sous-estime la valeur du put que le "
        "client vend, donc affiche un coupon trop faible. Un desk qui pricerait en "
        "Black-Scholes plat vendrait systematiquement trop cher pour le client et se ferait "
        "arbitrer par la concurrence.\n\n"
        "Inversement, un Sales qui recoit un coupon nettement au-dessus du marche doit se "
        "demander ce que le modele du desk suppose, ou quel est le spread de l'emetteur.")

    st.subheader("4. Spread emetteur")
    lignes = []
    for sp in [0, 50, 100, 200, 400]:
        tau = M.tire_defaut(len(ch_bt), T, sp, recovery)
        cp = M.coupon_brc(ch_bt, T, r, nominal, bar / 100, 1.0, True, marge, s_bt,
                          tau, recovery)
        lignes.append({"Spread emetteur": f"{sp} bp",
                       "Proba de defaut sur la periode": f"{(tau <= T).mean()*100:,.1f}%",
                       "Coupon": f"{cp*100:,.2f}% p.a."})
    st.table(lignes)
    st.info(
        "**Le contre-intuitif a retenir** : le spread emetteur augmente le coupon. La "
        "banque se finance moins cher via le produit structure que sur le marche "
        "obligataire, et reverse l'economie au client.\n\n"
        "**Ce que ca implique en clientele** : comparer deux coupons de deux emetteurs "
        "differents sans regarder leur signature n'a aucun sens. En 2008, les porteurs de "
        "produits Lehman ont recupere environ 9 cents par dollar, barrieres intactes ou non.")

    st.subheader("5. Calibrage sur le marche")
    cal, _ = _surface()
    echeances = [e["echeance"] for e in cal["par_echeance"]]
    par_ech = {e["echeance"]: e for e in cal["par_echeance"]}
    ech = st.selectbox(
        "Echeance Eurex", echeances, index=next(i for i, e in enumerate(cal["par_echeance"])
                                                if e["maturite"] >= 1.0),
        format_func=lambda e: f"{_date(e)} ({_duree(par_ech[e]['maturite'])})",
        help=f"Options europeennes sur l'Euro Stoxx 50 (Eurex OESX), prix de reglement du "
             f"{_date(cal['date_marche'])}, de 1 mois a 5 ans. Un seul jeu de 8 parametres "
             "doit coller a toutes les echeances a la fois.",
        key="labo_echeance", bind="query-params")
    d = _smiles(ech)
    e = par_ech[ech]
    ecarts = (d["cal"] - d["vol"]) * 100
    atm = float(np.interp(1.0, d["ks"], d["vol"]))
    k_bas = d["ks"][0]
    G.affiche(_graphe_smile(
        d, f"Bates calibre colle au marche a {e['moyenne_quadratique']:.2f} pt de vol pres",
        f"Echeance du {_date(ech)} : le put {k_bas * 100:.0f} % cote {d['vol'][0] * 100:.1f} % "
        f"de vol, l'option a la monnaie {atm * 100:.1f} %. Black-Scholes plat ignore ces "
        f"{(d['vol'][0] - atm) * 100:.0f} points de skew."))
    st.caption(f"Taux et dividende implicites de cette echeance (parite call-put Eurex) : "
               f"{d['r'] * 100:.2f} % et {d['q'] * 100:.2f} % par an.")

    erreur = cal["erreur_pts_vol"]
    c = st.columns(4)
    c[0].metric("Erreur, cette echeance", f"{e['moyenne_quadratique']:.2f} pt",
                help="Moyenne quadratique des ecarts de vol implicite modele - marche, en points "
                     "de vol, sur les strikes cotes de cette echeance.")
    c[1].metric("Pire ecart, cette echeance", f"{np.nanmax(np.abs(ecarts)):.2f} pt",
                help="Le strike le plus mal reproduit. Les ailes courtes sont les plus dures.")
    c[2].metric("Erreur, toute la surface", f"{erreur['moyenne_quadratique']:.2f} pt",
                f"{cal['n_options']} options, {len(echeances)} echeances", delta_color="off",
                delta_arrow="off",
                help="Un modele a 8 parametres constants ne colle jamais parfaitement de 1 mois "
                     "a 5 ans : la taille de l'ecart residuel est l'information.")
    c[3].metric("Aux crans des curseurs", f"{erreur['moyenne_quadratique_curseurs']:.2f} pt",
                f"cette echeance : {e['moyenne_quadratique_curseurs']:.2f} pt",
                delta_color="off", delta_arrow="off",
                help="Erreur avec les valeurs que le bouton applique a la barre laterale, "
                     "arrondies aux crans de chaque curseur puis recalibrees sur cette grille.")

    p, cu = cal["parametres"], cal["curseurs"]
    st.table([
        {"Parametre": "Vol de depart", "Calibre": G.pct(math.sqrt(p["v0"]) * 100),
         "Curseur": G.pct(cu["vol_atm"], 0),
         "Ce que dit le marche": "la vol courte du jour, " + (
             "sous la vol longue : marche calme" if p["v0"] < p["theta"]
             else "au-dessus de la vol longue : marche tendu")},
        {"Parametre": "Vol long terme", "Calibre": G.pct(math.sqrt(p["theta"]) * 100),
         "Curseur": G.pct(cu["theta_v"], 0),
         "Ce que dit le marche": "le niveau vers lequel le marche attend que la vol revienne"},
        {"Parametre": "Vitesse de retour a la moyenne", "Calibre": f"{p['kappa']:.2f}",
         "Curseur": f"{cu['kappa']:.1f}",
         "Ce que dit le marche": f"un choc de vol s'efface aux deux tiers en "
                                 f"{1 / p['kappa']:.1f} an"},
        {"Parametre": "Vol de la vol", "Calibre": f"{p['xi']:.2f}", "Curseur": f"{cu['xi']:.2f}",
         "Ce que dit le marche": "la courbure du smile : les ailes se paient cher"},
        {"Parametre": "Correlation spot / vol", "Calibre": f"{p['rho']:+.2f}",
         "Curseur": f"{cu['rho']:+.2f}",
         "Ce que dit le marche": "le skew : la vol monte quand le marche baisse"},
        {"Parametre": "Frequence des sauts", "Calibre": f"{p['lam']:.3f} par an",
         "Curseur": f"{cu['lam']:.1f} par an",
         "Ce que dit le marche": f"un krach tous les {1 / p['lam']:.0f} ans environ"},
        {"Parametre": "Taille moyenne d'un saut", "Calibre": G.pct(p["mu_j"] * 100, 0),
         "Curseur": G.pct(cu["mu_j"], 0),
         "Ce que dit le marche": "rare mais violent" + (
             f" : bute sur la borne {C.BORNES['mu_j'][0] * 100:.0f} % du curseur"
             if p["mu_j"] <= C.BORNES["mu_j"][0] + 1e-6 else "")},
        {"Parametre": "Dispersion des sauts", "Calibre": G.pct(p["sig_j"] * 100, 0),
         "Curseur": G.pct(cu["sig_j"], 0),
         "Ce que dit le marche": "l'incertitude sur la taille du krach"},
    ])
    st.caption(
        f"Calibrage maison : formule fermee de Bates (Lewis) et optimiseur Nelder-Mead sur les "
        f"vols implicites. Les curseurs ne sont pas de simples arrondis : arrondir la frequence "
        f"de sauts a 0 effacerait le skew court, donc les autres parametres sont recalibres sur "
        f"les crans ({erreur['moyenne_quadratique_curseurs']:.2f} pt au lieu de "
        f"{erreur['moyenne_quadratique']:.2f}). Condition de Feller 2 kappa theta / xi^2 = "
        f"{cal['feller']:.2f} < 1 : la variance peut toucher zero, comme presque toujours sur "
        "indice actions.")

    st.button("Appliquer le calibrage", key="labo_appliquer", type="primary",
              on_click=_applique_calibrage, args=(cu,),
              help="Passe la barre laterale sous Bates avec ces valeurs aux crans des curseurs. "
                   "Toutes les pages pricent alors avec la surface Eurex du "
                   f"{_date(cal['date_marche'])}.")
    if st.session_state.pop("labo_calibrage_applique", False):
        st.success(
            f"Barre laterale sous Bates, reglee sur le marche du {_date(cal['date_marche'])} : "
            f"vol de depart {cu['vol_atm']:.0f} %, vol long terme {cu['theta_v']:.0f} %, "
            f"retour {cu['kappa']:.1f}, vol de la vol {cu['xi']:.2f}, correlation "
            f"{cu['rho']:+.2f}, {cu['lam']:.1f} saut par an de {cu['mu_j']:.0f} % "
            f"(dispersion {cu['sig_j']:.0f} %). Les blocs 1 a 4 ci-dessus utilisent deja ces "
            "valeurs.")
    st.info("**Pourquoi un desk recalibre chaque matin.** Les parametres de Bates ne sont pas "
            "des constantes de la nature mais les valeurs qui font coller le modele aux prix "
            "d'options du jour : si le marche bouge pendant la nuit et que le desk garde ceux "
            "d'hier, il price et couvre ses exotiques sur un marche qui n'existe plus.")
