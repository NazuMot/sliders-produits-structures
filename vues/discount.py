"""7 - Discount Certificate."""

import math
import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0


def affiche(m):
    BATES, bandeau_modele, marge, nominal, q, r = m.BATES, m.bandeau_modele, m.marge, m.nominal, m.q, m.r
    skew, spread, trajectoires, vol_atm = m.skew, m.spread, m.trajectoires, m.vol_atm

    st.title("Discount Certificate", help="**Ce que c'est.** Le client achète le sous-jacent moins cher que le marché (la décote),\nen échange d'un plafond : à l'échéance il reçoit le sous-jacent, au plus au niveau du cap.\n\n**Comment c'est construit.** Détention du sous-jacent sans ses dividendes, et vente par le\nclient d'un call de strike égal au cap. Deux sources financent la décote : les dividendes\nabandonnés et la prime du call vendu.\n\n**Pourquoi un client l'achète.** Vue neutre à modérément haussière. Il gagne si le marché\nmonte un peu, stagne, ou même baisse moins que la décote. Il renonce à la hausse au-delà\ndu cap. C'est l'alternative à l'achat direct pour un client qui serait de toute façon entre\ndans le titre.\n\n**Le levier de vente.** Plus la vol est haute, plus le call vendu vaut cher, plus la décote\nest forte. Un Discount se vend quand la vol est élevée.\n\n**Ce qu'il faut regarder.** Le point mort (le niveau sous lequel le client perd) contre la\nbande de densité : où le sous-jacent finit réellement. Et le second graphe : un cap plus\nbas achète de la protection et coûte du potentiel.")
    st.caption("Le sous-jacent moins cher que le marché, avec un plafond.")
    G.en_30_secondes(
        "Le sous-jacent acheté moins cher que le marché : il gagne s'il monte un peu, s'il "
        "stagne, et même s'il baisse moins que la remise.",
        "Le gain est plafonné au niveau du cap. Sous le point mort, le client perd comme le "
        "sous-jacent (un peu moins, grâce à la remise).",
        "Un client qui voulait de toute façon acheter le titre, et pense qu'il ne va pas "
        "beaucoup monter.")
    bandeau_modele()

    c1, c2 = st.columns(2)
    T = c1.slider("Maturité (années)", 0.25, 3.0, 1.0, 0.25,
                  help="Plus c'est long, plus le call vendu et les dividendes abandonnés "
                       "valent cher : la décote grandit, le rendement maximal aussi.",
                  key="dc_maturite", bind="query-params")
    cap = c2.slider("Cap (% du spot)", 70, 130, 100, 5,
                    help="Plafond du remboursement, et strike du call vendu par le client. "
                         "Cap bas : grosse décote, faible rendement maximal. Cap haut : "
                         "petite décote, presque une détention directe.",
                    key="dc_cap", bind="query-params")

    K = cap / 100 * S0
    ch, _ = trajectoires(T, cap / 100)
    ST = ch[:, -1].astype(np.float64)
    if BATES:
        offre = M.discount_mc(ch, T, r, cap / 100, marge)
        methode = f"Monte Carlo sous Bates, {len(ST):,} trajectoires".replace(",", " ")
    else:
        sigma = M.vol_au_strike(cap / 100, vol_atm, skew)
        offre = M.discount_certificate(S0, K, T, r, sigma, q, marge)
        methode = f"Formule fermée, vol au cap {sigma*100:,.1f} %"
    prix, decote = offre["prix"], offre["decote"]
    rm, rm_pa = offre["rendement_max"], offre["rendement_max_pa"]
    p_perte = float((ST < prix).mean())
    p_cap = float((ST >= K).mean())

    mt = st.columns(4)
    mt[0].metric("Décote", f"{decote*100:,.2f}%",
                 help=f"Le client paie {prix:,.2f} un sous-jacent qui cote {S0:,.0f}.")
    mt[1].metric("Rendement maximal", f"{rm*100:,.2f}%",
                 help=f"Soit {rm_pa*100:,.2f} % par an, si le sous-jacent finit au-dessus "
                      f"du cap {K:,.0f}.")
    mt[2].metric("Point mort", f"{prix:,.2f}",
                 help="Niveau du sous-jacent à l'échéance (départ 100) sous lequel le client "
                      "perd de l'argent : c'est le prix qu'il a payé.")
    mt[3].metric("Proba de perte en capital", f"{p_perte*100:,.1f}%")
    st.caption(f"Rendement maximal {rm_pa*100:,.2f} % par an, atteint si le sous-jacent finit "
               f"au-dessus de {K:,.0f} : {p_cap*100:,.0f} % des scénarios. {methode}"
               + (", hors risque émetteur." if spread else "."))

    # --- graphe 1 : profil de remboursement
    spots = np.linspace(30, 170, 500)
    unites = nominal / prix                 # parts achetees avec le nominal
    produit = unites * np.minimum(spots, K)
    direct = nominal * spots / S0
    maxi = unites * K
    seuil_direct = K * S0 / prix            # au-dela, la detention directe fait mieux
    graphe = G.payoff(
        spots, produit, direct, nominal, "Discount Certificate",
        titre=f"Le client gagne tant que le sous-jacent finit au-dessus de {prix:,.1f}",
        sous_titre=f"Gain plafonné à {rm*100:,.1f} % au-dessus du cap {K:,.0f} ; au-delà de "
                   f"{seuil_direct:,.0f}, la détention directe fait mieux",
        niveaux=[(prix, f"point mort  {prix:,.1f}")],
        label_direct="S'il avait acheté le sous-jacent",
        echantillon=ST, seuil_densite=prix,
        niveaux_h=[(maxi, f"maximum  {maxi:,.0f}", G.GAIN)],
        notes=[(K, maxi, f"cap  {K:,.0f}", G.PRODUIT, 6, -6)],
        montrer_points_morts=False, legende_zones=True)
    G.affiche(graphe)
    G.en_clair(
        f"Pour {G.eur(nominal)} investis sur {G.duree(T)} : le client achète le sous-jacent à "
        f"{prix:,.1f} au lieu de {S0:,.0f}. Il gagne tant qu'il finit au-dessus de {prix:,.1f}, "
        f"au maximum {G.eur(maxi)} s'il finit au-dessus du cap {K:,.0f}. S'il baisse de 20 %, "
        f"le client récupère {G.eur(unites * min(0.8 * S0, K))}, contre {G.eur(nominal * 0.8)} "
        "en direct.")

    # --- graphe 2 : le choix du cap
    caps = np.arange(70, 131, 1)
    if BATES:
        offres = [M.discount_mc(ch, T, r, c / 100, marge) for c in caps]
    else:
        offres = [M.discount_certificate(S0, float(c), T, r,
                                         M.vol_au_strike(c / 100, vol_atm, skew), q, marge)
                  for c in caps]
    decotes = np.array([o["decote"] * 100 for o in offres])
    maxis = np.array([o["rendement_max"] * 100 for o in offres])
    pertes = np.array([(ST < o["prix"]).mean() * 100 for o in offres])
    graphe = G.courbes(
        caps, [{"nom": "Rendement maximal", "y": maxis, "couleur": G.PRODUIT},
               {"nom": "Décote", "y": decotes, "couleur": G.ACCENT},
               {"nom": "Proba de perte en capital", "y": pertes, "couleur": G.NIVEAU,
                "trait": "pointille", "epaisseur": 2.0}],
        "Un cap plus bas achète de la protection et coûte du potentiel",
        f"Cap {cap} : décote {decote*100:,.1f} %, rendement maximal {rm*100:,.1f} %, "
        f"proba de perte {p_perte*100:,.0f} %",
        "Cap (% du spot)", "%", nom_x="Cap", fmt_x=lambda v: f"{v:,.0f} %",
        fmt_y=lambda v: G.pct(v, 1), zero=True,
        niveaux_v=[(cap, f"cap choisi  {cap}")],
        marqueurs=[(cap, rm * 100, G.PRODUIT), (cap, decote * 100, G.ACCENT),
                   (cap, p_perte * 100, G.NIVEAU)])
    G.affiche(graphe)

    if st.checkbox("Décomposer la décote", key="dc_voir_decomposition", bind="query-params",
                   help="D'où viennent les points de remise : dividendes abandonnés, prime du "
                        "call vendu, moins la marge de la banque."):
        dividendes = S0 * (1.0 - math.exp(-q * T))
        call = S0 * math.exp(-q * T) - offre["valeur"]
        frais = prix - offre["valeur"]
        st.table([
            {"Composante": "Dividendes abandonnés par le client",
             "Points de % du spot": f"{dividendes:+,.2f}",
             "EUR sur le nominal": G.eur(dividendes / S0 * nominal, signe=True)},
            {"Composante": f"Prime du call de strike {K:,.0f} vendu par le client",
             "Points de % du spot": f"{call:+,.2f}",
             "EUR sur le nominal": G.eur(call / S0 * nominal, signe=True)},
            {"Composante": "Marge de la banque",
             "Points de % du spot": f"{-frais:+,.2f}",
             "EUR sur le nominal": G.eur(-frais / S0 * nominal, signe=True)},
            {"Composante": "Décote offerte au client",
             "Points de % du spot": f"{S0 - prix:+,.2f}",
             "EUR sur le nominal": G.eur((S0 - prix) / S0 * nominal, signe=True)},
        ])
        st.caption("Le client ne reçoit pas un cadeau : il revend à la banque les dividendes "
                   "et la hausse au-delà du cap. Plus la vol est haute, plus le call vaut cher.")

    st.info(
        f"**Le discours client** : « Vous achetez le sous-jacent avec {decote*100:,.1f} % de "
        f"remise. S'il ne baisse pas de plus de {decote*100:,.1f} % d'ici "
        f"{T:g} an{'s' if T > 1 else ''}, vous ne perdez rien ; s'il finit au-dessus de "
        f"{K:,.0f}, vous gagnez {rm*100:,.1f} %, soit {rm_pa*100:,.1f} % par an. En échange, "
        "vous renoncez à la hausse au-delà du cap et aux dividendes. »\n\n"
        "**Décomposition** : détention du sous-jacent sans ses dividendes, moins un call vendu "
        "au niveau du cap. Les dividendes et la prime du call financent la décote.\n\n"
        "**À ne jamais laisser passer** : sous le point mort, le client perd comme un "
        "actionnaire, simplement un peu moins. La décote amortit la baisse, elle ne protège "
        "pas le capital.")
