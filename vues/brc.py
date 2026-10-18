"""3 - Barrier Reverse Convertible."""

import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0


def affiche(m):
    bandeau_modele, defauts, marge, nominal, r = m.bandeau_modele, m.defauts, m.marge, m.nominal, m.r
    recovery, spread, trajectoires = m.recovery, m.spread, m.trajectoires

    st.title("Barrier Reverse Convertible", help="**Ce que c'est.** Un coupon garanti élevé, et le capital remboursé *sauf* si le sous-jacent\nfranchit la barrière ET finit sous le strike.\n\n**Comment c'est construit.** Une obligation, plus la vente par le client d'un put\ndown-and-in à la banque.\n\n**Le point essentiel.** Le coupon n'est pas un intérêt. C'est la prime que la banque paie au\nclient pour l'assurer contre la baisse. Coupon élevé = sinistre probable, jamais bonne\naffaire.\n\n**Pourquoi un client l'achète.** Vue neutre : il pense que le sous-jacent ne va pas\nbeaucoup bouger et se fait payer pour ça.\n\n**Ce qu'il faut regarder.** Le graphe des trajectoires : deux chemins peuvent finir au même\nniveau avec des résultats différents. Le chemin compte, pas seulement l'arrivée.")
    st.caption("Le client ne reçoit pas un intérêt. Il encaisse une prime d'assurance.")
    bandeau_modele()

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturité (années)", 0.25, 3.0, 1.0, 0.25,
                  key="brc_maturite", bind="query-params")
    bar = c2.slider("Barrière (% du spot)", 50, 95, 70, 5,
                    help="Le niveau qui ARME le risque. Le toucher ne fait pas perdre : "
                         "c'est le strike qui décide à la fin.",
                         key="brc_barriere", bind="query-params")
    strike = c3.slider("Strike (% du spot)", 80, 110, 100, 5,
                       key="brc_strike", bind="query-params")

    continu = st.radio("Observation de la barrière",
                       ["Continue (américaine)", "À l'échéance (européenne)"],
                       horizontal=True,
                       key="brc_observation", bind="query-params").startswith("Continue")

    ch, sg = trajectoires(T, bar / 100)
    tau = defauts(len(ch), T)
    cpn = M.coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu, marge,
                       sg, tau, recovery)

    ST = ch[:, -1].astype(np.float64)
    p_no = (M.proba_non_touche(ch, bar / 100 * S0, sg, T) if continu
            else (ST > bar / 100 * S0).astype(np.float64))

    m = st.columns(4)
    m[0].metric("Coupon équitable", f"{cpn*100:,.2f}% p.a.")
    m[1].metric("Proba de toucher", f"{(1-p_no.mean())*100:,.1f}%")
    m[2].metric("Proba de perte en capital",
                f"{((1-p_no)*(ST < strike/100*S0)).mean()*100:,.1f}%")
    m[3].metric("Proba de défaut émetteur",
                f"{0.0 if tau is None else (tau <= T).mean()*100:,.1f}%")

    rng = np.random.default_rng(3)
    touche = rng.random(len(p_no)) > p_no

    # --- graphe 1 : le payoff theorique du BRC
    spots = np.linspace(30, 190, 500)
    plafond = nominal + cpn * nominal * T
    pay_intact = np.full_like(spots, plafond)
    pay_touche = np.where(spots >= strike / 100 * S0, plafond,
                          nominal * spots / (strike / 100 * S0) + cpn * nominal * T)
    direct = nominal * spots / S0

    graphe = G.payoff(
        spots, pay_touche, direct, nominal, "Si la barrière a été touchée",
        titre=f"Coupon de {cpn*100:,.2f}% quoi qu'il arrive, capital à risqué sous "
              f"{strike/100*S0:,.0f}",
        sous_titre="Le coupon est plafonné : même si le sous-jacent double, le client "
                   "touche le même montant",
        niveaux=[(bar / 100 * S0, f"barrière  {bar/100*S0:,.0f}"),
                 (strike / 100 * S0, f"strike  {strike/100*S0:,.0f}")],
        label_direct="S'il avait acheté le sous-jacent",
        autres=[{"nom": "Si la barrière n'a jamais été touchée", "y": pay_intact,
                 "couleur": G.GAIN, "epaisseur": 2.4}],
        niveaux_h=[(plafond, f"plafond  {plafond:,.0f}", G.GAIN)],
        legende_zones=True)
    G.affiche(graphe)

    # --- graphe 2 : deux clients, meme niveau final, resultats differents
    cible = 0.88 * S0
    cand_i = np.where((np.abs(ST - cible) < 1.2) & ~touche)[0]
    cand_t = np.where((np.abs(ST - cible) < 1.2) & touche)[0]
    if len(cand_i) and len(cand_t):
        tt = np.linspace(0, T, ch.shape[1])
        i_ok = cand_i[np.argmax(ch[cand_i].min(axis=1))]
        i_ko = cand_t[np.argmin(ch[cand_t].min(axis=1))]
        pay_ko = nominal * ST[i_ko] / (strike / 100 * S0) + cpn * nominal * T
        bas = min(ch[i_ok].min(), ch[i_ko].min(), bar / 100 * S0)
        haut = max(ch[i_ok].max(), ch[i_ko].max(), S0)
        graphe = G.courbes(
            tt, [{"nom": f"Client A — barrière intacte, reçoit {plafond:,.0f}",
                  "y": ch[i_ok], "couleur": G.GAIN, "epaisseur": 2.4},
                 {"nom": f"Client B — barrière touchée, reçoit {pay_ko:,.0f}",
                  "y": ch[i_ko], "couleur": G.PERTE, "epaisseur": 2.4}],
            "Deux clients, le même niveau final, deux résultats",
            "Le chemin décide, pas seulement l'arrivée",
            "Temps (années)", "Niveau du sous-jacent", nom_x="Date",
            fmt_x=lambda v: f"{v:,.2f} an", fmt_y=lambda v: G.niveau(v, S0),
            x_domaine=(0, T * 1.02), y_domaine=(bas * 0.90, haut * 1.10),
            zone_y=(bas * 0.90, bar / 100 * S0, G.FOND_PERTE),
            niveaux_h=[(bar / 100 * S0, f"barrière  {bar/100*S0:,.0f}", G.PERTE),
                       (S0, f"départ  {S0:,.0f}", G.NIVEAU)])
        G.affiche(graphe)

    # --- graphe 3 : le faisceau complet
    couleurs = [G.PERTE if touche[i] else G.GAIN for i in range(45)]
    statuts = [f"barrière touchée : rembourse {min(nominal * ST[i] / (strike / 100 * S0), nominal) + cpn * nominal * T:,.0f}"
               if touche[i] else f"barrière intacte : rembourse {plafond:,.0f}"
               for i in range(45)]
    graphe = G.trajectoires(
        ch, T, [(bar / 100 * S0, f"barrière  {bar/100*S0:,.0f}", G.PERTE),
                (S0, f"départ  {S0:,.0f}", G.NIVEAU)],
        couleurs,
        f"{(1-p_no.mean())*100:,.0f}% des trajectoires franchissent la barrière",
        "Survole une trajectoire pour voir ce que touche le client.",
        zone_sous=bar / 100 * S0, statuts=statuts,
        legende=[("Barrière touchée", G.PERTE), ("Barrière intacte", G.GAIN)])
    G.affiche(graphe)

    if st.checkbox("Décomposer le coupon", key="brc_voir_decomposition", bind="query-params"):
        base = M.coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu,
                            marge, sg, None, recovery)
        st.table([
            {"Composante": "Coupon hors risque émetteur", "Valeur": f"{base*100:,.2f}% p.a."},
            {"Composante": "Apport du spread émetteur", "Valeur": f"{(cpn-base)*100:+,.2f}% p.a."},
            {"Composante": "Coupon offert au client", "Valeur": f"{cpn*100:,.2f}% p.a."},
        ])
        st.caption("Le spread émetteur n'est pas un cadeau : c'est la rémunération d'un "
                   "risque de crédit que le client porte sans toujours le savoir.")

    if st.checkbox("Risque émetteur lie au sous-jacent (wrong-way risk)",
                   key="brc_voir_wrong_way", bind="query-params",
                   help="Un BRC sur une action bancaire, émis par une banque comparable : "
                        "quand le sous-jacent s'effondre, l'émetteur vacille en même temps. "
                        "Le modèle de base suppose les deux indépendants."):
        sp = spread if spread > 0 else 150
        if spread == 0:
            st.caption("Aucun spread émetteur dans la barre latérale : la démonstration "
                       "utilise 150 bp, la signature d'une banque moyenne en période "
                       "tendue.")
        rho_ww = st.slider(
            "Corrélation défaut / sous-jacent", 0.0, 0.9, 0.6, 0.1,
            key="brc_ww_correlation", bind="query-params",
            help="0 : le défaut de l'émetteur ne dépend pas du marché (modèle de base). "
                 "Plus elle monte, plus le défaut tombe dans les scénarios où le "
                 "sous-jacent a chuté.")
        lignes, res = [], []
        for nom, c in (("Défaut indépendant", 0.0), ("Défaut lié au sous-jacent", rho_ww)):
            tau_c = M.tire_defaut(len(ch), T, sp, recovery, correlation=c, chemins=ch)
            cpn_c = M.coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu, marge,
                                 sg, tau_c, recovery, creance="valeur")
            prof = M.profil_brc(ch, T, nominal, bar / 100, strike / 100, continu, cpn_c, sg,
                                tau_c, recovery, creance="valeur", niveau_es=0.01)
            res.append(prof)
            lignes.append({"": nom, "Coupon équitable": f"{cpn_c*100:,.2f}% p.a.",
                           "Proba de défaut": f"{prof['p_defaut']*100:,.1f}%",
                           "Krach ET défaut": f"{prof['p_double']*100:,.2f}%",
                           "Perte moyenne dans le pire 1 %": f"{prof['es']*100:,.0f}% du nominal"})
        st.table(lignes)
        hyp = ["Défaut indépendant", f"Défaut lié ({rho_ww:.1f})"]
        g1, g2 = st.columns(2)
        with g1:
            G.affiche(G.barres(
                hyp, [res[0]["p_double"] * 100, res[1]["p_double"] * 100],
                [G.DIRECT, G.PERTE], "Krach ET défaut en même temps",
                "Probabilité que le client perde sur les deux tableaux", "Probabilité (%)",
                nom_x="Hypothèse", fmt=lambda v: G.pct(v, 2), hauteur=260, seuil_texte=0))
        with g2:
            G.affiche(G.barres(
                hyp, [res[0]["es"] * 100, res[1]["es"] * 100],
                [G.DIRECT, G.PERTE], "Perte moyenne dans le pire 1 % des cas",
                "Le coupon bouge à peine, la queue des pertes s'alourdit",
                "% du nominal", nom_x="Hypothèse", fmt=lambda v: G.pct(v, 0), hauteur=260,
                seuil_texte=0))
        st.info(
            "**Ce que ça change pour le client** : le prix moyen bouge peu, parce que la "
            "perte supplémentaire est concentrée dans quelques scénarios. Mais dans ces "
            "scénarios, le client perd deux fois au même moment : sur le sous-jacent, puis "
            "sur la créance envers l'émetteur.\n\n"
            "**Le réflexe Sales** : ne jamais placer un BRC sur une action bancaire émis par "
            "une banque du même pays ou du même modèle économique. Choisir un émetteur dont "
            "le risque n'est pas lié au sous-jacent, quitte à perdre un peu de coupon.")

    st.info(
        "**Le coupon a trois sources** : les taux, la prime du put que le client vend, et "
        "le spread de l'émetteur. Coupon élevé = sinistre probable, ou signature faible.\n\n"
        "**Pres de la barrière, le gamma explose** : le trader doit trader des volumes "
        "énormes en peu de temps. D'où le prix dégradé sur les barrières proches du spot. "
        "La page Coût de couverture chiffre ce mécanisme.")
