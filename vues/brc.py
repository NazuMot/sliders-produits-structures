"""3 - Barrier Reverse Convertible."""

import numpy as np
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0


def affiche(m):
    bandeau_modele, defauts, marge, nominal, r = m.bandeau_modele, m.defauts, m.marge, m.nominal, m.r
    recovery, spread, trajectoires = m.recovery, m.spread, m.trajectoires

    st.title("Barrier Reverse Convertible", help="**Ce que c'est.** Un coupon garanti eleve, et le capital rembourse *sauf* si le sous-jacent\nfranchit la barriere ET finit sous le strike.\n\n**Comment c'est construit.** Une obligation, plus la vente par le client d'un put\ndown-and-in a la banque.\n\n**Le point essentiel.** Le coupon n'est pas un interet. C'est la prime que la banque paie au\nclient pour l'assurer contre la baisse. Coupon eleve = sinistre probable, jamais bonne\naffaire.\n\n**Pourquoi un client l'achete.** Vue neutre : il pense que le sous-jacent ne va pas\nbeaucoup bouger et se fait payer pour ca.\n\n**Ce qu'il faut regarder.** Le graphe des trajectoires : deux chemins peuvent finir au meme\nniveau avec des resultats differents. Le chemin compte, pas seulement l'arrivee.")
    st.caption("Le client ne recoit pas un interet. Il encaisse une prime d'assurance.")
    bandeau_modele()

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 0.25, 3.0, 1.0, 0.25,
                  key="brc_maturite", bind="query-params")
    bar = c2.slider("Barriere (% du spot)", 50, 95, 70, 5,
                    help="Le niveau qui ARME le risque. Le toucher ne fait pas perdre : "
                         "c'est le strike qui decide a la fin.",
                         key="brc_barriere", bind="query-params")
    strike = c3.slider("Strike (% du spot)", 80, 110, 100, 5,
                       key="brc_strike", bind="query-params")

    continu = st.radio("Observation de la barriere",
                       ["Continue (americaine)", "A l'echeance (europeenne)"],
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
    m[0].metric("Coupon equitable", f"{cpn*100:,.2f}% p.a.")
    m[1].metric("Proba de toucher", f"{(1-p_no.mean())*100:,.1f}%")
    m[2].metric("Proba de perte en capital",
                f"{((1-p_no)*(ST < strike/100*S0)).mean()*100:,.1f}%")
    m[3].metric("Proba de defaut emetteur",
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
        spots, pay_touche, direct, nominal, "Si la barriere a ete touchee",
        titre=f"Coupon de {cpn*100:,.2f}% quoi qu'il arrive, capital a risque sous "
              f"{strike/100*S0:,.0f}",
        sous_titre="Le coupon est plafonne : meme si le sous-jacent double, le client "
                   "touche le meme montant",
        niveaux=[(bar / 100 * S0, f"barriere  {bar/100*S0:,.0f}"),
                 (strike / 100 * S0, f"strike  {strike/100*S0:,.0f}")],
        label_direct="S'il avait achete le sous-jacent",
        autres=[{"nom": "Si la barriere n'a jamais ete touchee", "y": pay_intact,
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
            tt, [{"nom": f"Client A — barriere intacte, recoit {plafond:,.0f}",
                  "y": ch[i_ok], "couleur": G.GAIN, "epaisseur": 2.4},
                 {"nom": f"Client B — barriere touchee, recoit {pay_ko:,.0f}",
                  "y": ch[i_ko], "couleur": G.PERTE, "epaisseur": 2.4}],
            "Deux clients, le meme niveau final, deux resultats",
            "Le chemin decide, pas seulement l'arrivee",
            "Temps (annees)", "Niveau du sous-jacent", nom_x="Date",
            fmt_x=lambda v: f"{v:,.2f} an", fmt_y=lambda v: G.niveau(v, S0),
            x_domaine=(0, T * 1.02), y_domaine=(bas * 0.90, haut * 1.10),
            zone_y=(bas * 0.90, bar / 100 * S0, G.FOND_PERTE),
            niveaux_h=[(bar / 100 * S0, f"barriere  {bar/100*S0:,.0f}", G.PERTE),
                       (S0, f"depart  {S0:,.0f}", G.NIVEAU)])
        G.affiche(graphe)

    # --- graphe 3 : le faisceau complet
    couleurs = [G.PERTE if touche[i] else G.GAIN for i in range(45)]
    statuts = [f"barriere touchee : rembourse {min(nominal * ST[i] / (strike / 100 * S0), nominal) + cpn * nominal * T:,.0f}"
               if touche[i] else f"barriere intacte : rembourse {plafond:,.0f}"
               for i in range(45)]
    graphe = G.trajectoires(
        ch, T, [(bar / 100 * S0, f"barriere  {bar/100*S0:,.0f}", G.PERTE),
                (S0, f"depart  {S0:,.0f}", G.NIVEAU)],
        couleurs,
        f"{(1-p_no.mean())*100:,.0f}% des trajectoires franchissent la barriere",
        "Survole une trajectoire pour voir ce que touche le client.",
        zone_sous=bar / 100 * S0, statuts=statuts,
        legende=[("Barriere touchee", G.PERTE), ("Barriere intacte", G.GAIN)])
    G.affiche(graphe)

    if st.checkbox("Decomposer le coupon", key="brc_voir_decomposition", bind="query-params"):
        base = M.coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu,
                            marge, sg, None, recovery)
        st.table([
            {"Composante": "Coupon hors risque emetteur", "Valeur": f"{base*100:,.2f}% p.a."},
            {"Composante": "Apport du spread emetteur", "Valeur": f"{(cpn-base)*100:+,.2f}% p.a."},
            {"Composante": "Coupon offert au client", "Valeur": f"{cpn*100:,.2f}% p.a."},
        ])
        st.caption("Le spread emetteur n'est pas un cadeau : c'est la remuneration d'un "
                   "risque de credit que le client porte sans toujours le savoir.")

    if st.checkbox("Risque emetteur lie au sous-jacent (wrong-way risk)",
                   key="brc_voir_wrong_way", bind="query-params",
                   help="Un BRC sur une action bancaire, emis par une banque comparable : "
                        "quand le sous-jacent s'effondre, l'emetteur vacille en meme temps. "
                        "Le modele de base suppose les deux independants."):
        sp = spread if spread > 0 else 150
        if spread == 0:
            st.caption("Aucun spread emetteur dans la barre laterale : la demonstration "
                       "utilise 150 bp, la signature d'une banque moyenne en periode "
                       "tendue.")
        rho_ww = st.slider(
            "Correlation defaut / sous-jacent", 0.0, 0.9, 0.6, 0.1,
            key="brc_ww_correlation", bind="query-params",
            help="0 : le defaut de l'emetteur ne depend pas du marche (modele de base). "
                 "Plus elle monte, plus le defaut tombe dans les scenarios ou le "
                 "sous-jacent a chute.")
        lignes, res = [], []
        for nom, c in (("Defaut independant", 0.0), ("Defaut lie au sous-jacent", rho_ww)):
            tau_c = M.tire_defaut(len(ch), T, sp, recovery, correlation=c, chemins=ch)
            cpn_c = M.coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu, marge,
                                 sg, tau_c, recovery, creance="valeur")
            prof = M.profil_brc(ch, T, nominal, bar / 100, strike / 100, continu, cpn_c, sg,
                                tau_c, recovery, creance="valeur", niveau_es=0.01)
            res.append(prof)
            lignes.append({"": nom, "Coupon equitable": f"{cpn_c*100:,.2f}% p.a.",
                           "Proba de defaut": f"{prof['p_defaut']*100:,.1f}%",
                           "Krach ET defaut": f"{prof['p_double']*100:,.2f}%",
                           "Perte moyenne dans le pire 1 %": f"{prof['es']*100:,.0f}% du nominal"})
        st.table(lignes)
        hyp = ["Defaut independant", f"Defaut lie ({rho_ww:.1f})"]
        g1, g2 = st.columns(2)
        with g1:
            G.affiche(G.barres(
                hyp, [res[0]["p_double"] * 100, res[1]["p_double"] * 100],
                [G.DIRECT, G.PERTE], "Krach ET defaut en meme temps",
                "Probabilite que le client perde sur les deux tableaux", "Probabilite (%)",
                nom_x="Hypothese", fmt=lambda v: G.pct(v, 2), hauteur=260, seuil_texte=0))
        with g2:
            G.affiche(G.barres(
                hyp, [res[0]["es"] * 100, res[1]["es"] * 100],
                [G.DIRECT, G.PERTE], "Perte moyenne dans le pire 1 % des cas",
                "Le coupon bouge a peine, la queue des pertes s'alourdit",
                "% du nominal", nom_x="Hypothese", fmt=lambda v: G.pct(v, 0), hauteur=260,
                seuil_texte=0))
        st.info(
            "**Ce que ca change pour le client** : le prix moyen bouge peu, parce que la "
            "perte supplementaire est concentree dans quelques scenarios. Mais dans ces "
            "scenarios, le client perd deux fois au meme moment : sur le sous-jacent, puis "
            "sur la creance envers l'emetteur.\n\n"
            "**Le reflexe Sales** : ne jamais placer un BRC sur une action bancaire emis par "
            "une banque du meme pays ou du meme modele economique. Choisir un emetteur dont "
            "le risque n'est pas lie au sous-jacent, quitte a perdre un peu de coupon.")

    st.info(
        "**Le coupon a trois sources** : les taux, la prime du put que le client vend, et "
        "le spread de l'emetteur. Coupon eleve = sinistre probable, ou signature faible.\n\n"
        "**Pres de la barriere, le gamma explose** : le trader doit trader des volumes "
        "enormes en peu de temps. D'ou le prix degrade sur les barrieres proches du spot. "
        "La page Cout de couverture chiffre ce mecanisme.")
