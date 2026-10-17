"""3 - Barrier Reverse Convertible."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import moteur as M
import graphiques as G
from moteur import S0


def affiche(m):
    bandeau_modele, defauts, marge, nominal, r = m.bandeau_modele, m.defauts, m.marge, m.nominal, m.r
    recovery, trajectoires = m.recovery, m.trajectoires

    st.title("Barrier Reverse Convertible", help="**Ce que c'est.** Un coupon garanti eleve, et le capital rembourse *sauf* si le sous-jacent\nfranchit la barriere ET finit sous le strike.\n\n**Comment c'est construit.** Une obligation, plus la vente par le client d'un put\ndown-and-in a la banque.\n\n**Le point essentiel.** Le coupon n'est pas un interet. C'est la prime que la banque paie au\nclient pour l'assurer contre la baisse. Coupon eleve = sinistre probable, jamais bonne\naffaire.\n\n**Pourquoi un client l'achete.** Vue neutre : il pense que le sous-jacent ne va pas\nbeaucoup bouger et se fait payer pour ca.\n\n**Ce qu'il faut regarder.** Le graphe des trajectoires : deux chemins peuvent finir au meme\nniveau avec des resultats differents. Le chemin compte, pas seulement l'arrivee.")
    st.caption("Le client ne recoit pas un interet. Il encaisse une prime d'assurance.")
    bandeau_modele()

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Maturite (annees)", 0.25, 3.0, 1.0, 0.25)
    bar = c2.slider("Barriere (% du spot)", 50, 95, 70, 5,
                    help="Le niveau qui ARME le risque. Le toucher ne fait pas perdre : "
                         "c'est le strike qui decide a la fin.")
    strike = c3.slider("Strike (% du spot)", 80, 110, 100, 5)

    continu = st.radio("Observation de la barriere",
                       ["Continue (americaine)", "A l'echeance (europeenne)"],
                       horizontal=True).startswith("Continue")

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

    fig = G.payoff(
        spots, pay_touche, direct, nominal, "Si la barriere a ete touchee",
        titre=f"Coupon de {cpn*100:,.2f}% quoi qu'il arrive, capital a risque sous "
              f"{strike/100*S0:,.0f}",
        sous_titre="Le coupon est plafonne : meme si le sous-jacent double, le client "
                   "touche le meme montant",
        niveaux=[(bar / 100 * S0, f"barriere  {bar/100*S0:,.0f}"),
                 (strike / 100 * S0, f"strike  {strike/100*S0:,.0f}")],
        label_direct="S'il avait achete le sous-jacent")
    ax = fig.axes[0]
    ax.plot(spots, pay_intact, color=G.GAIN, lw=2.4, ls="-", zorder=5,
            label="Si la barriere n'a jamais ete touchee")
    G.niveau_horizontal(ax, plafond, f"plafond  {plafond:,.0f}", G.GAIN, x=0.30)
    leg = ax.legend(fontsize=9.5, loc="upper left", frameon=True, framealpha=0.95,
                    edgecolor="#E0E0E0")
    leg.get_frame().set_linewidth(0.8)
    G.legende_zones(ax)
    st.pyplot(fig)
    plt.close(fig)

    # --- graphe 2 : deux clients, meme niveau final, resultats differents
    cible = 0.88 * S0
    cand_i = np.where((np.abs(ST - cible) < 1.2) & ~touche)[0]
    cand_t = np.where((np.abs(ST - cible) < 1.2) & touche)[0]
    if len(cand_i) and len(cand_t):
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        ax.axhspan(0, bar / 100 * S0, color=G.FOND_PERTE, alpha=0.07, zorder=0)
        tt = np.linspace(0, T, ch.shape[1])
        i_ok = cand_i[np.argmax(ch[cand_i].min(axis=1))]
        i_ko = cand_t[np.argmin(ch[cand_t].min(axis=1))]
        ax.plot(tt, ch[i_ok], lw=2.4, color=G.GAIN, zorder=3,
                label=f"Client A — barriere intacte, recoit {plafond:,.0f}")
        pay_ko = nominal * ST[i_ko] / (strike / 100 * S0) + cpn * nominal * T
        ax.plot(tt, ch[i_ko], lw=2.4, color=G.PERTE, zorder=3,
                label=f"Client B — barriere touchee, recoit {pay_ko:,.0f}")
        ax.scatter([T, T], [ch[i_ok, -1], ch[i_ko, -1]], s=70, zorder=5,
                   color=["white", "white"], edgecolor=[G.GAIN, G.PERTE], lw=2)
        ax.set_xlim(0, T * 1.02)
        bas = min(ch[i_ok].min(), ch[i_ko].min(), bar / 100 * S0)
        haut = max(ch[i_ok].max(), ch[i_ko].max(), S0)
        ax.set_ylim(bas * 0.90, haut * 1.10)
        G.style(ax, "Deux clients, le meme niveau final, deux resultats",
                "Le chemin decide, pas seulement l'arrivee",
                "Temps (annees)", "Niveau du sous-jacent")
        G.niveau_horizontal(ax, bar / 100 * S0, f"barriere  {bar/100*S0:,.0f}", G.PERTE)
        G.niveau_horizontal(ax, S0, f"depart  {S0:,.0f}", G.NIVEAU, x=0.16)
        leg = ax.legend(fontsize=9.5, loc="lower left", frameon=True, framealpha=0.95,
                        edgecolor="#E0E0E0")
        leg.get_frame().set_linewidth(0.8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- graphe 3 : le faisceau complet
    couleurs = [G.PERTE if touche[i] else G.GAIN for i in range(45)]
    fig = G.trajectoires(
        ch, T, [(bar / 100 * S0, f"barriere  {bar/100*S0:,.0f}", G.PERTE),
                (S0, f"depart  {S0:,.0f}", G.NIVEAU)],
        couleurs,
        f"{(1-p_no.mean())*100:,.0f}% des trajectoires franchissent la barriere",
        "Rouge : barriere touchee. Vert : intacte.",
        zone_sous=bar / 100 * S0)
    st.pyplot(fig)
    plt.close(fig)

    if st.checkbox("Decomposer le coupon"):
        base = M.coupon_brc(ch, T, r, nominal, bar / 100, strike / 100, continu,
                            marge, sg, None, recovery)
        st.table([
            {"Composante": "Coupon hors risque emetteur", "Valeur": f"{base*100:,.2f}% p.a."},
            {"Composante": "Apport du spread emetteur", "Valeur": f"{(cpn-base)*100:+,.2f}% p.a."},
            {"Composante": "Coupon offert au client", "Valeur": f"{cpn*100:,.2f}% p.a."},
        ])
        st.caption("Le spread emetteur n'est pas un cadeau : c'est la remuneration d'un "
                   "risque de credit que le client porte sans toujours le savoir.")

    st.info(
        "**Le coupon a trois sources** : les taux, la prime du put que le client vend, et "
        "le spread de l'emetteur. Coupon eleve = sinistre probable, ou signature faible.\n\n"
        "**Pres de la barriere, le gamma explose** : le trader doit trader des volumes "
        "enormes en peu de temps. D'ou le prix degrade sur les barrieres proches du spot. "
        "La page Cout de couverture chiffre ce mecanisme.")
