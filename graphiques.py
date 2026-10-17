"""
=======================================================================
GRAPHIQUES - graphes interactifs (Altair)
=======================================================================

Principe : un graphique doit repondre a la question avant qu'on lise la
legende. Trois regles appliquees partout.

  1. Les zones sont colorees. Vert = le produit bat la detention directe,
     rouge = il perd. La reponse se voit, elle ne se deduit pas.
  2. Les niveaux cles sont annotes sur le graphe, avec leur valeur.
     Pas d'aller-retour vers une legende.
  3. Le meme code couleur partout : une couleur = une signification.

Interactivite : au survol, une ligne verticale suit la souris et une
infobulle donne toutes les valeurs a cet endroit, comme sur une appli
bancaire. Les graphes de trajectoires mettent en avant la trajectoire
survolee. Un graphe peut aussi renvoyer le point clique (page grecques).

Altair est fourni avec Streamlit : aucune dependance en plus. Le theme
de Streamlit gere les couleurs de texte et de grille, ce qui garde les
graphes lisibles en theme clair comme en theme sombre.
=======================================================================
"""

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --- palette, une couleur = une signification
# Avant, le rouge designait a la fois le produit et la perte : impossible
# de savoir si une zone rouge signalait un danger ou simplement la courbe.
PRODUIT = "#2A78D6"      # le produit structure
DIRECT = "#8A8880"       # detention directe du sous-jacent
GAIN = "#1BAF7A"         # zone ou le produit fait mieux
PERTE = "#E34948"        # zone ou le sous-jacent fait mieux, et les barrieres
NIVEAU = "#9A9891"       # reperes neutres : depart, strike
ACCENT = "#EB6834"       # serie secondaire
FOND_GAIN = "#1BAF7A"
FOND_PERTE = "#E34948"

TRAITS = {"plein": [1, 0], "tirete": [6, 4], "pointille": [2, 3]}
HAUTEUR = 380


# =====================================================================
# FORMATS
# =====================================================================

def eur(v, signe=False):
    """1040.3 -> '1 040 EUR' (espace comme separateur de milliers)."""
    txt = f"{v:+,.0f}" if signe else f"{v:,.0f}"
    return txt.replace(",", " ") + " EUR"


def pct(v, decimales=1, signe=False):
    """v deja en pourcentage : 12.345 -> '12.3 %'."""
    return (f"{v:+.{decimales}f}" if signe else f"{v:.{decimales}f}") + " %"


def niveau(x, depart=100.0):
    """Niveau du sous-jacent avec sa variation : '85 (-15 %)'."""
    return f"{x:,.0f} ({(x / depart - 1) * 100:+.0f} %)".replace(",", " ")


# Expression Vega : separateur de milliers en espace sur les axes
AXE_EUR = "replace(format(datum.value, ',.0f'), /,/g, ' ')"


def _graduations(a, b, n=6):
    """Graduations rondes entre a et b (pour ne pas graduer la bande de densite)."""
    etendue = max(b - a, 1e-9)
    pas = 10 ** math.floor(math.log10(etendue / n))
    for mult in (1, 2, 2.5, 5, 10):
        if etendue / (pas * mult) <= n:
            pas *= mult
            break
    debut = math.ceil(a / pas) * pas
    return [round(v, 10) for v in np.arange(debut, b + pas * 1e-6, pas)]


def _titre(titre, sous_titre):
    if not titre:
        return alt.TitleParams("")
    return alt.TitleParams(titre, subtitle=sous_titre or "", anchor="start",
                           fontSize=15, fontWeight=500, subtitleFontSize=12,
                           subtitlePadding=4, offset=10)


def _legende():
    return alt.Legend(orient="top", title=None, labelLimit=420, symbolStrokeWidth=3,
                      symbolSize=260, labelFontSize=12)


# =====================================================================
# AFFICHAGE
# =====================================================================

def affiche(chart, cle=None, sur_clic=None):
    """
    Affiche un graphe sur toute la largeur. Avec sur_clic, un clic sur le
    graphe appelle la fonction (elle lit st.session_state[cle]) ; seule la
    selection nommee 'clic' declenche un rechargement, pas le survol.
    """
    if sur_clic is None:
        st.altair_chart(chart, width="stretch", key=cle)
    else:
        st.altair_chart(chart, width="stretch", key=cle, on_select=sur_clic,
                        selection_mode=["clic"])


def point_clique(cle, champ="idx"):
    """Valeur du champ au dernier point clique sur le graphe cle, ou None."""
    etat = st.session_state.get(cle)
    if not etat:
        return None
    points = etat.get("selection", {}).get("clic") or []
    if not points or champ not in points[0]:
        return None
    return points[0][champ]


# =====================================================================
# BRIQUES COMMUNES
# =====================================================================

def _niveaux_verticaux(niveaux, y_texte):
    """Traits verticaux annotes en bas du graphe : (x, texte[, couleur])."""
    if not niveaux:
        return []
    df = pd.DataFrame([{"x": n[0], "texte": n[1],
                        "couleur": n[2] if len(n) > 2 else NIVEAU} for n in niveaux])
    trait = alt.Chart(df).mark_rule(strokeDash=[2, 3], strokeWidth=1.4).encode(
        x="x:Q", color=alt.Color("couleur:N", scale=None))
    texte = alt.Chart(df).mark_text(align="left", baseline="bottom", dx=4, fontSize=11,
                                    fontWeight="bold").encode(
        x="x:Q", y=alt.datum(y_texte), text="texte:N",
        color=alt.Color("couleur:N", scale=None))
    return [trait, texte]


def _niveaux_horizontaux(niveaux, x_texte, aligne="right"):
    """Reperes horizontaux annotes : (y, texte, couleur[, position 0-1])."""
    if not niveaux:
        return []
    lignes = []
    for n in niveaux:
        pos = n[3] if len(n) > 3 else None
        lignes.append({"y": n[0], "texte": n[1], "couleur": n[2] if len(n) > 2 else NIVEAU,
                       "x": x_texte if pos is None else pos})
    df = pd.DataFrame(lignes)
    trait = alt.Chart(df).mark_rule(strokeDash=[4, 4], strokeWidth=1.3).encode(
        y="y:Q", color=alt.Color("couleur:N", scale=None))
    texte = alt.Chart(df).mark_text(align=aligne, baseline="bottom", dy=-3, fontSize=11,
                                    fontWeight="bold").encode(
        x="x:Q", y="y:Q", text="texte:N", color=alt.Color("couleur:N", scale=None))
    return [trait, texte]


def _notes(notes):
    """Textes libres poses sur le graphe : (x, y, texte, couleur[, dx, dy])."""
    if not notes:
        return []
    couches = []
    for n in notes:
        dx, dy = (n[4], n[5]) if len(n) > 5 else (8, -8)
        df = pd.DataFrame([{"x": n[0], "y": n[1], "texte": n[2]}])
        couches.append(alt.Chart(df).mark_text(
            align="left", baseline="bottom", dx=dx, dy=dy, fontSize=11.5, fontWeight="bold",
            color=n[3], lineBreak="\n").encode(x="x:Q", y="y:Q", text="texte:N"))
    return couches


def _echelles(series):
    noms = [s["nom"] for s in series]
    couleur = alt.Scale(domain=noms, range=[s.get("couleur", PRODUIT) for s in series])
    trait = alt.Scale(domain=noms, range=[TRAITS[s.get("trait", "plein")] for s in series])
    epaisseur = alt.Scale(domain=noms, range=[s.get("epaisseur", 2.6) for s in series])
    return couleur, trait, epaisseur


# =====================================================================
# COURBES : le graphe generique avec survol unifie
# =====================================================================

def courbes(x, series, titre=None, sous_titre=None, xlabel=None, ylabel=None,
            fmt_x=None, fmt_y=None, nom_x=None, axe_y_eur=False, x_log=False,
            x_domaine=None, y_domaine=None, niveaux_v=None, niveaux_h=None, notes=None,
            zone_x=None, zone_y=None, legende=True, clic=False, hauteur=HAUTEUR,
            lignes_infobulle=None, ticks_x=None, ticks_y=None, nb_ticks_x=None, zero=False,
            marqueurs=None):
    """
    Plusieurs series sur un meme axe x, avec une infobulle qui donne toutes
    les valeurs a l'abscisse survolee.

    series : liste de dict {nom, y, couleur, trait ('plein'|'tirete'|'pointille'),
             epaisseur, points (bool), aire (bool), interpolation}
    fmt_x, fmt_y : fonctions python valeur -> texte pour l'infobulle.
    lignes_infobulle : liste (titre, tableau de textes) ajoutee a l'infobulle.
    zone_x / zone_y : (debut, fin, couleur) bande de fond coloree.
    clic : ajoute une selection 'clic' sur le champ idx (indice du point).
    marqueurs : gros points poses sur le graphe, liste (x, y, couleur).
    """
    x = np.asarray(x, dtype=float)
    fmt_x = fmt_x or (lambda v: f"{v:,.2f}")
    fmt_y = fmt_y or (lambda v: f"{v:,.2f}")

    # table large pour l'infobulle, table longue pour les traces
    large = pd.DataFrame({"x": x, "idx": np.arange(len(x)), "x_txt": [fmt_x(v) for v in x]})
    infobulle = [alt.Tooltip("x_txt:N", title=nom_x or xlabel or "x")]
    longues = []
    for k, s in enumerate(series):
        y = np.asarray(s["y"], dtype=float)
        large[f"s{k}"] = [fmt_y(v) if np.isfinite(v) else "-" for v in y]
        infobulle.append(alt.Tooltip(f"s{k}:N", title=s["nom"]))
        longues.append(pd.DataFrame({"x": x, "y": y, "serie": s["nom"], "idx": np.arange(len(x))}))
    for j, (nom, textes) in enumerate(lignes_infobulle or []):
        large[f"i{j}"] = list(textes)
        infobulle.append(alt.Tooltip(f"i{j}:N", title=nom))
    long = pd.concat(longues, ignore_index=True)
    long = long[np.isfinite(long["y"])]

    if y_domaine is None:
        lo, hi = float(long["y"].min()), float(long["y"].max())
        if zero:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        marge = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1.0
        y_domaine = (lo - marge, hi + marge)
    if x_domaine is None:
        x_domaine = (float(x.min()), float(x.max()))

    ech_x = alt.Scale(domain=list(x_domaine), type="log" if x_log else "linear",
                      nice=False, zero=False)
    axe_x = alt.Axis(title=xlabel, values=ticks_x or alt.Undefined,
                     tickCount=nb_ticks_x or 10)
    ech_y = alt.Scale(domain=list(y_domaine), nice=False, zero=False)
    axe_y = alt.Axis(title=ylabel, labelExpr=AXE_EUR if axe_y_eur else alt.Undefined,
                     values=ticks_y or alt.Undefined)
    enc_x = alt.X("x:Q", scale=ech_x, axis=axe_x)
    enc_y = alt.Y("y:Q", scale=ech_y, axis=axe_y)

    couleur, trait, epaisseur = _echelles(series)
    leg = _legende() if legende and len(series) > 1 else None
    couches = []

    for zone, axe in ((zone_x, "x"), (zone_y, "y")):
        if zone:
            d = pd.DataFrame([{"a": zone[0], "b": zone[1]}])
            enc = {"x": alt.X("a:Q", scale=ech_x), "x2": "b:Q"} if axe == "x" else \
                  {"y": alt.Y("a:Q", scale=ech_y), "y2": "b:Q"}
            couches.append(alt.Chart(d).mark_rect(opacity=0.08, color=zone[2]).encode(**enc))

    if zero:
        couches.append(alt.Chart(pd.DataFrame([{"y": 0.0}])).mark_rule(
            color=NIVEAU, opacity=0.5).encode(y=alt.Y("y:Q", scale=ech_y)))

    aires = [s["nom"] for s in series if s.get("aire")]
    if aires:
        couches.append(alt.Chart(long[long["serie"].isin(aires)]).mark_area(
            opacity=0.30, interpolate="step-after").encode(
            enc_x, enc_y, color=alt.Color("serie:N", scale=couleur, legend=None)))

    interp = series[0].get("interpolation", "linear")
    couches.append(alt.Chart(long).mark_line(interpolate=interp).encode(
        enc_x, enc_y,
        color=alt.Color("serie:N", scale=couleur, legend=leg),
        strokeDash=alt.StrokeDash("serie:N", scale=trait, legend=None),
        size=alt.Size("serie:N", scale=epaisseur, legend=None)))

    a_points = [s["nom"] for s in series if s.get("points")]
    if a_points:
        couches.append(alt.Chart(long[long["serie"].isin(a_points)]).mark_point(
            filled=True, fill="white", size=70, strokeWidth=2).encode(
            enc_x, enc_y, stroke=alt.Stroke("serie:N", scale=couleur, legend=None)))

    if marqueurs:
        dm = pd.DataFrame([{"x": a, "y": b, "couleur": c} for a, b, c in marqueurs])
        couches.append(alt.Chart(dm).mark_point(filled=True, size=130, stroke="white",
                                                strokeWidth=2, opacity=1).encode(
            x=alt.X("x:Q", scale=ech_x), y=alt.Y("y:Q", scale=ech_y),
            color=alt.Color("couleur:N", scale=None)))

    couches += _niveaux_verticaux(niveaux_v, y_domaine[0] + (y_domaine[1] - y_domaine[0]) * 0.03)
    couches += _niveaux_horizontaux(niveaux_h, x_domaine[1])
    couches += _notes(notes)

    # survol : ligne verticale + infobulle unique + un point sur chaque courbe
    survol = alt.selection_point(name="survol", nearest=True, on="pointerover",
                                 fields=["x"], empty=False, clear="pointerout")
    regle = alt.Chart(large).mark_rule(color=NIVEAU, strokeWidth=1.2).encode(
        x=alt.X("x:Q", scale=ech_x),
        opacity=alt.condition(survol, alt.value(0.9), alt.value(0)),
        tooltip=infobulle).add_params(survol)
    if clic:
        regle = regle.add_params(alt.selection_point(
            name="clic", nearest=True, on="click", fields=["idx"], empty=False))
    couches.append(regle)
    couches.append(alt.Chart(long).mark_point(filled=True, size=65, opacity=1).encode(
        enc_x, enc_y, color=alt.Color("serie:N", scale=couleur, legend=None)
    ).transform_filter(survol))

    return alt.layer(*couches).properties(
        title=_titre(titre, sous_titre), height=hauteur).resolve_scale(color="independent")


# =====================================================================
# PROFIL DE REMBOURSEMENT
# =====================================================================

def points_morts(spots, produit, direct, max_pts=2):
    """
    Abscisses ou le produit croise VRAIMENT la detention directe.

    Piege evite ici : quand les deux courbes se superposent sur toute une
    plage (un produit qui suit le sous-jacent a 1:1 par exemple), l'ecart
    oscille autour de zero au bruit numerique pres et une detection naive
    signale des centaines de faux croisements. On ignore donc une bande
    morte autour de zero et on ne garde que les changements de signe
    francs, espaces les uns des autres.
    """
    d = np.asarray(produit, float) - np.asarray(direct, float)
    tol = max(abs(np.asarray(direct, float)).max() * 0.004, 1e-9)
    signe = np.where(d > tol, 1, np.where(d < -tol, -1, 0))
    out = []
    dernier = None
    for i in range(len(signe) - 1):
        a, b = signe[i], signe[i + 1]
        if a == 0 or b == 0 or a == b:
            continue
        if d[i + 1] == d[i]:
            continue
        x = spots[i] - d[i] * (spots[i + 1] - spots[i]) / (d[i + 1] - d[i])
        if dernier is not None and abs(x - dernier) < (spots[-1] - spots[0]) * 0.06:
            continue
        out.append(x)
        dernier = x
        if len(out) >= max_pts:
            break
    return out


def payoff(spots, produit, direct, nominal, label_produit,
           titre=None, sous_titre=None, niveaux=None, ylabel="Remboursement (EUR)",
           label_direct="S'il avait achete le sous-jacent", montrer_points_morts=True,
           echantillon=None, seuil_densite=None, autres=None, niveaux_h=None, notes=None,
           depart=100.0, legende_zones=False, hauteur=HAUTEUR + 40):
    """
    Graphe de payoff standard : le produit, la detention directe, et les
    zones coloriees entre les deux. Au survol : ce que touche le client, ce
    qu'il aurait eu en direct, l'ecart, et la probabilite que le sous-jacent
    finisse sous ce niveau (lue dans le Monte Carlo deja calcule).

    autres : series en plus, liste de dict {nom, y, couleur, trait, epaisseur}.
    niveaux_h : reperes horizontaux (y, texte, couleur[, x]).
    legende_zones : rappelle sous le titre le sens des aplats vert et rouge.
    """
    spots = np.asarray(spots, float)
    produit = np.asarray(produit, float)
    direct = np.asarray(direct, float)
    series = [{"nom": label_direct, "y": direct, "couleur": DIRECT, "trait": "tirete",
               "epaisseur": 1.6},
              {"nom": label_produit, "y": produit, "couleur": PRODUIT, "epaisseur": 3.0}]
    series += autres or []

    toutes = np.concatenate([produit, direct] + [np.asarray(s["y"], float) for s in autres or []])
    lo, hi = float(toutes.min()), float(toutes.max())
    span = hi - lo or 1.0
    lo, hi = lo - span * 0.04, hi + span * 0.06
    span = hi - lo

    # bande de densite sous l'axe : ou le sous-jacent finit reellement
    bande = None
    infos = [("Ecart avec le direct", [eur(v, signe=True) for v in produit - direct])]
    if echantillon is not None:
        ech = np.sort(np.asarray(echantillon, float))
        p_sous = np.searchsorted(ech, spots) / max(len(ech), 1) * 100
        infos.append(("Probabilite de finir sous ce niveau", [pct(v, 0) for v in p_sous]))
        hist, bords = np.histogram(ech, bins=64, range=(spots.min(), spots.max()))
        if hist.max() > 0:
            plancher = lo - span * 0.20
            haut_bande = span * 0.13
            part = hist / max(len(ech), 1) * 100
            bande = pd.DataFrame({
                "a": bords[:-1], "b": bords[1:], "y0": plancher,
                "y1": plancher + haut_bande * hist / hist.max(),
                "couleur": [PERTE if (seuil_densite is not None and c < seuil_densite)
                            else PRODUIT for c in 0.5 * (bords[:-1] + bords[1:])],
                "txt": [f"{a:,.0f} a {b:,.0f}" for a, b in zip(bords[:-1], bords[1:])],
                "part": [pct(p, 1) for p in part]})
            lo = plancher - span * 0.02

    ticks = _graduations(lo if bande is None else bande["y0"].iloc[0] + span * 0.20, hi, n=8)
    pm = []
    if montrer_points_morts:
        for xm in points_morts(spots, produit, direct):
            if spots.min() + 2 < xm < spots.max() - 2:
                pm.append((xm, float(np.interp(xm, spots, produit))))

    if legende_zones:
        note = "vert : le produit fait mieux   |   rouge : le sous-jacent fait mieux"
        sous_titre = [sous_titre, note] if sous_titre else note
    chart = courbes(
        spots, series, titre=titre, sous_titre=sous_titre,
        xlabel="Niveau du sous-jacent a l'echeance", ylabel=ylabel,
        fmt_x=lambda v: niveau(v, depart), fmt_y=eur, nom_x="Sous-jacent a l'echeance",
        axe_y_eur=True, y_domaine=(lo, hi), niveaux_h=niveaux_h, notes=notes,
        lignes_infobulle=infos, hauteur=hauteur, ticks_y=ticks, nb_ticks_x=10)

    ech_x = alt.Scale(domain=[float(spots.min()), float(spots.max())], nice=False, zero=False)
    ech_y = alt.Scale(domain=[lo, hi], nice=False, zero=False)
    zones = pd.DataFrame({"x": spots, "d": direct,
                          "haut": np.maximum(produit, direct), "bas": np.minimum(produit, direct)})
    fond = [
        alt.Chart(zones).mark_area(opacity=0.13, color=FOND_GAIN).encode(
            x=alt.X("x:Q", scale=ech_x), y=alt.Y("d:Q", scale=ech_y), y2="haut:Q"),
        alt.Chart(zones).mark_area(opacity=0.13, color=FOND_PERTE).encode(
            x=alt.X("x:Q", scale=ech_x), y=alt.Y("bas:Q", scale=ech_y), y2="d:Q"),
        alt.Chart(pd.DataFrame([{"y": nominal}])).mark_rule(color=NIVEAU, opacity=0.35).encode(
            y=alt.Y("y:Q", scale=ech_y)),
    ]
    if bande is not None:
        fond.append(alt.Chart(bande).mark_rect(opacity=0.45).encode(
            x=alt.X("a:Q", scale=ech_x), x2="b:Q", y=alt.Y("y0:Q", scale=ech_y), y2="y1:Q",
            color=alt.Color("couleur:N", scale=None),
            tooltip=[alt.Tooltip("txt:N", title="Sous-jacent a l'echeance entre"),
                     alt.Tooltip("part:N", title="Part des scenarios")]))
        fond.append(alt.Chart(pd.DataFrame([{"x": float(spots.min()), "y": float(bande["y0"].iloc[0] + span * 0.13)}]))
                    .mark_text(align="left", baseline="bottom", dx=4, dy=-2, fontSize=10.5,
                               color=DIRECT, text="ou le sous-jacent finit reellement")
                    .encode(x=alt.X("x:Q", scale=ech_x), y=alt.Y("y:Q", scale=ech_y)))

    niv = _niveaux_verticaux(niveaux, (lo if bande is None else float(bande["y0"].iloc[0]) + span * 0.20)
                             + span * 0.02)
    morts = []
    if pm:
        dpm = pd.DataFrame([{"x": a, "y": b, "txt": f"point mort\n{a:,.0f}"} for a, b in pm])
        morts = [
            alt.Chart(dpm).mark_point(filled=True, fill="white", size=80, strokeWidth=2,
                                      color=DIRECT).encode(
                x=alt.X("x:Q", scale=ech_x), y=alt.Y("y:Q", scale=ech_y)),
            alt.Chart(dpm).mark_text(baseline="top", dy=12, fontSize=11, color="#C0392B",
                                     lineBreak="\n", fontWeight=500).encode(
                x=alt.X("x:Q", scale=ech_x), y=alt.Y("y:Q", scale=ech_y), text="txt:N")]

    # l'axe y (ticks) ne gradue que la partie remboursement, pas la bande de densite
    chart.layer[0:0] = fond
    chart.layer += niv + morts
    return chart


# =====================================================================
# TRAJECTOIRES
# =====================================================================

def trajectoires(ch, T, niveaux, couleurs, titre, sous_titre=None,
                 ylabel="Niveau du sous-jacent", n_max=45, zone_sous=None, statuts=None,
                 legende=None, points=None, max_pas=130, hauteur=HAUTEUR, t_fin=None,
                 dates_v=None):
    """
    Faisceau de trajectoires simulees, avec zone de danger ombree.
    Au survol, la trajectoire la plus proche ressort et l'infobulle donne
    sa date, son niveau et son issue.

    couleurs : une couleur par trajectoire affichee.
    statuts : un texte par trajectoire (issue pour le client).
    legende : liste (texte, couleur) affichee au-dessus du graphe.
    points : liste de dict {t, y, couleur} (rappels anticipes, etc.).
    t_fin : longueur de chaque trajectoire (pour couper au rappel), en annees.
    dates_v : traits verticaux discrets (dates d'observation), en annees.
    """
    n = min(n_max, ch.shape[0])
    pas = ch.shape[1]
    garde = np.unique(np.linspace(0, pas - 1, min(pas, max_pas)).round().astype(int))
    tt = np.linspace(0, T, pas)[garde]
    lignes = []
    for i in range(n):
        y = np.asarray(ch[i, garde], float)
        t_i = tt
        if t_fin is not None and t_fin[i] < T:
            coupe = tt <= t_fin[i] + 1e-9
            t_i, y = tt[coupe], y[coupe]
        lignes.append(pd.DataFrame({"t": t_i, "y": y, "traj": i, "couleur": couleurs[i],
                                    "statut": statuts[i] if statuts else "",
                                    "t_txt": [f"{v:,.2f} an" for v in t_i],
                                    "y_txt": [f"{v:,.1f}" for v in y]}))
    df = pd.concat(lignes, ignore_index=True)

    niv_y = [nv[0] for nv in niveaux]
    lo = max(0.0, min(df["y"].min(), *niv_y) * 0.92)
    hi = max(df["y"].max(), *niv_y) * 1.06
    ech_x = alt.Scale(domain=[0, T * 1.02], nice=False)
    ech_y = alt.Scale(domain=[lo, hi], nice=False, zero=False)

    survol = alt.selection_point(name="survol", nearest=True, on="pointerover",
                                 fields=["traj"], empty=True, clear="pointerout")
    couches = []
    if zone_sous is not None:
        couches.append(alt.Chart(pd.DataFrame([{"a": lo, "b": zone_sous}])).mark_rect(
            opacity=0.07, color=FOND_PERTE).encode(y=alt.Y("a:Q", scale=ech_y), y2="b:Q"))
    if dates_v:
        couches.append(alt.Chart(pd.DataFrame({"t": list(dates_v)})).mark_rule(
            color=NIVEAU, opacity=0.35).encode(x=alt.X("t:Q", scale=ech_x)))
    infobulle = [alt.Tooltip("t_txt:N", title="Date"), alt.Tooltip("y_txt:N", title=ylabel)]
    if statuts:
        infobulle.append(alt.Tooltip("statut:N", title="Issue pour le client"))
    couches.append(alt.Chart(df).mark_line(interpolate="linear").encode(
        x=alt.X("t:Q", scale=ech_x, axis=alt.Axis(title="Temps (annees)", tickCount=10)),
        y=alt.Y("y:Q", scale=ech_y, title=ylabel),
        detail="traj:N", color=alt.Color("couleur:N", scale=None),
        opacity=alt.condition(survol, alt.value(0.85), alt.value(0.15)),
        size=alt.condition(survol, alt.value(1.3), alt.value(0.9))))
    couches.append(alt.Chart(df).mark_point(size=30, opacity=0).encode(
        x=alt.X("t:Q", scale=ech_x), y=alt.Y("y:Q", scale=ech_y),
        tooltip=infobulle).add_params(survol))
    if points:
        couches.append(alt.Chart(pd.DataFrame(points)).mark_point(
            filled=True, size=40, strokeWidth=0.8, stroke="white").encode(
            x=alt.X("t:Q", scale=ech_x), y=alt.Y("y:Q", scale=ech_y),
            color=alt.Color("couleur:N", scale=None)))
    couches += _niveaux_horizontaux(niveaux, T * 1.02)

    chart = alt.layer(*couches).properties(title=_titre(titre, sous_titre), height=hauteur)
    if legende:
        dl = pd.DataFrame([{"nom": a, "couleur": b} for a, b in legende])
        chart = alt.layer(*couches, alt.Chart(dl).mark_point(opacity=0).encode(
            color=alt.Color("nom:N", scale=alt.Scale(domain=[a for a, _ in legende],
                                                      range=[b for _, b in legende]),
                            legend=alt.Legend(orient="top", title=None, symbolType="stroke",
                                              symbolStrokeWidth=3, labelFontSize=12)))
        ).properties(title=_titre(titre, sous_titre), height=hauteur)
    return chart.resolve_scale(color="independent")


# =====================================================================
# BARRES ET HISTOGRAMMES
# =====================================================================

def barres(etiquettes, valeurs, couleurs, titre, sous_titre=None, ylabel=None,
           fmt=lambda v: pct(v, 0), infos=None, hauteur=320, seuil_texte=1.2, nom_x="Date"):
    """Diagramme en barres avec la valeur au-dessus et une infobulle par barre."""
    df = pd.DataFrame({"lab": list(etiquettes), "v": list(valeurs), "couleur": list(couleurs),
                       "txt": [fmt(v) if v >= seuil_texte else "" for v in valeurs],
                       "tip": [fmt(v) for v in valeurs], "ordre": range(len(valeurs))})
    infobulle = [alt.Tooltip("lab:N", title=nom_x), alt.Tooltip("tip:N", title=ylabel or "Valeur")]
    for j, (nom, textes) in enumerate(infos or []):
        df[f"i{j}"] = list(textes)
        infobulle.append(alt.Tooltip(f"i{j}:N", title=nom))
    x = alt.X("lab:N", sort=alt.SortField("ordre"), title=None,
              axis=alt.Axis(labelAngle=-45 if len(df) > 8 else 0))
    y = alt.Y("v:Q", title=ylabel, scale=alt.Scale(domain=[0, max(valeurs) * 1.18]))
    base = alt.Chart(df)
    return alt.layer(
        base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x, y, color=alt.Color("couleur:N", scale=None), tooltip=infobulle),
        base.mark_text(baseline="bottom", dy=-3, fontSize=11, fontWeight="bold").encode(
            x, y, text="txt:N"),
    ).properties(title=_titre(titre, sous_titre), height=hauteur)


def histogramme(valeurs, n_bins, titre, sous_titre=None, xlabel=None,
                couleur_bin=None, niveaux=None, fmt_x=lambda v: eur(v, signe=True),
                hauteur=HAUTEUR - 20):
    """Histogramme avec une infobulle par classe : bornes, nombre et part des scenarios."""
    v = np.asarray(valeurs, float)
    hist, bords = np.histogram(v, bins=n_bins)
    centres = 0.5 * (bords[:-1] + bords[1:])
    df = pd.DataFrame({
        "a": bords[:-1], "b": bords[1:], "n": hist,
        "couleur": [couleur_bin(c) if couleur_bin else PRODUIT for c in centres],
        "txt": [f"{fmt_x(a)} a {fmt_x(b)}" for a, b in zip(bords[:-1], bords[1:])],
        "part": [pct(h / len(v) * 100, 1) for h in hist]})
    ech_y = alt.Scale(domain=[0, hist.max() * 1.12], nice=False)
    couches = [alt.Chart(df).mark_rect(opacity=0.85).encode(
        x=alt.X("a:Q", title=xlabel, scale=alt.Scale(nice=False, zero=False)), x2="b:Q",
        y=alt.Y("n:Q", title="Nombre de scenarios", scale=ech_y), y2=alt.datum(0),
        color=alt.Color("couleur:N", scale=None),
        tooltip=[alt.Tooltip("txt:N", title="Resultat entre"),
                 alt.Tooltip("n:Q", title="Scenarios"),
                 alt.Tooltip("part:N", title="Part")])]
    couches += _niveaux_verticaux(niveaux, hist.max() * 1.04)
    return alt.layer(*couches).properties(title=_titre(titre, sous_titre), height=hauteur)
