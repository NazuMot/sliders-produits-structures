"""
=======================================================================
GRAPHIQUES - style et constructeurs
=======================================================================

Principe : un graphique doit repondre a la question avant qu'on lise la
legende. Trois regles appliquees partout.

  1. Les zones sont colorees. Vert = le produit bat la detention directe,
     rouge = il perd. La reponse se voit, elle ne se deduit pas.
  2. Les niveaux cles sont annotes sur le graphe, avec leur valeur.
     Pas d'aller-retour vers une legende.
  3. Le meme code couleur partout : une couleur = une signification.
=======================================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- palette, une couleur = une signification
# Une couleur = un sens, dans toute l'application.
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

ENCRE = "#1A1A19"
ENCRE_2 = "#55534E"
MUET = "#8A8880"
GRILLE = "#E8E7E1"
SURFACE = "#FCFBF8"


def applique_style():
    """
    Style global, appele une fois au demarrage.

    La surface est claire et assumee : l'appli peut etre affichee en theme
    sombre, et un fond blanc par defaut ressort comme un rectangle colle
    sur la page. Une surface volontaire, avec un cadre fin, se lit comme
    une carte voulue.
    """
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 130,
        "font.size": 10.5,
        "axes.edgecolor": GRILLE,
        "axes.linewidth": 1.0,
        "axes.labelcolor": ENCRE_2,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRILLE,
        "grid.linewidth": 0.9,
        "xtick.color": MUET,
        "ytick.color": MUET,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
    })


def figure(largeur=10.5, hauteur=5.4, n=1):
    """Figure au format standard de l'appli."""
    fig, ax = plt.subplots(1, n, figsize=(largeur, hauteur))
    return fig, ax


def bande_densite(ax, echantillon, x_min, x_max, hauteur=0.11, seuil=None):
    """
    Bande de densite sous l'axe : ou le sous-jacent finit reellement.

    Un profil de remboursement traite +100% et -10% a egalite, alors que
    l'un est quasi impossible. Sans cette bande, un point mort tres haut
    parait inquietant ; avec elle, on voit que la zone est presque vide.

    L'echantillon vient du Monte Carlo deja calcule : cout nul.
    """
    ech = np.asarray(echantillon, dtype=float)
    ech = ech[(ech >= x_min) & (ech <= x_max)]
    if ech.size < 50:
        return
    hist, bords = np.histogram(ech, bins=64, range=(x_min, x_max), density=True)
    if hist.max() <= 0:
        return
    hist = hist / hist.max()
    centres = 0.5 * (bords[:-1] + bords[1:])

    y0, y1 = ax.get_ylim()
    span = y1 - y0
    plancher = y0 - span * (hauteur + 0.10)
    ax.set_ylim(plancher - span * 0.02, y1)

    for i in range(len(centres)):
        col = PERTE if (seuil is not None and centres[i] < seuil) else PRODUIT
        ax.fill_between([bords[i], bords[i + 1]], plancher,
                        plancher + span * hauteur * hist[i],
                        color=col, alpha=0.45, lw=0, zorder=1)

    ax.plot([x_min, x_max], [plancher, plancher], color=GRILLE, lw=1.0, zorder=2)
    # les graduations ne doivent pas descendre dans la bande : un "0 CHF"
    # affiche en face de l'histogramme n'a aucun sens
    ax.set_yticks([t for t in ax.get_yticks() if y0 - span * 0.01 <= t <= y1])
    ax.annotate("ou le sous-jacent finit reellement",
                xy=(0.012, plancher + span * hauteur),
                xycoords=("axes fraction", "data"),
                xytext=(0, 6), textcoords="offset points",
                fontsize=8.5, color=MUET, va="bottom")


def style(ax, titre=None, sous_titre=None, xlabel=None, ylabel=None):
    """Applique le style commun : pas de cadre inutile, grille discrete."""
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color("#CCCCCC")
    ax.grid(alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors="#555555", length=3)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#444444", labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#444444", labelpad=8)
    if titre:
        ax.set_title(titre, fontsize=12.5, fontweight="medium", color=ENCRE,
                     loc="left", pad=30 if sous_titre else 12)
    if sous_titre:
        ax.annotate(sous_titre, xy=(0, 1.0), xycoords="axes fraction",
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=9.5, color=ENCRE_2, va="bottom")


def format_chf(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", " ")))


def niveau_vertical(ax, x, texte, couleur=NIVEAU, haut=0.045, style_trait=":"):
    """
    Trait vertical annote directement sur le graphe.
    Le texte est place en bas : c'est la zone la moins occupee, et ca evite
    la collision avec la legende qui est presque toujours en haut.
    """
    ax.axvline(x, color=couleur, ls=style_trait, lw=1.4, zorder=1)
    ymin, ymax = ax.get_ylim()
    ax.text(x, ymin + (ymax - ymin) * haut, f" {texte}", fontsize=8.5,
            color=couleur, va="bottom", ha="left", fontweight="bold")


def niveau_horizontal(ax, y, texte, couleur=NIVEAU, x=0.985):
    """
    Repere horizontal annote. Le cartouche opaque est indispensable des
    que le graphe porte des trajectoires : sans lui l'etiquette se perd
    dans le faisceau de courbes.
    """
    ax.axhline(y, color=couleur, ls=(0, (3, 3)), lw=1.3, zorder=5)
    xmin, xmax = ax.get_xlim()
    ax.text(xmin + (xmax - xmin) * x, y, texte, fontsize=8.5, color=couleur,
            va="center", ha="right" if x > 0.5 else "left", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="none", alpha=0.92))


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
           titre=None, sous_titre=None, niveaux=None, ylabel="Remboursement (CHF)",
           label_direct="S'il avait achete le sous-jacent", figsize=(10.5, 5.4),
           montrer_points_morts=True, echantillon=None, seuil_densite=None):
    """
    Graphe de payoff standard : le produit, la detention directe, et les
    zones coloriees entre les deux.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(spots, produit, direct, where=(produit >= direct),
                    color=FOND_GAIN, alpha=0.10, interpolate=True, zorder=0)
    ax.fill_between(spots, produit, direct, where=(produit < direct),
                    color=FOND_PERTE, alpha=0.10, interpolate=True, zorder=0)

    ax.plot(spots, direct, color=DIRECT, ls=(0, (5, 4)), lw=1.5, zorder=3,
            label=label_direct)
    ax.plot(spots, produit, color=PRODUIT, lw=2.8, zorder=4, label=label_produit)

    ax.axhline(nominal, color=GRILLE, lw=1.2, zorder=0)
    ax.set_xlim(spots.min(), spots.max())
    style(ax, titre, sous_titre, "Niveau du sous-jacent a l'echeance", ylabel)
    format_chf(ax)

    for x, txt in (niveaux or []):
        niveau_vertical(ax, x, txt)

    if montrer_points_morts:
        for x in points_morts(spots, produit, direct):
            if spots.min() + 2 < x < spots.max() - 2:
                y = np.interp(x, spots, produit)
                ax.scatter([x], [y], s=55, color="white", edgecolor=DIRECT,
                           lw=1.8, zorder=6)
                ax.annotate(f"point mort\n{x:,.0f}", (x, y),
                            textcoords="offset points", xytext=(0, -34),
                            ha="center", fontsize=9, color="#A32D2D",
                            fontweight="medium")

    leg = ax.legend(fontsize=9.5, loc="upper left", frameon=True, framealpha=0.95,
                    edgecolor=GRILLE)
    leg.get_frame().set_linewidth(0.8)

    if echantillon is not None:
        bande_densite(ax, echantillon, spots.min(), spots.max(),
                      seuil=seuil_densite)

    fig.tight_layout()
    return fig


def legende_zones(ax):
    """Petite note expliquant le sens des aplats de couleur."""
    ax.text(0.985, 1.0,
            "vert : le produit fait mieux   |   rouge : le sous-jacent fait mieux",
            transform=ax.transAxes, fontsize=8.5, color=MUET,
            ha="right", va="bottom")


def trajectoires(ch, T, niveaux, couleurs, titre, sous_titre=None,
                 ylabel="Niveau du sous-jacent", n_max=45, zone_sous=None,
                 figsize=(10.5, 5.0)):
    """
    Faisceau de trajectoires simulees, avec zone de danger ombree.
    couleurs : liste de couleurs, une par trajectoire affichee.
    """
    fig, ax = plt.subplots(figsize=figsize)
    tt = np.linspace(0, T, ch.shape[1])

    if zone_sous is not None:
        ax.axhspan(0, zone_sous, color=FOND_PERTE, alpha=0.07, zorder=0)

    for i in range(min(n_max, ch.shape[0])):
        ax.plot(tt, ch[i], lw=0.85, alpha=0.7, color=couleurs[i], zorder=2)

    ax.set_xlim(0, T)
    ax.set_ylim(max(0, ch[:n_max].min() * 0.92), ch[:n_max].max() * 1.06)
    style(ax, titre, sous_titre, "Temps (annees)", ylabel)
    for y, txt, col in niveaux:
        niveau_horizontal(ax, y, txt, col)
    fig.tight_layout()
    return fig
