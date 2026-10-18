"""
Regenere les images du README (docs/*.png) a partir du moteur et de graphiques.py.

    pip install vl-convert-python   (outil de rendu, pas une dependance de l'appli)
    python scripts/captures_readme.py

Parametres par defaut de l'appli (Black-Scholes + skew 2 en vol locale).
"""
import os
import sys

import numpy as np
import vl_convert as vlc

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
import graphiques as G  # noqa: E402
import moteur as M  # noqa: E402

S0, r, q, vol, skew, nominal = 100.0, 0.03, 0.025, 0.22, 2.0, 1000.0
DOCS = os.path.join(RACINE, "docs")


def png(chart, nom, largeur=900):
    spec = chart.to_dict()
    spec["width"] = largeur
    spec["background"] = "white"
    with open(os.path.join(DOCS, nom), "wb") as f:
        f.write(vlc.vegalite_to_png(spec, scale=1.5))
    print("ecrit", nom)


# 1. BRC 1 an, barriere 70 % : profil de remboursement
ch, sg = M.simule_lv(S0, 1.0, r, q, vol, skew, 10000)
cpn = M.coupon_brc(ch, 1.0, r, nominal, 0.70, 1.0, True, 0.008, sg)
spots = np.linspace(30, 190, 500)
plafond = nominal * (1 + cpn)
touche = np.where(spots >= S0, plafond, nominal * spots / S0 + cpn * nominal)
png(G.payoff(spots, touche, nominal * spots / S0, nominal, "Si la barrière a été touchée",
             titre=f"BRC 1 an : coupon de {cpn*100:.2f} % quoi qu'il arrive",
             sous_titre="Capital à risque sous 100 si la barrière de 70 a été touchée",
             niveaux=[(70, "barrière  70"), (100, "strike  100")],
             label_direct="S'il avait acheté le sous-jacent",
             autres=[{"nom": "Si la barrière n'a jamais été touchée", "y": np.full_like(spots, plafond),
                      "couleur": G.GAIN, "epaisseur": 2.4}],
             echantillon=ch[:, -1], seuil_densite=70, legende_zones=True), "brc.png")

# 2. Autocall 3 ans trimestriel : trajectoires et rappels
T, n_obs = 3.0, 12
ch, _ = M.simule_lv(S0, T, r, q, vol, skew, 10000)
idx = M.idx_obs(ch.shape[1] - 1, n_obs)
tt = np.linspace(0, T, ch.shape[1])
couleurs, t_fin, points = [], [], []
for i in range(60):
    k = next((j for j in idx[:-1] if ch[i, j] >= S0), None)
    if k is not None:
        couleurs.append(G.GAIN); t_fin.append(tt[k]); points.append({"t": tt[k], "y": float(ch[i, k]), "couleur": G.GAIN})
    else:
        couleurs.append(G.PERTE if ch[i, -1] < 60 else G.NIVEAU); t_fin.append(T)
duree = M.stats_autocall(ch, T, n_obs, 1.0, 0.6)[0]
png(G.trajectoires(ch, T, [(100, "trigger de rappel  100", G.GAIN), (60, "barrière capital  60", G.PERTE)],
                   couleurs, f"Autocall 3 ans : durée de vie moyenne {duree:.2f} an",
                   "Vert : rappelé par anticipation. Rouge : perte en capital. Gris : va au terme.",
                   n_max=60, zone_sous=60, t_fin=t_fin, points=points,
                   dates_v=[tt[j] for j in idx[:-1]]), "autocall.png")

# 3. Le skew : coupon de l'autocall selon la barriere de capital
bars = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
cps = [M.coupon_autocall(ch, T, r, nominal, n_obs, 1.0, 0.7, b, True, 0.008) * 400 for b in bars]
png(G.courbes(bars * 100, [{"nom": "Coupon annuel de l'autocall", "y": cps, "couleur": G.PRODUIT, "points": True}],
              "Plus de protection, moins de coupon",
              "Autocall 3 ans, trigger 100, barrière de coupon 70 : le coupon selon la barrière de capital",
              "Barrière de capital (% du départ)", "Coupon annuel (%)", fmt_x=lambda v: f"{v:.0f} %",
              fmt_y=lambda v: G.pct(v, 2), legende=False, hauteur=300), "arbitrage.png")
