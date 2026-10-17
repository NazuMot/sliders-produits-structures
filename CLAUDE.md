# Sliders Produits Structures — instructions pour Claude Code

## Le propriétaire

Tom Uzan, étudiant EDHEC BBA Finance, vise Sales en produits structurés sur desk institutionnel.
**Tom ne code pas.** Toutes les modifications sont faites par Claude, de bout en bout : écrire,
tester, committer, pousser. Expliquer en français, direct, sans jargon informatique inutile.
Expliquer toujours le *pourquoi financier* d'un changement, pas seulement le code.

## Ce qu'est l'appli

Application Streamlit qui calcule ce qu'un desk peut offrir sur six produits structurés à partir
des conditions de marché. **Principe central : rien n'est choisi à la main.** Participation et
coupons sont des *résultats* dérivés du budget disponible, comme le ferait un structureur.

Déployée sur Streamlit Community Cloud (plan gratuit, ~1 Go de RAM) depuis le dépôt GitHub de
Tom (compte `NazuMot`). **Tout push sur `main` redéploie automatiquement le site.**

## Fichiers

| Fichier | Rôle |
|---|---|
| `app.py` | Point d'entrée : menu `PAGES`, appel de la page choisie, pied de page |
| `barre.py` | Barre latérale : `barre_laterale()` renvoie tous les réglages de marché (objet `m`) |
| `communs.py` | Couleurs, `MAX_CACHE`, simulations mises en cache (`sim_bs`, `sim_bates`…) |
| `vues/*.py` | Une page par fichier, chacune avec une fonction `affiche(m)` |
| `moteur.py` | Tout le pricing (aucun `import streamlit` ici) |
| `graphiques.py` | Style et primitives de tracé matplotlib |
| `requirements.txt` | `streamlit`, `numpy`, `matplotlib`, versions figées — rien d'autre |
| `tests/verifie.py` | Vérification complète (toutes les pages, deux modèles, chiffres de référence) |
| `.github/workflows/tests.yml` | Relance `tests/verifie.py` sur GitHub à chaque push |

## Pages (sélecteur en haut de la barre latérale, dictionnaire `PAGES` dans `app.py`)

Accueil · Comparateur des produits · 1 Tracker · 2 Capital Protection · 3 Barrier Reverse
Convertible · 4 Autocall Phoenix (avec worst-of 1 à 4 sous-jacents + corrélation) · 5 Bonus
Certificate · 6 Twin-Win · Laboratoire modèle (Black-Scholes vs Bates) · Les grecques en
mouvement · Coût de couverture · Glossaire (6 onglets).

Pour ajouter une page : un fichier `vues/<nom>.py` avec `def affiche(m):`, puis une entrée dans
`PAGES` (`app.py`). En tête de `affiche`, la page déballe les réglages dont elle a besoin
(`r, q = m.r, m.q`). Ne pas toucher aux autres pages. Les tests lisent `PAGES` : une nouvelle page
est testée automatiquement.

## Barre latérale (vue de marché unique pour toutes les pages)

Modèle (Black-Scholes + skew / Bates), taux, dividende, vol ATM, skew (ou 8 paramètres Bates),
spread émetteur + taux de recouvrement, marge banque, nominal, nombre de trajectoires.

## Moteur — ce qui est validé, ne pas casser

- `simule_bs` : GBM, pas hebdomadaires, **float32**, variables antithétiques, drift `r - q`
  (risque-neutre).
- `simule_bates` : Heston + sauts Merton, Euler à troncature complète, renvoie `(chemins, sigma_pas)`.
- `simule_panier` : worst-of par Cholesky, corrélation uniforme.
- `proba_non_touche` : **pont brownien** pour les barrières continues, accepte un sigma scalaire
  ou une matrice. Validé contre simulation dense (23,96 % vs 23,52 %).
- `tire_defaut` : intensité `lambda = spread / (1 - recovery)`. Le spread **augmente** le coupon.
- `coupon_brc`, `coupon_autocall` : payoff affine en coupon → résolution directe.
- `put_down_and_in` : formule fermée Reiner-Rubinstein. Cas `K > H` → `B - C + D` ;
  cas `K <= H` → put vanille. **Validé à 1 % près contre Monte Carlo sur 144 combinaisons**,
  et identité `DI + DO = put vanille` vérifiée. Si on y touche, revalider.
- `grecques` : différences finies sur formule fermée (delta, gamma, vega /pt, theta /jour, rho /pt).

Chiffres de référence (paramètres par défaut) pour détecter une régression, contrôlés par
`tests/verifie.py` :
- Capital garanti 5 ans, r = 3 %, marge banque 0,8 %/an : participation ≈ 55 %
  (l'ancien chiffre de 69 % correspondait à une marge de 0,3 %/an)
- BRC 1 an barrière 70 % : coupon ≈ 5,9 % sous BS plat, ≈ 8,7 % sous Bates
- Autocall worst-of 3 sous-jacents : coupon et proba de perte montent quand la corrélation baisse

## Contraintes techniques

- **Mémoire** : Streamlit Cloud gratuit plante au-delà de ~1 Go. Garder float32, pas
  hebdomadaires, `@st.cache_data(max_entries=6)`. Une version précédente plantait en simulant
  quotidiennement en float64.
- **Vitesse** : les curseurs doivent réagir instantanément. Les calculs lourds (tableaux de
  sensibilité) sont derrière une case à cocher et en cache.
- Toujours `plt.close(fig)` après `st.pyplot(fig)`.
- Pas de dépendance nouvelle sans raison forte (pas de scipy : `norm_cdf_vec` existe).
- Pas d'API payante : Tom a refusé l'idée d'un assistant IA intégré.

## Conventions de contenu — demandes explicites de Tom

- **Devise : EUR**, jamais CHF.
- **Aucune formule de précaution** du type « outil pédagogique, aucune valeur d'offre ».
- **Titres courts et secs.** Pas de titres à rallonge ni de commentaires dans les titres
  (« Limites du modèle », pas « Limites qui restent, et elles sont réelles »).
- **Pas d'objectif de carrière affiché** : le nom seul, « Tom Uzan ».
- Garder la section « Limites du modèle » : c'est du contenu technique, pas une précaution.
- Chaque curseur a un `help=` ; chaque `st.title` a un `help=` long (ce que c'est, comment c'est
  construit, pourquoi un client l'achète, quoi regarder). Tout nouvel élément doit suivre ça.
- Textes en français sans accents dans le code (existant), en restant cohérent avec le fichier.

## Conventions graphiques (`graphiques.py`)

Une couleur = un seul sens dans toute l'appli :
- bleu `PRODUIT` = le produit structuré
- gris tireté `DIRECT` = détention directe
- vert `GAIN` / rouge `PERTE` = **uniquement** zones de gain/perte relatives (et barrières en rouge)
- gris `NIVEAU` = repères neutres

Surface claire `SURFACE`, pas de fond blanc par défaut (l'appli peut être en thème sombre).
Profils de payoff via `G.payoff(...)` : zones colorées, point mort, bande de densité Monte Carlo
sous l'axe (« où le sous-jacent finit réellement »). Titres de graphes = ce que le graphe
*montre* (« 41 % des trajectoires franchissent la barrière »), pas ce qu'il contient.

## Procédure pour chaque modification

1. Modifier.
2. Lancer `python tests/verifie.py` (sur une copie à part) : toutes les pages sous les **deux**
   modèles, cases décochées puis cochées, plus les chiffres de référence. Aucune exception
   tolérée. Une nouvelle fonctionnalité derrière un réglage s'ajoute au script.
3. Si un chiffre de référence change volontairement : mettre à jour le script et ce fichier,
   en expliquant pourquoi.
4. Après le push, GitHub relance le même script (onglet Actions du dépôt) : vérifier qu'il est vert.
5. Si un graphe a changé : le rendre en PNG et le regarder avant de pousser.
6. Commit clair en français, push sur `main`. Dire à Tom en une phrase ce qui a changé et
   qu'il peut recharger le site d'ici 1 à 2 minutes.

## Limites connues (affichées dans l'appli)

Paramètres réglés au curseur, non calibrés sur des prix d'options cotées · worst-of simulé en
Black-Scholes multi-actifs · corrélation constante · pas de wrong-way risk · discrétisation
hebdomadaire de la variance sous Bates · coût de couverture simulé sur put vanille.

## Pistes déjà évoquées, non faites

- Recherche dans le glossaire (gratuit, simple).
- Calibration de Bates sur des prix d'options réels.
- Corrélation qui monte en période de stress pour le worst-of.
