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
| `app.py` | Point d'entrée : bouton English, puis `principal` (ou sa version anglaise) |
| `principal.py` | Menu `PAGES`, appel de la page choisie, pied de page (fonction `lance()`) |
| `traduction.py` | Version anglaise à la volée : relit chaque module et remplace ses textes via `traductions/en.json` |
| `traductions/en.json` | Dictionnaire « texte français → texte anglais » (1 444 textes) |
| `scripts/traductions.py` | Liste les textes sans traduction (`--export fichier` pour les traduire) |
| `barre.py` | Barre latérale : `barre_laterale()` renvoie tous les réglages de marché (objet `m`) |
| `communs.py` | Couleurs, `MAX_CACHE`, simulations mises en cache (`sim_bs`, `sim_bates`…) |
| `vues/*.py` | Une page par fichier, chacune avec une fonction `affiche(m)` |
| `moteur.py` | Tout le pricing (aucun `import streamlit` ici) |
| `graphiques.py` | Graphes interactifs Altair : `payoff`, `courbes`, `trajectoires`, `barres`, `histogramme`, `affiche` |
| `requirements.txt` | `streamlit`, `numpy`, `altair` (fourni avec Streamlit), versions figées — rien d'autre |
| `sales.py` | Outils du Sales : reverse enquiry, où va l'argent, marché secondaire (calcul seul) |
| `backtest.py`, `calibration.py`, `data/` | Données réelles : backtest Euro Stoxx 50, calibrage Bates sur Eurex |
| `scripts/recupere_donnees.py` | Refait la photo des données (jamais appelé par l'appli) |
| `tests/verifie.py` | Vérification complète (toutes les pages, deux modèles, chiffres de référence) |
| `tests/valide_*.py` | Validation chiffrée du moteur, de `sales.py` et des données réelles |
| `.github/workflows/tests.yml` | Relance les quatre scripts de tests sur GitHub à chaque push |

## Pages (menu par étapes en haut de la barre latérale, `PAGES` et `ETAPES` dans `principal.py`)

Accueil · Comparateur des produits · Atelier Sales · 1 Tracker · 2 Capital Protection ·
3 Barrier Reverse Convertible (+ case wrong-way risk) · 4 Autocall Phoenix (worst-of 1 à 4
sous-jacents, step-down, indice à décrément, corrélation de stress, grecques du desk) · 5 Bonus
Certificate · 6 Twin-Win · 7 Discount Certificate · Backtest historique · Laboratoire modèle
(Black-Scholes vs Bates, calibrage Eurex) · Les grecques en mouvement · Coût de couverture ·
Entraînement (40 questions d'entretien de Sales en 5 thèmes, réponse rédigée + démonstration
chiffrée, mode hasard, compteur) · Glossaire (6 onglets + recherche insensible aux accents).

Le détail des pages ajoutées le 18/10/2026 est dans la section « Pages enrichies » plus bas.

Menu (demande de Tom, 18/10/2026) : une liste de boutons par étape, `ETAPES` = Commencer ·
Comprendre les produits · Vendre et s'entraîner · Côté desk. La page affichée est dans
`st.session_state.page` et dans l'adresse (`?page=...`, liens partagés). Les tests choisissent une
page en cliquant le bouton du menu : `next(r for r in at.sidebar.radio if page in r.options)`, et
le modèle par `at.sidebar.radio(key="modele")`.

Pour ajouter une page : un fichier `vues/<nom>.py` avec `def affiche(m):`, puis une entrée dans
`PAGES` **et** dans une étape de `ETAPES` (`principal.py`). En tête de `affiche`, la page déballe les réglages dont elle a besoin
(`r, q = m.r, m.q`). Ne pas toucher aux autres pages. Les tests lisent `PAGES` : une nouvelle page
est testée automatiquement.

## Barre latérale (vue de marché unique pour toutes les pages)

Conditions de marché en un clic (`SCENARIOS` dans `barre.py` : défaut, octobre 2026, 2017,
taux zéro 2020-2021, Covid, automne 2008, 2022-2023 ; vols d'après le VSTOXX de `data/`). Le
sélecteur est une action : il écrit les curseurs via `st.session_state` puis revient à vide ; ce
sont les curseurs, liés au lien, qui gardent les valeurs. `.streamlit/config.toml` coupe
l'avertissement de Streamlit sur ces écritures, voulues. Puis : modèle (Black-Scholes + skew /
Bates), taux, dividende, vol ATM, skew (ou 8 paramètres Bates), spread émetteur + taux de
recouvrement, marge banque, nominal, nombre de trajectoires.

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
- `simule_lv` (18/10/2026) : **Black-Scholes + skew en volatilité locale**,
  `sigma(S,t) = vol_atm + 1,9 × skew × ln(F(t)/S) / 0,10` (bornée 5-150 %), renvoie
  `(chemins, sigma_pas)` pour le pont brownien, gère le décrément. C'est ce qu'utilisent
  `barre.trajectoires` et `sales.Marche.trajectoires` pour tous les produits à trajectoire
  mono sous-jacent. Avant, une seule vol par produit lue au niveau de la barrière inversait le
  sens (coupon qui monte quand la barrière baisse). Elle redonne le skew affiché à moins d'un
  point (skew 2 : 25,9 / 23,9 / 22,0 % aux strikes 80 / 90 / 100 %). skew 0 = `simule_bs` au bit
  près. Restent en vol unique au bon strike : les formules fermées (capital garanti, Discount,
  grecques en mouvement) et les worst-of (`simule_panier`, vol au niveau de la barrière).
- `coupon_autocall(..., marge_sur="vie")` (18/10/2026) : la marge annuelle est prélevée sur la
  **durée de vie attendue**, pas sur la maturité maximale (`marge_sur="maturite"` = ancien
  calcul). Trigger 80 % : coupon 0 % -> 4,28 % p.a. `sales.py` compte marge et rétro de
  l'autocall de la même façon.

- Ajouts optionnels (valeur par défaut = ancien résultat au bit près, contrôlé par
  `tests/valide_moteur_produits.py`) :
  - step-down : `trigger` scalaire ou liste (un niveau par observation, `math.inf` = pas de
    rappel), `trigger_degressif` ;
  - indice à décrément : `simule_bs/simule_bates(..., decrement=D)`, D en points/an base 100,
    q = 0 ; `compare_decrement` sépare effet niveau et effet points ;
  - Discount Certificate : `discount_certificate` (formule fermée), `discount_mc` ;
  - corrélation de stress : `simule_panier(..., correlation_stress, seuil_stress)`. Elle FAIT
    BAISSER coupon et proba de perte du worst-of (moins de dispersion) : skew de corrélation ;
  - wrong-way : `tire_defaut(..., correlation, chemins)` (copule gaussienne, rho = 0 = tirages
    actuels), `creance="valeur"`, `profil_brc` (perte dans les pires scénarios). L'effet est
    dans la queue, le coupon baisse légèrement ;
  - `grecques_autocall` : grecques du desk avec les mêmes tirages (0,03 à 0,4 s) ; le desk est
    long vega, long dividendes, court corrélation, theta négatif. Derrière une case + cache.
- Chiffres de référence : autocall 3 ans trimestriel (BS + skew, q = 2,5 %) coupon 8,75 % p.a. ;
  décrément 5 points 12,91 % ; worst-of 3, corrélation 60 % : 16,7 % -> 13,0 % avec 90 % en
  stress ; BRC banque, wrong-way 0,6 : pire 1 % 68,9 -> 81,1 % du nominal.

Chiffres de référence (paramètres par défaut) pour détecter une régression, contrôlés par
`tests/verifie.py` :
- Capital garanti 5 ans, r = 3 %, marge banque 0,8 %/an : participation ≈ 55 %
- Par défaut (BS + skew 2, vol locale) : BRC 1 an barrière 70 % 7,84 % p.a. ; autocall 3 ans
  trimestriel 100/70/60 9,28 % p.a., vie moyenne 0,92 an, perte 8,3 %
  (l'ancien chiffre de 69 % correspondait à une marge de 0,3 %/an)
- BRC 1 an barrière 70 % : coupon ≈ 5,9 % sous BS plat, ≈ 8,7 % sous Bates
- Autocall worst-of 3 sous-jacents : coupon et proba de perte montent quand la corrélation baisse

## Outils du Sales (`sales.py`)

Module de calcul sans interface (aucun `import streamlit`), validé par `tests/valide_sales.py`.
Vue de marché : `Marche.depuis_barre(m, retro=...)`, les réglages de la barre latérale plus la
rétrocession distributeur. Avec rétro nulle, ses coupons redonnent **exactement** ceux des pages
(mêmes trajectoires, même graine) : si une page change sa façon d'appeler le moteur, adapter
`sales.py`.
- Reverse enquiry : `brc_barriere`, `brc_strike`, `brc_maturite`, `autocall_barriere_capital`,
  `autocall_trigger`, `cp_protection`, `cp_cap` -> `Reponse` (possible, valeur, obtenu, message
  prêt à dire au client, courbe à tracer). Hors d'atteinte : « impossible » + meilleure valeur.
- Où va l'argent : `cascade_brc`, `cascade_autocall`, `cascade_cp` -> lignes dont la somme vaut le
  nominal au centime ; `infos["cout_retro_par_pt"]` = coupon perdu par 1 % de rétro par an.
- Marché secondaire : `valeur_secondaire_brc / _autocall / _cp` (+ `etat_autocall`) -> valeur,
  prix de rachat (fourchette 1 % par défaut), décomposition temps / sous-jacent / vol / taux /
  spread. Référence : à t = 0, valeur = nominal - marge - rétro (982 EUR, BRC 1 an).
- 0,02 à 0,4 s par appel : derrière un bouton et en cache (`Marche` est hachable).

## Données réelles (`data/`, `backtest.py`, `calibration.py`)

- `data/eurostoxx50.csv` : Euro Stoxx 50 quotidien 1986-2026 (prix SX5E, net_return SX5T,
  vstoxx). Sources STOXX officiel + Yahoo + ETF SXRT.DE pour le net return 2016-2026 (dérive
  0,48 %/an retirée). `data/options_eurostoxx50.csv` : 122 prix de règlement Eurex OESX du
  16/10/2026. `data/bates_calibre.json` : paramètres calibrés, réglage `curseurs`, erreurs.
- L'appli ne va jamais chercher de données en ligne. Refaire la photo :
  `python scripts/recupere_donnees.py` puis `python calibration.py`, puis mettre à jour les
  chiffres de référence de `tests/valide_donnees.py`.
- `backtest.py` : `charge_historique()`, `backtest_autocall(h, ...)`, `backtest_brc(h, ...)`
  (sous-jacent `prix`, `decrement` 50 pts base 1000 à l'émission, ou `net_return`), un résultat
  par émission + `.resume`. 5 ms. Ne compte que les produits terminés.
- `calibration.py` : `prix_bates` / `vol_bates` (formule de Lewis, 1 ms), `charge_parametres()`.
  Optimiseur Nelder-Mead maison, pas de scipy.
- Ne pas casser : Bates = Black-Scholes et Merton à 1e-4 près, = Monte Carlo de `simule_bates`
  à ~1 % ATM ; backtest = `pv_autocall` / `pv_brc` à 1e-9 sur les mêmes chemins.
- Chiffres de référence : autocall 3 ans trimestriel 100/70/60 mémoire 8 % émis chaque mois
  depuis 2000 : 286 émissions, 57,7 % rappelés à la 1re date, 3,5 % de pertes, pire cas émis le
  02/05/2000 (42,7 % rendu). BRC 1 an barrière 70 % continue 7 % : 15,8 % de pertes, pire cas
  03/03/2008. Calibrage : 0,77 pt de vol (réglage curseurs 0,91).
- Limites : net return 2016-2026 reconstitué via ETF, pas de SX5GT ; barrière continue =
  clôtures quotidiennes ; coupon fixe sur tout l'historique ; émissions qui se chevauchent ;
  saut moyen calibré sur la borne -30 % du curseur ; Feller violé (biais du pas hebdomadaire
  ~0,2 pt sur les calls hors de la monnaie).

## Pages enrichies

- **Atelier Sales** (`vues/atelier_sales.py`, clés `as_*`, testé par
  `tests/verifie_atelier_sales.py`) : met `sales.py` en scène face au client. Produit
  (BRC / Autocall / Capital Protection) et section choisis par `segmented_control` liés au lien ;
  seule la section choisie est calculée, chaque appel à `sales.py` en cache (`Structure` et
  `Marche` figées, hachables). Cadre Structure commun (curseurs du produit + rétro distributeur,
  1 %/an par défaut).
  - Le client demande : reverse enquiry, valeur exacte puis arrondie dans le sens qui garde la
    demande satisfaite (`arrondi_client`), phrase client ; « Retenir » écrit la valeur dans le
    curseur de la Structure (callback sur session_state).
  - Où va l'argent : cascade du budget (sources : taux, option vendue, risque émetteur ;
    emplois : coupons ou call, marge, rétro), deux réseaux de distribution, droite
    coupon = f(rétro).
  - Revendre : `valeur_secondaire_*` + cascade de l'émission au prix de rachat.
  - Pitch : `pitch(s, profil, vue, faits(mk, s))`, modèles de phrases, 3 à 5 phrases, toujours le
    scénario de perte chiffré et le risque émetteur.
  - Références : rétro nulle -> coupon BRC 7,84 % (= page) ; à t = 0, valeur 982 / 983 / 910
    EUR (BRC 1 an / autocall 3 ans, frais sur ~0,9 an de vie attendue / CP 5 ans, marge 0,8 %
    + rétro 1 %). Graphe en cascade construit dans la page (`graphe_cascade`).
- **Autocall, options** (`vues/autocall.py`, testé par `tests/verifie_autocall_discount.py`) :
  step-down (`ac_step_down`, `ac_sd_pas`, `ac_sd_plancher`, escalier tracé par `escalier()`) ;
  indice à décrément (`ac_indice`, `ac_dec_points`, `ac_dec_niveau`, défaut 50 points sur 1 000
  = 5 %/an, mono sous-jacent) avec tableau classique / décrément en % / en points ; corrélation de
  stress (`ac_stress`, visible seulement en worst-of car AppTest refuse les cases grisées) ;
  grecques du desk (`ac_voir_grecques`, en cache). Garde-fou mémoire : `trajectoires_panier()`
  plafonne un worst-of à 12,5 millions de points et 20 000 trajectoires (4 sous-jacents / 6 ans /
  40 000 : pointe 951 -> 378 Mo). Références (BS + skew en vol locale, marge sur la vie
  attendue) : 9,28 % p.a. ; step-down 9,28 % (vie plus courte, moins de pertes) ; décrément
  50 pts/1 000 11,75 % ; worst-of 3 : le stress fait baisser coupon et perte.
- **7 Discount Certificate** (`vues/discount.py`, `dc_*`) : `discount_certificate` (vol au cap)
  sous BS, `discount_mc` sous Bates, hors risque émetteur. Référence : 1 an cap 100 : décote
  10,52 %, rendement max 11,76 %, point mort 89,48.
- **Backtest historique** (`vues/backtest.py`, clés `bt_*`, testé par `tests/verifie_backtest.py`) :
  onglets Autocall / BRC (mêmes curseurs que les pages produits + coupon fixe), sous-jacent prix /
  net return / décrément, première émission 1987-2020 (défaut 2000). Graphe : l'indice, un point
  par émission (vert rappelé ou barrière intacte, gris terme ou touchée sans perte, rouge perte),
  survol = détail + vie du produit. Comparaison avec le Monte Carlo de la vue de marché. Onglets
  `st.tabs(on_change=...)` : seul l'onglet ouvert calcule ; l'onglet est recopié à la main dans le
  lien (`bt_onglet`). AppTest ne sait pas cliquer un onglet : `session_state["bt_onglet"]`.
- **Laboratoire, section 5 Calibrage sur le marché** (`labo_echeance`, bouton `labo_appliquer`) :
  smile Eurex contre Bates calibré et Black-Scholes plat, erreurs, 8 paramètres. Le bouton écrit
  `st.session_state` (modele, vol, vol_lt, kappa, vol_vol, rho, sauts, taille_saut, disp_saut)
  dans un on_click, et `st.query_params` pour les curseurs Bates pas encore affichés.
- Graphes construits dans les pages, à remonter un jour dans `graphiques.py` s'ils servent
  ailleurs : `graphe_cascade` (Atelier Sales), `_graphe_emissions` et `_graphe_repartition`
  (Backtest), `_graphe_smile` (Laboratoire), `escalier()` (Autocall).

## Version anglaise (`traduction.py`)

Bouton « English » en haut de la barre latérale (clé `anglais`, liée au lien). Le code français
reste la seule source : en anglais, `traduction.py` relit chaque module du projet (`principal`,
`barre`, `graphiques`, `sales`, `backtest`, `vues/*`), remplace chaque texte affiché par sa
traduction de `traductions/en.json`, et l'exécute sous un autre nom (`en_vues.brc`…). `moteur`,
`communs` et `calibration` ne sont pas traduits (pas de texte affiché).
- Ce qui est traduit : les mêmes chaînes que celles qu'on accentue (pas les docstrings, ni les
  `key=`, ni les identifiants, couleurs et champs Altair). f-strings : la clé est le gabarit,
  chaque `{expression}` devient `{0}`, `{1}`… et les textes cités dans une expression sont
  traduits aussi.
- Un texte absent du dictionnaire reste en français. **Après tout ajout ou changement de texte :
  `python scripts/traductions.py` doit dire « 0 à traduire »** ; sinon `--export f`, traduire,
  ajouter à `traductions/en.json`. Changer un texte français change sa clé : reporter la
  traduction sur la nouvelle clé. Les noms de produits, « Delta », « Strike »… sont des entrées
  identiques (« X » → « X »).
- Les options des menus sont traduites : une comparaison `== "Marché calme"` reste juste, car les
  deux chaînes sont traduites pareil. À la bascule, `traduction.bascule()` traduit les réglages
  texte (page choisie, options) pour que la page reste la même.
- Testé par `tests/verifie_anglais.py` : toutes les pages en anglais sous les deux modèles, cases
  cochées, bascule FR/EN, et aucun texte français visible dans titres, métriques et légendes.

## Contraintes techniques

- **Mémoire** : Streamlit Cloud gratuit plante au-delà de ~1 Go. Garder float32, pas
  hebdomadaires, `@st.cache_data(max_entries=6)`. Une version précédente plantait en simulant
  quotidiennement en float64.
- **Vitesse** : les curseurs doivent réagir instantanément. Les calculs lourds (tableaux de
  sensibilité) sont derrière une case à cocher et en cache.
- Graphes : toujours via `graphiques.py` et `G.affiche(...)`. Plus de matplotlib dans l'appli.
- Pas de dépendance nouvelle sans raison forte (pas de scipy : `norm_cdf_vec` existe).
- Pas d'API payante : Tom a refusé l'idée d'un assistant IA intégré.

## Conventions de contenu — demandes explicites de Tom

- **Devise : EUR**, jamais CHF.
- **Aucune formule de précaution** du type « outil pédagogique, aucune valeur d'offre ».
- **Titres courts et secs.** Pas de titres à rallonge ni de commentaires dans les titres
  (« Limites du modèle », pas « Limites qui restent, et elles sont réelles »).
- **Pas d'objectif de carrière affiché** : le nom seul, « Tom Uzan ».
- Garder la section « Limites du modèle » : c'est du contenu technique, pas une précaution.
- **Lien partageable** (demande de Tom : le lien rouvre le site comme on l'a laissé) : chaque
  widget a une `key` courte et unique (préfixe de page, ex. `brc_barriere`) et
  `bind="query-params"`. Tout nouveau widget suit cette règle, sinon son réglage ne passe pas
  dans le lien.
- Chaque curseur a un `help=` ; chaque `st.title` a un `help=` long (ce que c'est, comment c'est
  construit, pourquoi un client l'achète, quoi regarder). Tout nouvel élément doit suivre ça.
- **Textes affichés en français AVEC accents** (demande de Tom du 18/10/2026 : titres, aides,
  légendes, graphes, glossaire, questions, menu). Les commentaires et docstrings du code restent
  sans accents. Ne jamais accentuer les identifiants : clés de widgets (`key=`), clés de
  dictionnaires techniques, noms de colonnes des graphes (`"serie:N"`), couleurs (`"#1BAF7A"`).
  Changer un libellé change aussi la valeur dans les liens partagés et les tests qui le
  cherchent : mettre à jour les tests en même temps.

## Conventions graphiques (`graphiques.py`)

Graphes **interactifs** (Altair, déjà fourni avec Streamlit), demande explicite de Tom : « comme une
appli de banque ». Au survol, une ligne verticale suit la souris et **une infobulle donne toutes
les valeurs à cet endroit** (pas de nuage de points visibles). Les faisceaux de trajectoires
mettent en avant la trajectoire survolée et donnent son issue pour le client. Page grecques :
un **clic** sur un graphe place le curseur Jour sur ce point (`clic=True` + `G.affiche(...,
cle=..., sur_clic=...)` + `G.point_clique`). Toute nouvelle courbe suit ce modèle.

Une couleur = un seul sens dans toute l'appli :
- bleu `PRODUIT` = le produit structuré
- gris tireté `DIRECT` = détention directe
- vert `GAIN` / rouge `PERTE` = **uniquement** zones de gain/perte relatives (et barrières en rouge)
- gris `NIVEAU` = repères neutres

Pas de couleur de texte ni de fond imposée : le thème Streamlit les gère (clair et sombre).
Profils de payoff via `G.payoff(...)` : zones colorées, point mort, bande de densité Monte Carlo
sous l'axe (« où le sous-jacent finit réellement »), et dans l'infobulle la probabilité de finir
sous le niveau survolé. Titres de graphes = ce que le graphe *montre* (« 41 % des trajectoires
franchissent la barrière »), pas ce qu'il contient. Pour regarder un graphe avant de pousser :
le rendre en PNG avec `vl-convert-python` (outil de test, pas dans requirements.txt) ou ouvrir
l'appli en local.

Lisibilité (demande de Tom, 18/10/2026) : chaque page produit a, sous son titre,
`G.en_30_secondes(gagne, risque, pour_qui)` (texte fixe, lisible sans ouvrir l'aide du titre), et
sous son graphe de payoff `G.en_clair(...)` : une phrase en euros avec les réglages du moment
(« Pour 1 000 EUR investis sur 2 ans : ... »). Un nouveau produit suit ce modèle.

## Procédure pour chaque modification

1. Modifier.
2. Lancer `python tests/verifie.py`, les `tests/valide_*.py` et les `tests/verifie_*.py` (sur
   une copie à part ; ce sont ceux du workflow GitHub) : toutes les pages sous les **deux**
   modèles, cases décochées puis cochées, plus les chiffres de référence. Aucune exception
   tolérée. Une nouvelle fonctionnalité derrière un réglage s'ajoute au script.
3. Si un chiffre de référence change volontairement : mettre à jour le script et ce fichier,
   en expliquant pourquoi.
4. **Avant le push, changer la ligne « Publication du ... » de `requirements.txt`** (date et heure).
   Sinon Streamlit Cloud ne recharge que les fichiers nouveaux et garde en mémoire les anciennes
   versions des fichiers modifiés (constaté le 18/10/2026 : accueil et barre latérale restés
   anciens alors que les nouvelles pages étaient en ligne). Vérifier ensuite sur le site qu'un
   texte modifié apparaît bien.
5. Après le push, GitHub relance le même script (onglet Actions du dépôt) : vérifier qu'il est vert.
6. Si un graphe a changé : le rendre en PNG et le regarder avant de pousser.
6. Commit clair en français, push sur `main`. Dire à Tom en une phrase ce qui a changé et
   qu'il peut recharger le site d'ici 1 à 2 minutes.

## Limites connues (affichées dans l'appli)

Calibrage sur une seule photo de marché (16/10/2026) · worst-of simulé en Black-Scholes
multi-actifs · corrélation de stress en deux régimes, sans hausse de vol · wrong-way risk
seulement en option (copule sur le niveau final) · discrétisation hebdomadaire de la variance
sous Bates · coût de couverture simulé sur put vanille.

Défauts corrigés le 18/10/2026 : marge de l'autocall sur la maturité maximale, et vol unique
au niveau de la barrière sous Black-Scholes + skew (voir Moteur). Reste :
- les worst-of restent en Black-Scholes multi-actifs à vol unique (au niveau de la barrière) ;
- `sales.py` n'applique pas le plafond de trajectoires des worst-of (identique tant qu'il ne
  joue pas).

## Pistes déjà évoquées, non faites

- Version anglaise : faite le 18/10/2026 (voir « Version anglaise »).
- README : fait le 18/10/2026 (`README.md`, images `docs/*.png` régénérées par
  `scripts/captures_readme.py`, rendu `vl-convert-python` hors requirements).
- Remonter dans `graphiques.py` les graphes construits dans les pages.
