"""Accueil."""

import streamlit as st


def affiche(m):
    st.title("Sliders Produits Structurés", help="**Ce que fait cet outil.** Il calcule ce qu'un desk peut offrir sur sept produits structurés,\nà partir des conditions de marché que tu règles à gauche.\n\n**Le point essentiel.** Rien n'est choisi à la main : la participation d'un capital garanti\net le coupon d'un BRC ou d'un autocall sont des *résultats*, pas des paramètres. Ils\ndécoulent du budget disponible, exactement comme sur un desk.\n\n**Par où commencer.** Le Comparateur pour la vue d'ensemble, puis les pages 1 à 7 dans\nl'ordre : chaque produit ajoute une brique au précédent.")
    st.markdown("Tom Uzan")

    st.markdown("---")
    st.markdown("""
### À quoi sert cet outil

Comprendre les produits structurés en manipulant leurs paramètres plutôt qu'en lisant
des formules. **Rien n'est choisi à la main** : la participation d'un capital garanti et
le coupon d'un BRC ou d'un autocall sont calculés à partir du budget disponible, comme
le ferait un structureur. Quand tu bouges les taux, la volatilité ou la barrière, tu vois
ce que le desk peut réellement offrir ce jour-la.

### Par où commencer

| Page | Ce qu'on y voit |
|---|---|
| **Comparateur** | Les structures principales côte à côte, sous la même vue de marché |
| **Atelier Sales** | Partir de la demande du client, voir où va l'argent, revendre en cours de vie, pitcher |
| **1 à 7** | Un produit par page, avec son profil de remboursement et ses trajectoires |
| **Backtest historique** | Ce qu'aurait donne le produit, émis chaque mois sur l'Euro Stoxx 50 depuis 2000 |
| **Laboratoire modèle** | Ce que change le choix du modèle, et le calibrage sur de vrais prix Eurex |
| **Les grecques en mouvement** | Ce que porte le desk jour après jour, et ce qu'il doit couvrir |
| **Coût de couverture** | Pourquoi la marge n'est pas un simple prélèvement |
| **Entraînement** | 40 questions d'entretien de Sales, avec réponse rédigée et chiffres |
| **Glossaire** | Tout le vocabulaire, avec ce que chaque terme implique pour un Sales |

Tous les paramètres de marché sont dans la barre latérale à gauche et alimentent
l'ensemble des produits ; un scénario (2008, Covid, taux zéro...) les remplit en un clic.
Chaque curseur a une infobulle qui explique son effet. Survole les graphes pour lire les
valeurs, et copie l'adresse du site pour partager exactement ce que tu vois.

### Ce qu'il y a sous le capot

- **Deux modèles** : Black-Scholes avec skew imposé, et Bates (volatilité stochastique
  de Heston + sauts de Merton), qui génère le skew au lieu de le postuler.
- **Barrières continues traitées par pont brownien**, ce qui évite le biais classique
  des simulations discrètes.
- **Risque émetteur** : intensité de défaut déduite du spread de crédit, avec taux de
  recouvrement.
- **Paniers worst-of** avec corrélation, y compris une corrélation qui monte dans les
  krachs, et indice à décrément, sur la page Autocall.
- **Données réelles** : historique de l'Euro Stoxx 50 depuis 1986 et 122 prix d'options
  Eurex, sur lesquels Bates est calibré.
- **Couverture en delta** simulée avec rebalancement discret et frais de transaction.

### Ce que ce n'est pas

Pas un pricer de production : le calibrage repose sur une seule photo de marché
(16/10/2026), alors qu'un desk recalibre chaque matin, et les curseurs restent libres. Les
niveaux affichés sont des ordres de grandeur, pas des prix de marché. La liste complète des
limites est en bas de chaque page.
""")
