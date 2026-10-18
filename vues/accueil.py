"""Accueil."""

import streamlit as st


def affiche(m):
    st.title("Sliders Produits Structures", help="**Ce que fait cet outil.** Il calcule ce qu'un desk peut offrir sur sept produits structures,\na partir des conditions de marche que tu regles a gauche.\n\n**Le point essentiel.** Rien n'est choisi a la main : la participation d'un capital garanti\net le coupon d'un BRC ou d'un autocall sont des *resultats*, pas des parametres. Ils\ndecoulent du budget disponible, exactement comme sur un desk.\n\n**Par ou commencer.** Le Comparateur pour la vue d'ensemble, puis les pages 1 a 6 dans\nl'ordre : chaque produit ajoute une brique au precedent.")
    st.markdown("Tom Uzan")

    st.markdown("---")
    st.markdown("""
### A quoi sert cet outil

Comprendre les produits structures en manipulant leurs parametres plutot qu'en lisant
des formules. **Rien n'est choisi a la main** : la participation d'un capital garanti et
le coupon d'un BRC ou d'un autocall sont calcules a partir du budget disponible, comme
le ferait un structureur. Quand tu bouges les taux, la volatilite ou la barriere, tu vois
ce que le desk peut reellement offrir ce jour-la.

### Par ou commencer

| Page | Ce qu'on y voit |
|---|---|
| **Comparateur** | Les structures principales cote a cote, sous la meme vue de marche |
| **Atelier Sales** | Partir de la demande du client, voir ou va l'argent, revendre en cours de vie, pitcher |
| **1 a 7** | Un produit par page, avec son profil de remboursement et ses trajectoires |
| **Backtest historique** | Ce qu'aurait donne le produit, emis chaque mois sur l'Euro Stoxx 50 depuis 2000 |
| **Laboratoire modele** | Ce que change le choix du modele, et le calibrage sur de vrais prix Eurex |
| **Les grecques en mouvement** | Ce que porte le desk jour apres jour, et ce qu'il doit couvrir |
| **Cout de couverture** | Pourquoi la marge n'est pas un simple prelevement |
| **Entrainement** | 40 questions d'entretien de Sales, avec reponse redigee et chiffres |
| **Glossaire** | Tout le vocabulaire, avec ce que chaque terme implique pour un Sales |

Tous les parametres de marche sont dans la barre laterale a gauche et alimentent
l'ensemble des produits ; un scenario (2008, Covid, taux zero...) les remplit en un clic.
Chaque curseur a une infobulle qui explique son effet. Survole les graphes pour lire les
valeurs, et copie l'adresse du site pour partager exactement ce que tu vois.

### Ce qu'il y a sous le capot

- **Deux modeles** : Black-Scholes avec skew impose, et Bates (volatilite stochastique
  de Heston + sauts de Merton), qui genere le skew au lieu de le postuler.
- **Barrieres continues traitees par pont brownien**, ce qui evite le biais classique
  des simulations discretes.
- **Risque emetteur** : intensite de defaut deduite du spread de credit, avec taux de
  recouvrement.
- **Paniers worst-of** avec correlation, y compris une correlation qui monte dans les
  krachs, et indice a decrement, sur la page Autocall.
- **Donnees reelles** : historique de l'Euro Stoxx 50 depuis 1986 et 122 prix d'options
  Eurex, sur lesquels Bates est calibre.
- **Couverture en delta** simulee avec rebalancement discret et frais de transaction.

### Ce que ce n'est pas

Pas un pricer de production : le calibrage repose sur une seule photo de marche
(16/10/2026), alors qu'un desk recalibre chaque matin, et les curseurs restent libres. Les
niveaux affiches sont des ordres de grandeur, pas des prix de marche. La liste complete des
limites est en bas de chaque page.
""")
