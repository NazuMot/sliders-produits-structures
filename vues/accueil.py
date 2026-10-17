"""Accueil."""

import streamlit as st


def affiche(m):
    st.title("Sliders Produits Structures", help="**Ce que fait cet outil.** Il calcule ce qu'un desk peut offrir sur six produits structures,\na partir des conditions de marche que tu regles a gauche.\n\n**Le point essentiel.** Rien n'est choisi a la main : la participation d'un capital garanti\net le coupon d'un BRC ou d'un autocall sont des *resultats*, pas des parametres. Ils\ndecoulent du budget disponible, exactement comme sur un desk.\n\n**Par ou commencer.** Le Comparateur pour la vue d'ensemble, puis les pages 1 a 6 dans\nl'ordre : chaque produit ajoute une brique au precedent.")
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
| **Comparateur** | Les quatre structures principales cote a cote, sous la meme vue de marche |
| **1 a 6** | Un produit par page, avec son profil de remboursement et ses trajectoires |
| **Laboratoire modele** | Ce que change le choix du modele de pricing : pres de 3 points de coupon |
| **Cout de couverture** | Pourquoi la marge n'est pas un simple prelevement |
| **Glossaire** | Tout le vocabulaire, avec ce que chaque terme implique pour un Sales |

Tous les parametres de marche sont dans la barre laterale a gauche et alimentent
l'ensemble des produits. Chaque curseur a une infobulle qui explique son effet.

### Ce qu'il y a sous le capot

- **Deux modeles** : Black-Scholes avec skew impose, et Bates (volatilite stochastique
  de Heston + sauts de Merton), qui genere le skew au lieu de le postuler.
- **Barrieres continues traitees par pont brownien**, ce qui evite le biais classique
  des simulations discretes.
- **Risque emetteur** : intensite de defaut deduite du spread de credit, avec taux de
  recouvrement.
- **Paniers worst-of** avec correlation, sur la page Autocall.
- **Couverture en delta** simulee avec rebalancement discret et frais de transaction.

### Ce que ce n'est pas

Les parametres sont regles au curseur, **pas calibres** sur des prix d'options cotees.
Un desk recalibre son modele chaque matin. Les niveaux affiches sont donc des ordres de
grandeur, pas des prix de marche. La liste complete des limites est en bas de chaque page.
""")
