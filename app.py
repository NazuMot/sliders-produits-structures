"""
=======================================================================
SLIDERS PRODUITS STRUCTURES  -  v4
Tom Uzan
=======================================================================

Interface. Le pricing est dans moteur.py.

Lancer avec :  streamlit run app.py
=======================================================================
"""

import streamlit as st

from barre import barre_laterale
from vues import (accueil, comparateur, tracker, capital_protection, brc, autocall,
                  bonus, twin_win, laboratoire, grecques, couverture, glossaire)
from vues import entrainement

st.set_page_config(page_title="Sliders Produits Structures", layout="wide")

# Pour ajouter une page : un fichier dans vues/ avec une fonction affiche(m),
# puis une entree ici. L'ordre du dictionnaire est celui du menu.
PAGES = {
    "Accueil": accueil,
    "Comparateur des produits": comparateur,
    "1 - Tracker Certificate": tracker,
    "2 - Capital Protection": capital_protection,
    "3 - Barrier Reverse Convertible": brc,
    "4 - Autocall Phoenix": autocall,
    "5 - Bonus Certificate": bonus,
    "6 - Twin-Win": twin_win,
    "Laboratoire modele": laboratoire,
    "Les grecques en mouvement": grecques,
    "Cout de couverture": couverture,
    "Entrainement": entrainement,
    "Glossaire": glossaire,
}

page = PAGES[st.sidebar.selectbox("Page", list(PAGES.keys()), index=0,
                                  key="page", bind="query-params")]
m = barre_laterale()
page.affiche(m)


# =====================================================================
# =====================================================================
# PIED DE PAGE
# =====================================================================

st.markdown("---")
with st.expander("Limites du modele"):
    st.markdown("""
### Traite

**Volatilite stochastique et sauts.** Le modele de Bates genere le skew au lieu de le
postuler, via la correlation spot-vol, et produit de vrais gaps a la baisse. C'est le
standard de marche pour les exotiques actions. Compare les deux modeles dans le
Laboratoire : l'ecart sur le coupon d'un BRC depasse regulierement 2 points.

**Barrieres continues.** Correction par pont brownien, generalisee a une volatilite
variable. Evite le biais classique des simulations discretes, qui sous-estiment
systematiquement la probabilite de franchissement.

**Risque emetteur.** Intensite de defaut deduite du spread de credit, avec taux de
recouvrement. Le spread augmente le coupon, ce qui est contre-intuitif et essentiel a
comprendre en clientele.

**Paniers worst-of avec correlation.** Decomposition de Cholesky. La correlation devient
le parametre de pricing dominant, comme sur un vrai desk.

**Cout de couverture.** Simulation de la couverture en delta avec rebalancement discret
et frais de transaction. Montre que la marge n'est pas un simple prelevement.

### Limites

**1. Parametres regles a la main, pas calibres.**
C'est la difference principale avec un pricer de production. Un desk recalibre Bates
chaque matin sur les prix d'options cotees. Ici tu choisis les parametres au curseur : le
modele est juste, l'etalonnage est arbitraire. Les niveaux sont donc indicatifs, pas des
prix de marche.

**2. Worst-of simule en Black-Scholes multi-actifs.**
Un Bates multi-dimensionnel demanderait de correler aussi les volatilites entre actifs.
Le mecanisme de correlation est juste, la dynamique de vol ne l'est pas.

**3. Correlation constante.**
En pratique elle monte fortement dans les krachs, exactement quand ca fait mal sur un
worst-of. Le modele sous-estime donc le risque des paniers en periode de stress.

**4. Pas de wrong-way risk.**
Le defaut de l'emetteur est simule independamment du sous-jacent. Dans la realite une
banque fait defaut au pire moment, quand les marches s'effondrent.

**5. Discretisation hebdomadaire de la variance sous Bates.**
Euler a troncature complete introduit un biais modeste sur le processus de variance. Un
schema de type QE (Andersen) serait plus propre pour des maturites longues.

**6. Couverture simulee sur un put vanille, pas sur un put a barriere.**
Sur un vrai BRC le gamma explose autour de la barriere. Le cout affiche est un plancher.
""")

st.caption("Tom Uzan")
