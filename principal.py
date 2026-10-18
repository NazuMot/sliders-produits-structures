"""
Menu des pages, page choisie, pied de page.
Traduit en anglais a la volee par traduction.py (module en_principal).
"""

import streamlit as st

from barre import barre_laterale
from vues import (accueil, comparateur, atelier_sales, tracker, capital_protection, brc,
                  autocall, bonus, twin_win, discount, backtest, laboratoire, grecques,
                  couverture, entrainement, glossaire)

# Pour ajouter une page : un fichier dans vues/ avec une fonction affiche(m),
# puis une entree ici. L'ordre du dictionnaire est celui du menu.
PAGES = {
    "Accueil": accueil,
    "Comparateur des produits": comparateur,
    "Atelier Sales": atelier_sales,
    "1 - Tracker Certificate": tracker,
    "2 - Capital Protection": capital_protection,
    "3 - Barrier Reverse Convertible": brc,
    "4 - Autocall Phoenix": autocall,
    "5 - Bonus Certificate": bonus,
    "6 - Twin-Win": twin_win,
    "7 - Discount Certificate": discount,
    "Backtest historique": backtest,
    "Laboratoire modèle": laboratoire,
    "Les grecques en mouvement": grecques,
    "Coût de couverture": couverture,
    "Entraînement": entrainement,
    "Glossaire": glossaire,
}


def lance():
    """Affiche la page choisie dans le menu, puis le pied de page."""
    page = PAGES[st.sidebar.selectbox("Page", list(PAGES.keys()), index=0,
                                      key="page", bind="query-params")]
    m = barre_laterale()
    page.affiche(m)

    st.markdown("---")
    with st.expander("Limites du modèle"):
        st.markdown("""
### Traite

**Volatilité stochastique et sauts.** Le modèle de Bates génère le skew au lieu de le
postuler, via la corrélation spot-vol, et produit de vrais gaps à la baisse. C'est le
standard de marché pour les exotiques actions. Compare les deux modèles dans le
Laboratoire : l'écart sur le coupon d'un BRC dépasse régulièrement 2 points.

**Barrières continues.** Correction par pont brownien, généralisée à une volatilité
variable. Évite le biais classique des simulations discrètes, qui sous-estiment
systématiquement la probabilité de franchissement.

**Risque émetteur.** Intensité de défaut déduite du spread de crédit, avec taux de
recouvrement. Le spread augmente le coupon, ce qui est contre-intuitif et essentiel à
comprendre en clientèle.

**Paniers worst-of avec corrélation.** Décomposition de Cholesky. La corrélation devient
le paramètre de pricing dominant, comme sur un vrai desk.

**Coût de couverture.** Simulation de la couverture en delta avec rebalancement discret
et frais de transaction. Montre que la marge n'est pas un simple prélèvement.

### Limites

**1. Calibrage sur une seule photo de marché.**
Le Laboratoire calibre Bates sur 122 prix Eurex du 16/10/2026 (0,9 point de vol d'erreur
moyenne aux crans des curseurs). Un desk recalibre chaque matin ; ici les curseurs restent
libres et la photo vieillit. Les niveaux sont donc indicatifs, pas des prix de marché.

**2. Worst-of simulé en Black-Scholes multi-actifs.**
Un Bates multi-dimensionnel demanderait de corréler aussi les volatilités entre actifs.
Le mécanisme de corrélation est juste, la dynamique de vol ne l'est pas.

**3. Corrélation de stress simplifiée.**
Sur l'Autocall, une option fait monter la corrélation quand le marché baisse, mais en deux
régimes seulement, et sans que la volatilité monte en même temps.

**4. Wrong-way risk seulement sur demande.**
Par défaut, le défaut de l'émetteur est indépendant du sous-jacent. La page BRC propose une
corrélation (copule gaussienne sur le niveau final) : elle montre l'effet, sans modèle de
crédit complet.

**5. Discrétisation hebdomadaire de la variance sous Bates.**
Euler à troncature complète introduit un biais modeste sur le processus de variance. Un
schéma de type QE (Andersen) serait plus propre pour des maturités longues.

**6. Couverture simulée sur un put vanille, pas sur un put à barrière.**
Sur un vrai BRC le gamma explose autour de la barrière. Le coût affiché est un plancher.
""")

    st.caption("Tom Uzan")
