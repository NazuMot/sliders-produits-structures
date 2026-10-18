"""Glossaire, avec recherche dans les six onglets."""

import unicodedata

import streamlit as st


PARAMETRES = """
### Taux sans risque (r)
Carburant du budget option. Pour garantir 1 000 EUR dans 5 ans à 3%, la banque ne met
que 861 aujourd'hui. Les 139 restants achètent des options. À 0,5%, elle doit mettre 975
et il ne reste que 25.

**Conséquence commerciale** : taux hauts = capital garanti vendable. Taux bas = le marché
bascule vers BRC et autocalls.

### Taux de dividende (q)
Le détenteur d'un produit structuré ne le touche pas. Sur un tracker 3 ans avec 2,5% de
dividende, le client abandonne environ 7,8% de rendement sans le voir.

### Spread de crédit émetteur
Ce que la banque paie au-dessus du taux sans risque pour se financer. **Il augmente le
coupon** : la banque se finance moins cher via le produit structuré que sur le marché
obligataire, et reverse l'économie au client. Un coupon supérieur à la concurrence peut
donc simplement signifier un émetteur moins bien noté.

### Taux de recouvrement
Ce que le client récupère en cas de défaut. 40% est la convention senior non sécurisé.
Les porteurs de produits Lehman ont récupéré environ 9 cents par dollar.

### Marge banque
Prélevée sur le budget option, invisible dans le payoff. Divulguée réglementairement
(PRIIP, guidelines SSPA sur les coûts).

### Monte Carlo
Pricing par simulation : des milliers de trajectoires, le payoff dans chacune, la moyenne
actualisée. Indispensable dès qu'un produit dépend du chemin.

### Univers risque-neutre
Le sous-jacent monte en moyenne au taux (r - q), pas selon une prévision. **Ce n'est pas
un pari directionnel** : c'est la seule hypothèse qui interdit l'arbitrage.

### Pont brownien
Correction appliquée aux barrières. Une simulation à pas discrets rate les franchissements
entre deux observations : elle sous-estime la probabilité de toucher. Le pont brownien la
calcule analytiquement entre deux points. Plus rapide **et** plus juste qu'une simulation
quotidienne.
"""

MECANIQUES = """
### Strike
Niveau de référence qui décide du remboursement. Standard : 100% du spot initial.

### Barrière
- **Toucher une barrière ne fait pas perdre.** Ça arme le risque. C'est le strike qui
  décide à la fin.
- **Observation continue (américaine)** : la barrière compte à tout moment, même une
  mèche intraday. Standard marché.
- **Observation européenne** : seul le niveau à l'échéance compte. Plus favorable au
  client, donc coupon plus faible.

### Knock-in / Knock-out
- **Knock-in** : l'option s'active au franchissement. Le put d'un BRC.
- **Knock-out** : l'option disparaît au franchissement. Le put d'un Bonus ou d'un Twin-Win.

### Participation
Ratio de suivi. 70% : le sous-jacent fait +20%, le client touche +14%.
**Jamais choisie, toujours calculée** : budget option divisé par prix du call.

### Coupon
**Ce n'est pas un intérêt.** C'est le prix d'une option que le client vend. Trois
sources : les taux, la prime du put vendu, et le spread émetteur.

Coupon élevé = sinistre probable, ou émetteur risqué. Jamais bonne affaire.

### Coupon conditionnel vs inconditionnel
- **Inconditionnel** (BRC) : payé quoi qu'il arrive.
- **Conditionnel** (Phoenix) : payé seulement au-dessus de la barrière de coupon.

### Effet mémoire
Les coupons manqués sont rattrapés au premier paiement suivant.

### Trigger de rappel
Si le sous-jacent est au-dessus à une observation, le produit s'arrête.
**Le piège** : le rappel coupe tous les coupons futurs.

### Livraison physique
Si la barrière casse et que le sous-jacent finit sous le strike, le client reçoit des
actions au lieu de son cash.

### Worst-of
Panier où le plus mauvais des sous-jacents décide de tout. Le coupon affiché est
spectaculaire, et pour cause : il faut que TOUS tiennent, pas un seul.

**La corrélation devient le paramètre dominant** : corrélation haute, les actifs bougent
ensemble et le worst-of ressemble à un actif unique, donc coupon faible. Corrélation
basse, il y a presque toujours un traînard, donc coupon élevé et risque élevé.

### Risque émetteur
**Un produit structuré est une dette de la banque**, pas une détention d'actions.
En 2008, les porteurs de produits Lehman ont tout perdu alors que leurs barrières
tenaient. Premier point sur lequel un institutionnel te challengera.
"""

GRECQUES = """
| Grecque | Ce que c'est | Ce que ça change pour toi |
|---|---|---|
| **Delta** | Sensibilité au prix du sous-jacent | Le nombre d'actions que le trader détient pour être couvert. Il l'ajuste tous les jours : delta-hedging. |
| **Gamma** | Vitesse à laquelle le delta change | Explose près d'une barrière : volumes énormes à trader en peu de temps (pin risk). LA raison du prix dégradé sur les barrières proches du spot. |
| **Vega** | Sensibilité à la volatilité implicite | Ta grecque. Vol haute au pricing = coupon élevé. Ta fenêtre commerciale. |
| **Theta** | Érosion du prix avec le temps | Le vendeur d'option gagne du theta chaque jour. Un client qui achète un BRC est structurellement long theta. |
| **Rho** | Sensibilité aux taux | Déterminante sur le capital garanti, secondaire sur les produits à coupon. |

### Bump and revalue
On décale un paramètre de peu, on reprice, on regarde l'écart.

**Le détail qui compte** : réutiliser exactement les mêmes tirages aléatoires pour les
deux évaluations (common random numbers). Sinon le bruit Monte Carlo domine la
différence et la grecque est inexploitable.

### Coût de couverture
En théorie de Black-Scholes, une couverture continue et sans frais donne un résultat
exactement nul. Dans la vraie vie, deux fuites : le rebalancement est discret, et chaque
transaction coûte. Vois la page Coût de couverture : l'arbitrage fréquence / frais est un
vrai sujet de desk.
"""

VOLATILITE = """
### Volatilité implicite
**Pas une prévision.** Le prix d'une option exprimé dans une autre unité. Une option vaut
55 EUR, ou elle vaut 22% de vol : même information.

### Skew
Les puts bas coûtent plus cher **en vol** que les options à la monnaie. Typiquement sur
indice à 1 an : put 70% à 26%, ATM à 18%, call 110% à 16%.

**Deux causes** :
1. Les actions ne baissent pas comme elles montent. Un krach de -20% en une semaine
   existe ; une hausse de +20% en une semaine, non.
2. Les institutionnels achètent massivement des puts de protection, sans vendeur naturel.

**Ce que ça change** : le skew fait monter le coupon d'un BRC et améliore la participation
d'un capital garanti. Il profite au client dans les deux cas, pour des raisons opposées.

### Terme structure
La vol varie avec la maturité. En régime normal, la vol longue est au-dessus de la courte.
En stress, ça s'inverse : la vol 1 mois explose alors que la vol 2 ans bouge peu.

*Pour toi* : en période de stress, les BRC courts offrent des coupons exceptionnels.

### Surface de volatilité
Skew (par strike) + terme structure (par maturité) = une surface à deux dimensions.
L'écran du trader, la matière première du structureur.

### Vol implicite vs vol réalisée
L'implicite est structurellement au-dessus de la réalisée, de 2 à 4 points sur les
indices. Cet écart s'appelle la **variance risk premium**.

**C'est la raison d'être économique des BRC et des autocalls.** Le client capture cette
prime. Ni hasard ni anomalie : c'est la rémunération de celui qui accepte d'être exposé
aux krachs. Il gagne neuf années sur dix, et perd beaucoup la dixième.
"""

MODELES = """
### Black-Scholes
Une seule volatilité, constante, pour toutes les options d'un même sous-jacent.

**Ce qu'il fait bien** : les payoffs, l'intuition, les ordres de grandeur.
**Ce qu'il rate** : le skew (il faut le plaquer à la main), les krachs, la dynamique de
la volatilité.

### Heston
La volatilité devient elle-même un processus aléatoire, qui revient vers une moyenne.

    dv = kappa*(theta - v) dt + xi*sqrt(v) dW2

- **rho** : corrélation entre le sous-jacent et sa volatilité. Négatif = la vol monte
  quand le marché baisse. **C'est ce qui crée le skew**, au lieu de le postuler.
- **xi** : vol de la vol. Donne de la convexité au smile.
- **kappa, theta** : retour à la moyenne. Crée la structure par terme.

### Merton (sauts)
On ajoute des sauts poissonniens au sous-jacent. Un gap de -15% en une séance devient
possible. Le brownien géométrique en est incapable, et c'est exactement ce qui fait mal
sur un produit à barrière.

### Bates
Heston + Merton. **Le standard de marché pour les exotiques actions.** C'est le modèle
actif dans cette appli quand tu le sélectionnes.

### Ce que les desks utilisent vraiment
Bates ou une variante, calibré chaque jour sur les prix d'options cotées. Pour certains
produits, un modèle à volatilité locale (Dupire) qui reproduit exactement la surface
observée. Pour les worst-of, un modèle multi-actifs avec corrélation calibrée.

Dans cette appli, le mode Black-Scholes + skew utilise une volatilité locale, et Bates peut
être calibré sur une photo de prix Eurex (Laboratoire modèle), mais les curseurs restent libres.
C'est la différence entre cet outil et un pricer de production.
"""

JARGON = """
### Qui fait quoi

| Rôle | Ce qu'il fait |
|---|---|
| **Sales (toi)** | Interface client. Tu traduis une vue de marché en paramètres, tu fais pricer en interne, tu ramènes le prix. **Tu ne portes aucun risque.** |
| **Structureur** | Assemble le produit, calcule ce que la banque peut offrir, rédige le termsheet. |
| **Trader** | Porte le risque et le couvre. Quand un client achète un BRC, la banque se retrouve acheteuse d'un put qu'elle doit gérer pendant toute la durée. |

### Les deux ventes qu'il ne faut pas confondre
**La vente commerciale** : toi vers le client. Il te donne 1 000 EUR, tu lui livres le
produit. La seule qui apparaît dans ton métier au quotidien.

**La vente économique, cachée dans le produit** : le client vers ta banque. Il lui vend
une protection contre la baisse, et le coupon est le prix qu'elle lui paie pour ça.

### Vocabulaire courant

| Terme | Définition |
|---|---|
| Sous-jacent | L'actif dont dépend le produit : action, indice, panier. |
| Spot | Le prix actuel du sous-jacent. |
| Termsheet | Le document contractuel. Ce que le client signe. |
| Autocall | Tout produit avec rappel anticipé automatique. |
| Phoenix | Autocall à coupons conditionnels, généralement avec effet mémoire. La structure la plus vendue. |
| Reverse convertible | Le client vend de l'optionalité contre un coupon. Le "reverse" : c'est l'émetteur qui a le droit de livrer les actions. |
| Call spread | Achat d'un call + vente d'un call plus haut. Permet de capper un capital garanti pour remonter la participation. |
| Delta-hedging | L'activité quotidienne du trader : rester neutre à la direction. |
| Pin risk | Risque de couverture quand le sous-jacent stagne autour d'une barrière. |
| Wrong-way risk | Le défaut de l'émetteur survient au pire moment, corrélé au marché. Proposé en option sur la page BRC. |
| SSPA | Swiss Structured Products Association. Référence du marché suisse, premier marché mondial. |
| PRIIP / KID | Réglementation européenne : document d'information standardisé pour tout produit structuré vendu au retail. |
"""

TITRE_AIDE = "**Ce que tu trouves ici.** Tout le vocabulaire de l'appli, classé en six onglets : paramètres\nde marché, mécaniques produit, grecques, volatilité, modèles, jargon de desk.\n\n**Comment c'est écrit.** Chaque terme est défini, puis relié à ce qu'il change\nconcrètement : sur le prix, sur la couverture, ou sur la conversation avec un client.\n\n**Astuce.** Chaque curseur de l'appli a aussi sa propre infobulle. Le glossaire sert quand\ntu veux le contexte complet d'une notion.\n\n**Recherche.** Le champ en haut cherche dans les six onglets à la fois, sans tenir compte\ndes accents ni des majuscules, et liste chaque définition trouvée avec son onglet d'origine."


ONGLETS = [
    ("Paramètres de marché", PARAMETRES),
    ("Mécaniques produit", MECANIQUES),
    ("Les grecques", GRECQUES),
    ("Volatilité", VOLATILITE),
    ("Modèles", MODELES),
    ("Jargon de desk", JARGON),
]


# =====================================================================
# RECHERCHE
# =====================================================================

def normalise(texte):
    """Minuscules, sans accents ni mise en forme : un mot tape avec accent est trouve."""
    texte = unicodedata.normalize("NFKD", texte.replace("\u2019", "'").replace("\u0153", "oe"))
    texte = "".join(c for c in texte if not unicodedata.combining(c)).lower()
    for signe in "*|`#":
        texte = texte.replace(signe, " ")
    return " ".join(texte.split())


def _cellules(ligne):
    return [c.strip() for c in ligne.strip().strip("|").split("|")]


def decoupe(onglet, texte):
    """
    Decoupe un onglet en definitions : une par titre '###', et une par ligne
    de tableau (chaque ligne d'un tableau definit un terme).
    """
    defs, titre, lignes, entete = [], onglet, [], None

    def ferme():
        corps = "\n".join(lignes).strip("\n")
        if corps.strip():
            defs.append({"onglet": onglet, "titre": titre, "corps": corps})

    for ligne in texte.split("\n"):
        if ligne.startswith("### "):
            ferme()
            titre, lignes, entete = ligne[4:].strip(), [], None
        elif ligne.startswith("|"):
            cell = _cellules(ligne)
            if entete is None:
                entete = cell
            elif not set("".join(cell)) <= set("-: "):
                terme = cell[0].replace("**", "")
                if len(cell) == 2:
                    corps = cell[1]
                else:
                    corps = "\n\n".join(f"**{h}** : {v}" for h, v in zip(entete[1:], cell[1:]))
                defs.append({"onglet": onglet, "titre": terme, "corps": corps})
        else:
            if entete is not None and not ligne.strip():
                entete = None
            lignes.append(ligne)
    ferme()
    for d in defs:
        d["cle_titre"] = normalise(d["titre"])
        d["cle"] = d["cle_titre"] + " " + normalise(d["corps"])
    return defs


DEFINITIONS = [d for nom, texte in ONGLETS for d in decoupe(nom, texte)]


def cherche(requete):
    """Definitions qui contiennent tous les mots de la requete, titres d'abord."""
    mots = normalise(requete).split()
    if not mots:
        return []
    trouve = [d for d in DEFINITIONS if all(w in d["cle"] for w in mots)]
    return sorted(trouve, key=lambda d: not all(w in d["cle_titre"] for w in mots))


# =====================================================================
# PAGE
# =====================================================================

def affiche(m):
    st.title("Glossaire", help=TITRE_AIDE)
    requete = st.text_input(
        "Rechercher dans le glossaire", key="glossaire_recherche", bind="query-params",
        type="search",
        live=True, placeholder="barrière, skew, coupon, émetteur...",
        help="Cherche dans les six onglets à la fois, sans tenir compte des accents ni "
             "des majuscules. Plusieurs mots : seules les définitions qui les contiennent "
             "tous sont gardées. Les termes dont le titre correspond viennent en premier. "
             "Efface le champ pour retrouver les onglets.")

    if requete.strip():
        resultats = cherche(requete)
        if not resultats:
            st.info(f"Aucune définition ne contient « {requete.strip()} ». Essaie un mot "
                    "plus court ou un synonyme : vol, barrière, coupon, émetteur, delta.")
            return
        st.caption(f"{len(resultats)} définition(s) trouvée(s) pour « {requete.strip()} »")
        for d in resultats:
            with st.container(border=True):
                st.markdown(f"#### {d['titre']}")
                st.caption(f"Onglet : {d['onglet']}")
                st.markdown(d["corps"])
        return

    for onglet, (nom, texte) in zip(st.tabs([nom for nom, _ in ONGLETS]), ONGLETS):
        with onglet:
            st.markdown(texte)
