"""Glossaire."""

import streamlit as st


def affiche(m):
    st.title("Glossaire", help="**Ce que tu trouves ici.** Tout le vocabulaire de l'appli, classe en six onglets : parametres\nde marche, mecaniques produit, grecques, volatilite, modeles, jargon de desk.\n\n**Comment c'est ecrit.** Chaque terme est defini, puis relie a ce qu'il change\nconcretement : sur le prix, sur la couverture, ou sur la conversation avec un client.\n\n**Astuce.** Chaque curseur de l'appli a aussi sa propre infobulle. Le glossaire sert quand\ntu veux le contexte complet d'une notion.")
    o = st.tabs(["Parametres de marche", "Mecaniques produit", "Les grecques",
                 "Volatilite", "Modeles", "Jargon de desk"])

    with o[0]:
        st.markdown("""
### Taux sans risque (r)
Carburant du budget option. Pour garantir 1 000 EUR dans 5 ans a 3%, la banque ne met
que 861 aujourd'hui. Les 139 restants achetent des options. A 0,5%, elle doit mettre 975
et il ne reste que 25.

**Consequence commerciale** : taux hauts = capital garanti vendable. Taux bas = le marche
bascule vers BRC et autocalls.

### Taux de dividende (q)
Le detenteur d'un produit structure ne le touche pas. Sur un tracker 3 ans avec 2,5% de
dividende, le client abandonne environ 7,8% de rendement sans le voir.

### Spread de credit emetteur
Ce que la banque paie au-dessus du taux sans risque pour se financer. **Il augmente le
coupon** : la banque se finance moins cher via le produit structure que sur le marche
obligataire, et reverse l'economie au client. Un coupon superieur a la concurrence peut
donc simplement signifier un emetteur moins bien note.

### Taux de recouvrement
Ce que le client recupere en cas de defaut. 40% est la convention senior non securise.
Les porteurs de produits Lehman ont recupere environ 9 cents par dollar.

### Marge banque
Prelevee sur le budget option, invisible dans le payoff. Divulguee reglementairement
(PRIIP, guidelines SSPA sur les couts).

### Monte Carlo
Pricing par simulation : des milliers de trajectoires, le payoff dans chacune, la moyenne
actualisee. Indispensable des qu'un produit depend du chemin.

### Univers risque-neutre
Le sous-jacent monte en moyenne au taux (r - q), pas selon une prevision. **Ce n'est pas
un pari directionnel** : c'est la seule hypothese qui interdit l'arbitrage.

### Pont brownien
Correction appliquee aux barrieres. Une simulation a pas discrets rate les franchissements
entre deux observations : elle sous-estime la probabilite de toucher. Le pont brownien la
calcule analytiquement entre deux points. Plus rapide **et** plus juste qu'une simulation
quotidienne.
""")

    with o[1]:
        st.markdown("""
### Strike
Niveau de reference qui decide du remboursement. Standard : 100% du spot initial.

### Barriere
- **Toucher une barriere ne fait pas perdre.** Ca arme le risque. C'est le strike qui
  decide a la fin.
- **Observation continue (americaine)** : la barriere compte a tout moment, meme une
  meche intraday. Standard marche.
- **Observation europeenne** : seul le niveau a l'echeance compte. Plus favorable au
  client, donc coupon plus faible.

### Knock-in / Knock-out
- **Knock-in** : l'option s'active au franchissement. Le put d'un BRC.
- **Knock-out** : l'option disparait au franchissement. Le put d'un Bonus ou d'un Twin-Win.

### Participation
Ratio de suivi. 70% : le sous-jacent fait +20%, le client touche +14%.
**Jamais choisie, toujours calculee** : budget option divise par prix du call.

### Coupon
**Ce n'est pas un interet.** C'est le prix d'une option que le client vend. Trois
sources : les taux, la prime du put vendu, et le spread emetteur.

Coupon eleve = sinistre probable, ou emetteur risque. Jamais bonne affaire.

### Coupon conditionnel vs inconditionnel
- **Inconditionnel** (BRC) : paye quoi qu'il arrive.
- **Conditionnel** (Phoenix) : paye seulement au-dessus de la barriere de coupon.

### Effet memoire
Les coupons manques sont rattrapes au premier paiement suivant.

### Trigger de rappel
Si le sous-jacent est au-dessus a une observation, le produit s'arrete.
**Le piege** : le rappel coupe tous les coupons futurs.

### Livraison physique
Si la barriere casse et que le sous-jacent finit sous le strike, le client recoit des
actions au lieu de son cash.

### Worst-of
Panier ou le plus mauvais des sous-jacents decide de tout. Le coupon affiche est
spectaculaire, et pour cause : il faut que TOUS tiennent, pas un seul.

**La correlation devient le parametre dominant** : correlation haute, les actifs bougent
ensemble et le worst-of ressemble a un actif unique, donc coupon faible. Correlation
basse, il y a presque toujours un trainard, donc coupon eleve et risque eleve.

### Risque emetteur
**Un produit structure est une dette de la banque**, pas une detention d'actions.
En 2008, les porteurs de produits Lehman ont tout perdu alors que leurs barrieres
tenaient. Premier point sur lequel un institutionnel te challengera.
""")

    with o[2]:
        st.markdown("""
| Grecque | Ce que c'est | Ce que ca change pour toi |
|---|---|---|
| **Delta** | Sensibilite au prix du sous-jacent | Le nombre d'actions que le trader detient pour etre couvert. Il l'ajuste tous les jours : delta-hedging. |
| **Gamma** | Vitesse a laquelle le delta change | Explose pres d'une barriere : volumes enormes a trader en peu de temps (pin risk). LA raison du prix degrade sur les barrieres proches du spot. |
| **Vega** | Sensibilite a la volatilite implicite | Ta grecque. Vol haute au pricing = coupon eleve. Ta fenetre commerciale. |
| **Theta** | Erosion du prix avec le temps | Le vendeur d'option gagne du theta chaque jour. Un client qui achete un BRC est structurellement long theta. |
| **Rho** | Sensibilite aux taux | Determinante sur le capital garanti, secondaire sur les produits a coupon. |

### Bump and revalue
On decale un parametre de peu, on reprice, on regarde l'ecart.

**Le detail qui compte** : reutiliser exactement les memes tirages aleatoires pour les
deux evaluations (common random numbers). Sinon le bruit Monte Carlo domine la
difference et la grecque est inexploitable.

### Cout de couverture
En theorie de Black-Scholes, une couverture continue et sans frais donne un resultat
exactement nul. Dans la vraie vie, deux fuites : le rebalancement est discret, et chaque
transaction coute. Vois la page Cout de couverture : l'arbitrage frequence / frais est un
vrai sujet de desk.
""")

    with o[3]:
        st.markdown("""
### Volatilite implicite
**Pas une prevision.** Le prix d'une option exprime dans une autre unite. Une option vaut
55 EUR, ou elle vaut 22% de vol : meme information.

### Skew
Les puts bas coutent plus cher **en vol** que les options a la monnaie. Typiquement sur
indice a 1 an : put 70% a 26%, ATM a 18%, call 110% a 16%.

**Deux causes** :
1. Les actions ne baissent pas comme elles montent. Un krach de -20% en une semaine
   existe ; une hausse de +20% en une semaine, non.
2. Les institutionnels achetent massivement des puts de protection, sans vendeur naturel.

**Ce que ca change** : le skew fait monter le coupon d'un BRC et ameliore la participation
d'un capital garanti. Il profite au client dans les deux cas, pour des raisons opposees.

### Terme structure
La vol varie avec la maturite. En regime normal, la vol longue est au-dessus de la courte.
En stress, ca s'inverse : la vol 1 mois explose alors que la vol 2 ans bouge peu.

*Pour toi* : en periode de stress, les BRC courts offrent des coupons exceptionnels.

### Surface de volatilite
Skew (par strike) + terme structure (par maturite) = une surface a deux dimensions.
L'ecran du trader, la matiere premiere du structureur.

### Vol implicite vs vol realisee
L'implicite est structurellement au-dessus de la realisee, de 2 a 4 points sur les
indices. Cet ecart s'appelle la **variance risk premium**.

**C'est la raison d'etre economique des BRC et des autocalls.** Le client capture cette
prime. Ni hasard ni anomalie : c'est la remuneration de celui qui accepte d'etre expose
aux krachs. Il gagne neuf annees sur dix, et perd beaucoup la dixieme.
""")

    with o[4]:
        st.markdown("""
### Black-Scholes
Une seule volatilite, constante, pour toutes les options d'un meme sous-jacent.

**Ce qu'il fait bien** : les payoffs, l'intuition, les ordres de grandeur.
**Ce qu'il rate** : le skew (il faut le plaquer a la main), les krachs, la dynamique de
la volatilite.

### Heston
La volatilite devient elle-meme un processus aleatoire, qui revient vers une moyenne.

    dv = kappa*(theta - v) dt + xi*sqrt(v) dW2

- **rho** : correlation entre le sous-jacent et sa volatilite. Negatif = la vol monte
  quand le marche baisse. **C'est ce qui cree le skew**, au lieu de le postuler.
- **xi** : vol de la vol. Donne de la convexite au smile.
- **kappa, theta** : retour a la moyenne. Cree la structure par terme.

### Merton (sauts)
On ajoute des sauts poissonniens au sous-jacent. Un gap de -15% en une seance devient
possible. Le brownien geometrique en est incapable, et c'est exactement ce qui fait mal
sur un produit a barriere.

### Bates
Heston + Merton. **Le standard de marche pour les exotiques actions.** C'est le modele
actif dans cette appli quand tu le selectionnes.

### Ce que les desks utilisent vraiment
Bates ou une variante, calibre chaque jour sur les prix d'options cotees. Pour certains
produits, un modele a volatilite locale (Dupire) qui reproduit exactement la surface
observee. Pour les worst-of, un modele multi-actifs avec correlation calibree.

Les parametres de cette appli sont regles a la main, pas calibres sur un marche reel.
C'est la difference entre cet outil et un pricer de production.
""")

    with o[5]:
        st.markdown("""
### Qui fait quoi

| Role | Ce qu'il fait |
|---|---|
| **Sales (toi)** | Interface client. Tu traduis une vue de marche en parametres, tu fais pricer en interne, tu ramenes le prix. **Tu ne portes aucun risque.** |
| **Structureur** | Assemble le produit, calcule ce que la banque peut offrir, redige le termsheet. |
| **Trader** | Porte le risque et le couvre. Quand un client achete un BRC, la banque se retrouve acheteuse d'un put qu'elle doit gerer pendant toute la duree. |

### Les deux ventes qu'il ne faut pas confondre
**La vente commerciale** : toi vers le client. Il te donne 1 000 EUR, tu lui livres le
produit. La seule qui apparait dans ton metier au quotidien.

**La vente economique, cachee dans le produit** : le client vers ta banque. Il lui vend
une protection contre la baisse, et le coupon est le prix qu'elle lui paie pour ca.

### Vocabulaire courant

| Terme | Definition |
|---|---|
| Sous-jacent | L'actif dont depend le produit : action, indice, panier. |
| Spot | Le prix actuel du sous-jacent. |
| Termsheet | Le document contractuel. Ce que le client signe. |
| Autocall | Tout produit avec rappel anticipe automatique. |
| Phoenix | Autocall a coupons conditionnels, generalement avec effet memoire. La structure la plus vendue. |
| Reverse convertible | Le client vend de l'optionalite contre un coupon. Le "reverse" : c'est l'emetteur qui a le droit de livrer les actions. |
| Call spread | Achat d'un call + vente d'un call plus haut. Permet de capper un capital garanti pour remonter la participation. |
| Delta-hedging | L'activite quotidienne du trader : rester neutre a la direction. |
| Pin risk | Risque de couverture quand le sous-jacent stagne autour d'une barriere. |
| Wrong-way risk | Le defaut de l'emetteur survient au pire moment, correle au marche. Ignore dans cette appli. |
| SSPA | Swiss Structured Products Association. Reference du marche suisse, premier marche mondial. |
| PRIIP / KID | Reglementation europeenne : document d'information standardise pour tout produit structure vendu au retail. |
""")
