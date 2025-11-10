## SIG Spatial IA — DP IA2 Brest S9

### Préambule
Ce projet met en pratique des notions d’optimisation sur graphes autour du problème du voyageur de commerce (TSP). L’objectif est de modéliser un graphe de lieux, de calculer les plus courts chemins, puis de rechercher un cycle minimal visitant chaque lieu une seule fois avant de revenir au point de départ. Face à la complexité combinatoire \(O(n!)\), des approches heuristiques et métaheuristiques sont mobilisées, notamment un Algorithme Génétique (GA) et un Algorithme à base de Colonies de Fourmis (ACO).

## Objectifs pédagogiques
- **Plus court chemin**: mise en œuvre de Dijkstra / A* pour générer une matrice OD.
- **Matrice OD (Origine–Destination)**: calcul/chargement et utilisation dans les algorithmes.
- **Complexité algorithmique**: caractéristiques du TSP (NP-difficile), minima locaux.
- **Optimisation de tournées**: TSP et heuristiques.
- **IA inspirée par la nature**: Algorithmes génétiques, Colonies de fourmis, Recuit simulé (culture générale).

## Périmètre de cette partie
- Implémenter une base objet en Python pour le TSP:
  - `Lieu`, `Graph`, `Route`, `Affichage` dans `tsp_graph_init.py`.
- Implémenter deux solveurs TSP séparés:
  - `TSP_GA` (Algorithme Génétique).
  - `TSP_ACO` (Colonies de Fourmis).
- Mettre à jour l’affichage à chaque itération: lieux, informations de progression, meilleures routes/pheromones, meilleure distance et itération d’obtention.

## Contraintes techniques
- **Langage**: Python (POO, classes).
- **Bibliothèques autorisées**: `numpy`, `random`, `time`, `pandas`, `tkinter`, `csv`.
- **Interface**: `tkinter` avec `Canvas` aux dimensions constantes:
  - `LARGEUR = 800`, `HAUTEUR = 600`.
- **Données**: génération aléatoire contrôlée et/ou chargement CSV (format Moodle).

## Architecture attendue
### Fichier: `tsp_graph_init.py`
- **Classe `Lieu`**
  - Stocke `x`, `y`, `nom`.
  - Méthode de distance euclidienne entre deux `Lieu`.
- **Classe `Graph`**
  - Mémorise `liste_lieux` (taille: `NB_LIEUX`).
  - Génération aléatoire des lieux dans l’espace `[0, LARGEUR] × [0, HAUTEUR]` ou chargement via `charger_graph` (CSV).
  - `calcul_matrice_cout_od`: calcule ou importe la matrice de distances et la stocke dans `matrice_od`.
  - `plus_proche_voisin(lieu_idx)`: renvoie l’indice du lieu le plus proche selon `matrice_od`.
  - `calcul_distance_route(route)`: distance totale d’une route (euclidienne).
- **Classe `Route`**
  - Contient `ordre` (ex: `[0,3,8,1,2,4,6,5,9,7,0]`) commençant et finissant par le lieu `0`.
- **Classe `Affichage`**
  - Fenêtre avec titre incluant le nom du groupe.
  - Dessin des lieux (cercles + index au centre) dans un `Canvas` \(LARGEUR × HAUTEUR\).
  - Zone de texte sous le `Canvas` pour l’état des algorithmes (itération, meilleure distance, etc.).
  - Affiche en continu la meilleure route (ligne bleue pointillée) et l’ordre de visite au-dessus de chaque lieu.
  - Raccourcis clavier:
    - Touche dédiée: affiche les N meilleures routes en gris clair (GA) ou la matrice de coûts/pheromones (ACO).
    - `ESC`: quitte complètement l’application.

## Étape Python 2.A — Algorithme Génétique (`TSP_GA`)
### Attendus “analyse”
- Définir: complexité, NP-difficile, heuristique, minima locaux.
- Expliquer comment approximer “simplement” un bon résultat et les compromis.
- Présenter une **veille synthétique** au tableau (sans diapos).
### Attendus “réalisation”
- Classe `TSP_GA` utilisant `Lieu`, `Graph`, `Route`, `Affichage`.
- À chaque itération: afficher
  - les lieux,
  - les N meilleures routes,
  - la meilleure route,
  - le texte d’avancement (meilleure distance, itération d’obtention).
- Paramètres usuels à prévoir: taille de population, sélection, croisement, mutation, élitisme, critère d’arrêt.

## Étape Python 2.B — Colonies de Fourmis (`TSP_ACO`)
### Attendus “analyse”
- Même trame que GA: complexité, heuristique, NP-difficile, minima locaux.
- Veille et présentation au tableau.
### Attendus “réalisation”
- Classe `TSP_ACO` utilisant `Lieu`, `Graph`, `Route`, `Affichage`.
- À chaque itération: afficher
  - les lieux,
  - les phéromones sur les arêtes,
  - la meilleure route,
  - le texte d’avancement (meilleure distance, itération).
- Paramètres usuels: alpha/beta (poids phéromones vs heuristique), taux d’évaporation, dépôt, nombre de fourmis, stratégie de choix (roulette/argmax), critère d’arrêt.

## Données et CSV
- Les CSV fournis sur Moodle ont un format stable (utilisé pour l’évaluation).
- La méthode `charger_graph` doit lire ce format sans modification.
- Possibilité d’initialiser aléatoirement les lieux si aucun CSV n’est fourni.

## Organisation du dépôt (suggestion)
```
SIG_Spatial_IA/
├─ tsp_graph_init.py       # Lieu, Graph, Route, Affichage
├─ tsp_ga.py               # Classe TSP_GA
├─ tsp_aco.py              # Classe TSP_ACO
├─ data/
│  └─ exemples.csv         # Jeux de données (ex. Moodle)
├─ main.py                 # Lancement, paramètres, boucle d’itérations + Affichage
└─ README.md               # Ce document
```

## Démarrage rapide (suggestion)
1. Créer/placer un CSV dans `data/` ou activer la génération aléatoire.
2. Implémenter `tsp_graph_init.py`.
3. Implémenter `tsp_ga.py` et/ou `tsp_aco.py`.
4. Lancer l’application:
   - `python main.py`
5. Interagir via le clavier:
   - Touche dédiée: N meilleures routes (GA) / matrice de phéromones (ACO).
   - `ESC`: quitter.

## Critères d’évaluation (indicatifs)
- Respect des contraintes (POO, bibliothèques, interface, CSV).
- Correction des calculs (distances, matrice OD), robustesse I/O.
- Clarté et réactivité de l’affichage, interactions clavier.
- Qualité des implémentations GA/ACO, paramétrabilité, convergence.
- Qualité de code (lisibilité, nommage, découplage), commentaires utiles.
- Qualité de la veille et de la présentation pédagogique.

## Auteurs
Renseignez le nom du groupe ici (utilisé dans le titre de la fenêtre).

## Licence
Projet pédagogique — usage académique.
