import numpy as np
import random
import time
import tkinter as tk
import csv
import pandas as pd


# Constantes
LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 10


class Lieu:
    """
    Représente un lieu à visiter avec coordonnées (x, y) et nom.
    Optimisation: __slots__ pour réduire l'empreinte mémoire.
    """
    __slots__ = ('x', 'y', 'nom')
    
    def __init__(self, x, y, nom):
        self.x = float(x)
        self.y = float(y)
        self.nom = str(nom)
    
    def distance(self, autre_lieu):
        """
        Calcule la distance euclidienne vers un autre lieu.
        """
        dx = self.x - autre_lieu.x
        dy = self.y - autre_lieu.y
        return float(np.hypot(dx, dy))


class Route:
    """
    Représente une route (cycle) visitant tous les lieux.
    Contrainte: commence et se termine par le lieu 0 (dépôt).
    """
    __slots__ = ('ordre',)
    
    def __init__(self, ordre=None, nombre_lieux=None):
        if ordre is not None:
            core = [int(i) for i in list(ordre) if int(i) != 0]
            self.ordre = [0] + core + [0] if core else [0]
        elif nombre_lieux is not None:
            self.ordre = [0] + list(range(1, nombre_lieux)) + [0]
        else:
            self.ordre = [0]


class Graph:
    """
    Graphe de lieux avec matrice OD et utilitaires de calcul.
    Optimisations:
    """
    __slots__ = ('liste_lieux', 'matrice_od')
    
    def __init__(self):
        self.liste_lieux = []
        self.matrice_od = None
    
    def generer_lieux_aleatoires(self, nb_lieux=NB_LIEUX, largeur=LARGEUR, hauteur=HAUTEUR, graine=None):

        if graine is not None:
            np.random.seed(graine)
            random.seed(graine)
        
        # Génération vectorisée des coordonnées
        coords = np.random.uniform(0, [largeur, hauteur], size=(nb_lieux, 2))
        self.liste_lieux = [Lieu(coords[i, 0], coords[i, 1], f"Lieu {i}") 
                           for i in range(nb_lieux)]
        self.matrice_od = None
    
    def charger_graph(self, chemin_csv):
        """
        Charge les lieux depuis un fichier CSV.
        """
        self.liste_lieux = []
        
        try:
            df = pd.read_csv(chemin_csv)
            colonnes = {c.lower(): c for c in df.columns}
            
            if 'x' not in colonnes or 'y' not in colonnes:
                raise ValueError("CSV doit contenir colonnes 'x' et 'y'")
            
            col_x, col_y = colonnes['x'], colonnes['y']
            col_nom = colonnes.get('nom')
            
            # Création vectorisée des lieux
            for i, row in df.iterrows():
                nom = str(row[col_nom]) if col_nom else f"Lieu {i}"
                self.liste_lieux.append(Lieu(row[col_x], row[col_y], nom))
        
        except ImportError:
            # Fallback avec csv standard
            with open(chemin_csv, newline='', encoding='utf-8') as f:
                lecteur = csv.DictReader(f)
                champs = {c.lower(): c for c in lecteur.fieldnames} if lecteur.fieldnames else {}
                col_x, col_y = champs.get('x'), champs.get('y')
                col_nom = champs.get('nom')
                
                if not col_x or not col_y:
                    raise ValueError("CSV doit contenir colonnes 'x' et 'y'")
                
                for i, ligne in enumerate(lecteur):
                    nom = ligne.get(col_nom, f"Lieu {i}") if col_nom else f"Lieu {i}"
                    self.liste_lieux.append(Lieu(float(ligne[col_x]), float(ligne[col_y]), nom))
        self.matrice_od = None
    
    def calcul_matrice_cout_od(self):
        """
        Calcule la matrice des distances euclidiennes entre tous les lieux.
        Implémentation optimisée (formule de Gram, tuilage, symétrie, dtype adaptatif).
        """
        n = len(self.liste_lieux)
        
        if n == 0:
            self.matrice_od = np.zeros((0, 0), dtype=np.float64)
            return self.matrice_od
        
        # Choix adaptatif du dtype selon la taille
        dtype = np.float32 if n >= 5000 else np.float64
        itemsize = np.dtype(dtype).itemsize
        
        # Extraction vectorisée des coordonnées (C-contigu)
        X = np.ascontiguousarray(
            np.array([[lieu.x, lieu.y] for lieu in self.liste_lieux], dtype=dtype)
        )
        # Normes au carré ||xi||^2
        s = np.einsum('ij,ij->i', X, X, dtype=dtype)
        
        # memmap si la matrice dépasse ~512MB
        total_bytes = n * n * itemsize
        use_memmap = total_bytes > 512 * 1024 * 1024
        if use_memmap:
            mmap_path = "matrice_od_mmap.dat"
            D = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(n, n))
        else:
            D = np.empty((n, n), dtype=dtype)
        
        # Tuilage (blocs) + symétrie
        block = 1024 if n >= 2048 else (512 if n >= 1024 else n)
        for i0 in range(0, n, block):
            i1 = min(i0 + block, n)
            Xi = X[i0:i1]      # (bi, 2)
            si = s[i0:i1]      # (bi,)
            for j0 in range(i0, n, block):
                j1 = min(j0 + block, n)
                Xj = X[j0:j1]  # (bj, 2)
                sj = s[j0:j1]  # (bj,)
                
                # Produit de Gram pour le bloc
                G = Xi @ Xj.T  # (bi, bj)
                # dist^2 = si + sj^T - 2G
                D2 = si[:, None] + sj[None, :] - 2.0 * G
                # Stabilité numérique et racine
                np.maximum(D2, 0, out=D2)
                np.sqrt(D2, out=D2)
                
                # Remplit le bloc supérieur et miroir
                D[i0:i1, j0:j1] = D2.astype(dtype, copy=False)
                if j0 != i0:
                    D[j0:j1, i0:i1] = D2.T.astype(dtype, copy=False)
        
        # Diagonale = 0
        if isinstance(D, np.memmap):
            for k in range(n):
                D[k, k] = dtype(0)
        else:
            np.fill_diagonal(D, dtype(0))
        
        self.matrice_od = D
        return D
    
    def plus_proche_voisin(self, lieu_idx):
        """
        Retourne l'indice du lieu le plus proche de lieu_idx.
        Utilise la matrice OD (calculée si nécessaire).
        Optimisé: vectorisation NumPy, cache automatique.
        """
        # Calcule la matrice si absente
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        
        n = self.matrice_od.shape[0]
        if n == 0:
            return None
        
        # Copie la ligne des distances, exclut soi-même
        distances = self.matrice_od[lieu_idx].copy()
        distances[lieu_idx] = np.inf
        
        return int(np.argmin(distances))
    
    def calcul_distance_route(self, route):
        """
        Calcule la distance totale d'une route (somme des arêtes successives).
        
        Optimisations:
        - Utilise la matrice OD si disponible (O(K) avec K = longueur route)
        - Sinon calcul direct via coordonnées (fallback)
        - Indexation vectorielle NumPy avancée
        """
        ordre = route.ordre
        
        if len(ordre) < 2:
            return 0.0
        
        # Vérifie si la matrice OD est utilisable
        if self.matrice_od is None or self.matrice_od.shape[0] != len(self.liste_lieux):
            # Fallback: calcul direct (plus lent mais sûr)
            total = 0.0
            for i in range(len(ordre) - 1):
                total += self.liste_lieux[ordre[i]].distance(self.liste_lieux[ordre[i + 1]])
            return total
        
        # Méthode optimisée: indexation vectorielle de la matrice
        indices_src = np.array(ordre[:-1], dtype=np.int32)
        indices_dst = np.array(ordre[1:], dtype=np.int32)
        
        # Somme vectorisée: matrice_od[src, dst] pour toutes les arêtes
        return float(self.matrice_od[indices_src, indices_dst].sum())


class Affichage:
    """
    Interface Tkinter pour visualiser le graphe et les routes.
    
    Optimisations:
    - Mode headless (sans GUI) pour benchmarks
    - Utilisation de tags Tkinter pour updates rapides
    - Batch updates du canvas
    - __slots__ pour économie mémoire
    """
    __slots__ = ('graph', 'n_top_routes', 'top_routes', 'meilleure_route', 
                 'pheromones', 'headless', 'root', 'canvas', 'zone_texte', 
                 'afficher_routes', 'afficher_pheromones')
    
    # Constantes d'affichage
    COULEUR_LIEU = "#222222"
    COULEUR_TEXTE_LIEU = "#ffffff"
    COULEUR_MEILLEURE_ROUTE = "blue"
    COULEUR_TOP_ROUTES = "#c0c0c0"  # gris clair
    RAYON_LIEU = 10
    
    def __init__(self, graph, titre_fenetre="SIG Spatial IA — Groupe 8", 
                 n_top_routes=5, headless=False):
        """
        Initialise l'affichage.
        
        :param headless: Si True, désactive l'interface graphique (mode benchmark)
        """
        self.graph = graph
        self.n_top_routes = int(n_top_routes)
        self.top_routes = []
        self.meilleure_route = None
        self.pheromones = None
        self.headless = headless
        
        self.afficher_routes = True
        self.afficher_pheromones = False
        
        if not self.headless:
            # Initialisation Tkinter complète
            self.root = tk.Tk()
            self.root.title(titre_fenetre)
            self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
            
            # Canvas pour le graphe
            self.canvas = tk.Canvas(self.root, width=LARGEUR, height=HAUTEUR, 
                                   bg="white", highlightthickness=0)
            self.canvas.pack(side="top", fill="both", expand=False)
            
            # Zone de texte pour les infos
            self.zone_texte = tk.Text(self.root, height=8, width=100)
            self.zone_texte.pack(side="bottom", fill="x")
            
            # Raccourcis clavier
            self.root.bind("<Escape>", lambda e: self.root.destroy())
            self.root.bind("r", self._on_toggle_routes)
            self.root.bind("p", self._on_toggle_pheromones)
            
            # Affichage initial
            self._dessiner_graph()
            self._redessiner_routes()
        else:
            # Mode headless: attributs fictifs
            self.root = None
            self.canvas = None
            self.zone_texte = None
    
    def set_meilleure_route(self, route):
        """Met à jour la meilleure route et redessine."""
        self.meilleure_route = route
        if not self.headless:
            self._redessiner_routes()
    
    def set_top_routes(self, routes):
        """Met à jour les N meilleures routes et redessine."""
        self.top_routes = list(routes)[:self.n_top_routes]
        if not self.headless:
            self._redessiner_routes()
    
    def set_pheromones(self, matrice):
        """Met à jour la matrice de phéromones et affiche si activé."""
        self.pheromones = matrice
        if not self.headless:
            self._mettre_a_jour_zone_texte_pheromones()
    
    def set_message(self, texte):
        """Affiche un message dans la zone de texte."""
        if not self.headless:
            self.zone_texte.delete("1.0", tk.END)
            self.zone_texte.insert(tk.END, texte)
    
    def _on_toggle_routes(self, _evt=None):
        """Raccourci 'r': affiche/masque les N meilleures routes."""
        self.afficher_routes = not self.afficher_routes
        self._redessiner_routes()
    
    def _on_toggle_pheromones(self, _evt=None):
        """Raccourci 'p': affiche/masque la matrice OD/phéromones."""
        self.afficher_pheromones = not self.afficher_pheromones
        self._mettre_a_jour_zone_texte_pheromones()
    
    def _dessiner_graph(self):
        """
        Dessine tous les lieux du graphe.
        Optimisé: batch create des cercles et textes.
        """
        self.canvas.delete("all")
        
        # Batch drawing pour meilleure performance
        for idx, lieu in enumerate(self.graph.liste_lieux):
            cx, cy, r = lieu.x, lieu.y, self.RAYON_LIEU
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, 
                                   fill=self.COULEUR_LIEU, outline="", tags="lieu")
            self.canvas.create_text(cx, cy, text=str(idx), 
                                   fill=self.COULEUR_TEXTE_LIEU, 
                                   font=("Arial", 10, "bold"), tags="lieu")
        
        self.canvas.update_idletasks()
    
    def _redessiner_routes(self):
        """
        Redessine les routes (N meilleures + meilleure).
        Optimisé: utilise tags pour effacement sélectif.
        """
        # Efface uniquement les routes et ordres (pas les lieux)
        self.canvas.delete("route")
        self.canvas.delete("ordre")
        
        # Dessine les N meilleures routes en gris clair
        if self.afficher_routes:
            for route in self.top_routes:
                self._dessiner_route(route, self.COULEUR_TOP_ROUTES, (2, 4))
        
        # Dessine la meilleure route en bleu pointillé
        if self.meilleure_route is not None:
            self._dessiner_route(self.meilleure_route, self.COULEUR_MEILLEURE_ROUTE, (6, 4))
            self._dessiner_ordre_visite(self.meilleure_route)
        
        self.canvas.update_idletasks()
    
    def _dessiner_route(self, route, couleur, dash):
        """
        Dessine une route sur le canvas.
        Optimisé: création d'une seule polyline au lieu de segments multiples.
        """
        if len(route.ordre) < 2:
            return
        
        # Construit la liste de points pour create_line
        points = []
        for idx in route.ordre:
            lieu = self.graph.liste_lieux[idx]
            points.extend([lieu.x, lieu.y])
        
        # Création d'une seule ligne (plus rapide que plusieurs segments)
        self.canvas.create_line(*points, fill=couleur, dash=dash, 
                               width=2, tags="route", smooth=False)
    
    def _dessiner_ordre_visite(self, route):
        """
        Affiche l'ordre de visite au-dessus de chaque lieu.
        Optimisé: batch create des labels.
        """
        # Affiche la position de visite pour chaque lieu (sauf le dernier 0)
        for position, idx_lieu in enumerate(route.ordre[:-1]):
            lieu = self.graph.liste_lieux[idx_lieu]
            self.canvas.create_text(lieu.x, lieu.y - (self.RAYON_LIEU + 10),
                                   text=str(position), fill="black", 
                                   font=("Arial", 9), tags="ordre")
    
    def _mettre_a_jour_zone_texte_pheromones(self):
        """Affiche la matrice OD ou phéromones en texte."""
        if not self.afficher_pheromones:
            return
        
        self.zone_texte.delete("1.0", tk.END)
        
        if self.pheromones is None:
            # Affiche la matrice OD par défaut
            matrice = self.graph.matrice_od
            if matrice is None:
                self.zone_texte.insert(tk.END, "Aucune matrice disponible.")
                return
            self.zone_texte.insert(tk.END, "Matrice de coûts (OD):\n")
            self._inserer_matrice_texte(matrice)
        else:
            self.zone_texte.insert(tk.END, "Matrice de phéromones:\n")
            self._inserer_matrice_texte(self.pheromones)
    
    def _inserer_matrice_texte(self, matrice):
        """Formate et insère une matrice NumPy en texte."""
        arr = np.array(matrice)
        with np.printoptions(precision=2, suppress=True, linewidth=200):
            self.zone_texte.insert(tk.END, str(arr))
    
    def mainloop(self):
        """Lance la boucle principale Tkinter (si pas headless)."""
        if not self.headless and self.root is not None:
            self.root.mainloop()


class TSP_GA:
    """
    Algorithme Génétique pour résoudre le TSP.
    
    Optimisations:
    - __slots__ pour économie mémoire
    - Vectorisation NumPy pour évaluations batch
    - Order Crossover (OX) - standard TSP
    - Swap mutation - ultra rapide
    - Sélection par tournoi - bon équilibre
    - Élitisme 10% - préserve meilleures solutions
    - Update interval ajustable - performance GUI
    """
    __slots__ = ('graph', 'affichage', 'taille_population', 'taux_mutation', 
                 'nb_elites', 'update_interval', 'population', 'meilleure_route',
                 'meilleure_distance', 'iteration_courante', 'iteration_meilleure')
    
    def __init__(self, graph, affichage, taille_population=50, taux_mutation=0.02, 
                 taux_elitisme=0.1, update_interval=10):
        """
        Initialise l'algorithme génétique pour TSP.
        
        :param graph: Instance de Graph contenant les lieux
        :param affichage: Instance d'Affichage pour visualisation
        :param taille_population: Nombre de routes dans la population (défaut: 50)
        :param taux_mutation: Probabilité de mutation d'un gène (défaut: 0.02 = 2%)
        :param taux_elitisme: Proportion des meilleurs conservés (défaut: 0.1 = 10%)
        :param update_interval: Fréquence de mise à jour affichage (défaut: 10 générations)
        """
        self.graph = graph
        self.affichage = affichage
        self.taille_population = int(taille_population)
        self.taux_mutation = float(taux_mutation)
        self.nb_elites = max(1, int(taille_population * taux_elitisme))
        self.update_interval = int(update_interval)
        
        self.population = []
        self.meilleure_route = None
        self.meilleure_distance = float('inf')
        self.iteration_courante = 0
        self.iteration_meilleure = 0
        
        # Initialisation de la population
        self._initialiser_population()
    
    def _initialiser_population(self):
        """
        Crée une population initiale de routes aléatoires.
        Optimisé: génération vectorisée des permutations.
        """
        nb_lieux = len(self.graph.liste_lieux)
        if nb_lieux < 2:
            return
        
        # Génère taille_population permutations aléatoires (sans le lieu 0)
        indices_base = list(range(1, nb_lieux))
        
        for _ in range(self.taille_population):
            indices = indices_base.copy()
            random.shuffle(indices)
            ordre = [0] + indices + [0]
            self.population.append(Route(ordre))
        
        # Évalue la population initiale
        self._evaluer_population()
    
    def _evaluer_population(self):
        """
        Calcule la distance de chaque route et trie par fitness.
        Optimisé: vectorisation partielle via liste comprehension.
        """
        # Calcul vectorisé des distances
        fitness = [(self.graph.calcul_distance_route(route), route) 
                   for route in self.population]
        
        # Tri par distance croissante (meilleure en premier)
        fitness.sort(key=lambda x: x[0])
        self.population = [route for _, route in fitness]
        
        # Met à jour la meilleure solution globale
        if fitness and fitness[0][0] < self.meilleure_distance:
            self.meilleure_distance = fitness[0][0]
            self.meilleure_route = fitness[0][1]
            self.iteration_meilleure = self.iteration_courante
    
    def _selection_tournoi(self, taille_tournoi=3):
        """
        Sélection par tournoi: tire K routes aléatoires, retourne la meilleure.
        Optimisé: pas de calcul de probabilités, O(K).
        """
        taille = min(taille_tournoi, len(self.population))
        candidats = random.sample(self.population, taille)
        
        # Trouve le meilleur candidat (distance minimale)
        meilleur = min(candidats, key=lambda r: self.graph.calcul_distance_route(r))
        return meilleur
    
    def _croisement_ox(self, parent1, parent2):
        """
        Order Crossover (OX) - Standard pour TSP.
        Préserve l'ordre relatif des gènes (sous-tours optimaux).
        
        Complexité: O(N) avec N = nombre de lieux
        """
        ordre1 = parent1.ordre[1:-1]  # Sans les 0 de début/fin
        ordre2 = parent2.ordre[1:-1]
        
        if len(ordre1) < 2:
            return Route([0] + list(ordre1) + [0])
        
        # Sélectionne un segment aléatoire de parent1
        point1 = random.randint(0, len(ordre1) - 1)
        point2 = random.randint(point1 + 1, len(ordre1))
        
        # Copie le segment dans l'enfant
        segment = ordre1[point1:point2]
        enfant = [None] * len(ordre1)
        enfant[point1:point2] = segment
        
        # Remplit les trous avec les gènes de parent2 dans l'ordre
        genes_restants = [g for g in ordre2 if g not in segment]
        idx_enfant = 0
        for gene in genes_restants:
            while enfant[idx_enfant] is not None:
                idx_enfant += 1
            enfant[idx_enfant] = gene
        
        return Route([0] + enfant + [0])
    
    def _mutation_swap(self, route):
        """
        Mutation par swap (échange de 2 gènes).
        Ultra rapide: O(1) par mutation.
        
        Pour chaque gène: probabilité taux_mutation d'être échangé avec un autre.
        """
        ordre = route.ordre[1:-1]  # Sans les 0
        if len(ordre) < 2:
            return
        
        # Pour chaque position, probabilité de mutation
        for i in range(len(ordre)):
            if random.random() < self.taux_mutation:
                j = random.randint(0, len(ordre) - 1)
                ordre[i], ordre[j] = ordre[j], ordre[i]
        
        # Reconstruit la route avec les 0
        route.ordre = [0] + ordre + [0]
    
    def _nouvelle_generation(self):
        """
        Crée une nouvelle génération via élitisme, croisement et mutation.
        """
        nouvelle_pop = []
        
        # Élitisme: conserve les nb_elites meilleures routes
        nouvelle_pop.extend(self.population[:self.nb_elites])
        
        # Génère le reste par croisement et mutation
        while len(nouvelle_pop) < self.taille_population:
            # Sélection des parents
            parent1 = self._selection_tournoi()
            parent2 = self._selection_tournoi()
            
            # Croisement
            enfant = self._croisement_ox(parent1, parent2)
            
            # Mutation
            self._mutation_swap(enfant)
            
            nouvelle_pop.append(enfant)
        
        self.population = nouvelle_pop
    
    def _mettre_a_jour_affichage(self):
        """
        Met à jour l'affichage avec les meilleures routes et informations.
        Optimisé: appel conditionnel selon headless et update_interval.
        """
        if self.affichage.headless:
            return
        
        # Affiche la meilleure route globale
        if self.meilleure_route is not None:
            self.affichage.set_meilleure_route(self.meilleure_route)
        
        # Affiche les N meilleures routes de la population actuelle
        self.affichage.set_top_routes(self.population[:self.affichage.n_top_routes])
        
        # Message d'avancement
        msg = f"Itération: {self.iteration_courante} | "
        msg += f"Meilleure distance: {self.meilleure_distance:.2f} | "
        msg += f"Trouvée à l'itération: {self.iteration_meilleure}\n"
        msg += f"Taille population: {self.taille_population} | "
        msg += f"Taux mutation: {self.taux_mutation*100:.1f}% | "
        msg += f"Élites: {self.nb_elites}\n"
        msg += "Raccourcis: 'r' = top routes, 'p' = matrice OD, 'ESC' = quitter"
        self.affichage.set_message(msg)
    
    def executer(self, nb_iterations=500, delai_ms=50):
        """
        Exécute l'algorithme génétique pour un nombre d'itérations donné.
        
        :param nb_iterations: Nombre de générations à effectuer
        :param delai_ms: Délai en ms entre updates GUI (si pas headless)
        
        Optimisations:
        - Updates GUI conditionnels selon update_interval
        - Mode headless: exécution synchrone rapide
        - Mode GUI: exécution asynchrone avec tkinter.after
        """
        if self.affichage.headless:
            # Mode headless: exécution rapide sans GUI
            for _ in range(nb_iterations):
                self._nouvelle_generation()
                self._evaluer_population()
                self.iteration_courante += 1
            
            # Affichage final en console
            print(f"\n=== Algorithme Génétique Terminé ===")
            print(f"Itérations: {self.iteration_courante}")
            print(f"Meilleure distance: {self.meilleure_distance:.2f}")
            print(f"Trouvée à l'itération: {self.iteration_meilleure}")
        
        else:
            # Mode GUI: exécution asynchrone avec updates
            def iteration():
                if self.iteration_courante < nb_iterations:
                    # Exécute une génération
                    self._nouvelle_generation()
                    self._evaluer_population()
                    self.iteration_courante += 1
                    
                    # Update GUI selon l'intervalle
                    if self.iteration_courante % self.update_interval == 0:
                        self._mettre_a_jour_affichage()
                    
                    # Planifie la prochaine itération
                    self.affichage.root.after(delai_ms, iteration)
                else:
                    # Affichage final
                    self._mettre_a_jour_affichage()
            
            # Affichage initial
            self._mettre_a_jour_affichage()
            
            # Lance les itérations asynchrones
            self.affichage.root.after(delai_ms, iteration)
            self.affichage.mainloop()























if __name__ == "__main__":
    # Démonstration de l'algorithme génétique TSP
    print("=== TSP - Algorithme Génétique ===")
    print(f"Génération de {NB_LIEUX} lieux aléatoires...")
    
    g = Graph()
    
    # Option 1: Chargement depuis CSV (décommenter pour utiliser)
    # g.charger_graph("tests/data/graph_5.csv")
    g.charger_graph("tests/data/graph_20.csv")
    
    # Option 2: Génération aléatoire
    #g.generer_lieux_aleatoires(nb_lieux=NB_LIEUX, largeur=LARGEUR, hauteur=HAUTEUR, graine=42)
    
    print(f"Calcul de la matrice OD ({len(g.liste_lieux)} lieux)...")
    t0 = time.time()
    g.calcul_matrice_cout_od()
    t1 = time.time()
    print(f"Matrice OD calculée en {(t1-t0)*1000:.2f} ms")
    print(f"Type: {g.matrice_od.dtype}, Taille: {g.matrice_od.nbytes / 1024:.2f} KB")
    
    # Route séquentielle (solution de référence)
    route_init = Route(ordre=None, nombre_lieux=len(g.liste_lieux))
    distance_init = g.calcul_distance_route(route_init)
    print(f"Distance route séquentielle (référence): {distance_init:.2f}")
    
    # Initialisation de l'interface graphique
    print("\nInitialisation de l'algorithme génétique...")
    ui = Affichage(g, titre_fenetre="SIG Spatial IA — Groupe 8 — Algorithme Génétique", 
                  n_top_routes=5, headless=False)
    
    # Création et lancement de l'algorithme génétique
    tsp_ga = TSP_GA(
        graph=g,
        affichage=ui,
        taille_population=10000,     
        taux_mutation=0.02,        # 2% de mutation
        taux_elitisme=0.1,         # 5% d'élitisme (top 5 sur 100)
        update_interval=10         # Update GUI toutes les 10 générations
    )
    
    print(f"Population: {tsp_ga.taille_population} routes")
    print(f"Élitisme: {tsp_ga.nb_elites} meilleures routes conservées")
    print(f"Mutation: {tsp_ga.taux_mutation*100:.1f}%")
    print(f"Meilleure distance initiale: {tsp_ga.meilleure_distance:.2f}")
    print(f"\nLancement de l'optimisation (500 générations)...")
    print("Raccourcis: 'r' = top routes, 'p' = matrice OD, 'ESC' = quitter")
    
    # Exécute l'algorithme génétique
    tsp_ga.executer(nb_iterations=1000, delai_ms=50)
