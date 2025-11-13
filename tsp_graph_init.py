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
    
    Implémente les comparaisons pour tri direct par fitness.
    """
    __slots__ = ('ordre', '_fitness')
    
    def __init__(self, ordre=None, nombre_lieux=None):
        if ordre is not None:
            # Normaliser: enlever les 0 internes, garder structure [0, ..., 0]
            core = [int(i) for i in list(ordre) if int(i) != 0]
            self.ordre = [0] + core + [0] if core else [0]
        elif nombre_lieux is not None:
            self.ordre = [0] + list(range(1, nombre_lieux)) + [0]
        else:
            self.ordre = [0]
        
        self._fitness = None  # Cache pour la fitness (distance)
    
    def __eq__(self, other):
        """Égalité basée sur la séquence d'ordre"""
        if not isinstance(other, Route):
            return NotImplemented
        return self.ordre == other.ordre
    
    def __lt__(self, other):
        """Inférieur = meilleure fitness (distance plus courte)"""
        if not isinstance(other, Route):
            return NotImplemented
        # Sécurité: retourne False si fitness non calculée (au lieu de raise)
        if self._fitness is None or other._fitness is None:
            return False
        return self._fitness < other._fitness
    
    def __gt__(self, other):
        """Supérieur = pire fitness (distance plus longue)"""
        if not isinstance(other, Route):
            return NotImplemented
        # Sécurité: retourne False si fitness non calculée
        if self._fitness is None or other._fitness is None:
            return False
        return self._fitness > other._fitness
    
    def __repr__(self):
        """Représentation pour debug"""
        if self._fitness is not None:
            return f"Route(dist={self._fitness:.2f}, ordre={self.ordre[:5]}...{self.ordre[-2:]})"
        return f"Route(ordre={self.ordre[:5]}...{self.ordre[-2:]})"


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
        Optimisé pour VM 8GB RAM / 4 cœurs.
        """
        n = len(self.liste_lieux)
        
        if n == 0:
            self.matrice_od = np.zeros((0, 0), dtype=np.float64)
            return self.matrice_od
        
        dtype = np.float32 if n >= 1000 else np.float64
        itemsize = np.dtype(dtype).itemsize
        
        # Extraction vectorisée des coordonnées (C-contigu)
        X = np.ascontiguousarray(np.array([[lieu.x, lieu.y] for lieu in self.liste_lieux], dtype=dtype))

        # Normes au carré ||xi||^2
        s = np.einsum('ij,ij->i', X, X, dtype=dtype)
        
        # memmap si la matrice dépasse ~2GB (avec 8GB RAM disponible)
        # Garde ~6GB pour l'OS, Python et le reste du programme
        total_bytes = n * n * itemsize
        use_memmap = total_bytes > 2 * 1024 * 1024 * 1024
        if use_memmap:
            mmap_path = "matrice_od_mmap.dat"
            D = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(n, n))
        else:
            D = np.empty((n, n), dtype=dtype)
        


        # Tuilage (blocs) + symétrie
        block = 1024 if n >= 2048 else (512 if n >= 1024 else n)
        for i0 in range(0, n, block):
            i1 = min(i0 + block, n)
            Xi = X[i0:i1] 
            si = s[i0:i1] 
            for j0 in range(i0, n, block):
                j1 = min(j0 + block, n)
                Xj = X[j0:j1] 
                sj = s[j0:j1] 
                
                # Produit de Gram pour le bloc
                G = Xi @ Xj.T 

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
        """
        # Calcule la matrice si absente
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        
        n = self.matrice_od.shape[0]
        if n == 0:
            return None
        
        distances = self.matrice_od[lieu_idx].copy()
        distances[lieu_idx] = np.inf
        
        return int(np.argmin(distances))
    
    def construire_route_plus_proche_voisin(self, depart=0):
        """
        Construit une route complète avec l'heuristique du plus proche voisin.
        Utile pour initialisation semi-greedy du GA.
        
        :param depart: Indice du lieu de départ (par défaut 0)
        :return: Route complète construite par PPN
        """
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        
        n = len(self.liste_lieux)
        if n == 0:
            return None
        
        ordre = [depart]
        visites = {depart}
        
        # Construit la route lieu par lieu (plus proche non visité)
        for _ in range(n - 1):
            dernier = ordre[-1]
            distances = np.array(self.matrice_od[dernier], dtype=float)
            
            # Exclut les lieux déjà visités
            for idx in visites:
                distances[idx] = np.inf
            
            prochain = int(np.argmin(distances))
            ordre.append(prochain)
            visites.add(prochain)
        
        # Retour au dépôt (lieu 0)
        if depart != 0:
            # Si départ ≠ 0, on reconstruit en commençant par 0
            idx_zero = ordre.index(0)
            ordre = ordre[idx_zero:] + ordre[:idx_zero] + [0]
        else:
            ordre.append(0)
        
        return Route(ordre)
    
    def plus_proches_voisins_k(self, lieu_idx, k=20):
        """
        Retourne les k indices des lieux les plus proches de lieu_idx.
        Utilisé pour 2-opt restreint (plus rapide).
        
        :param lieu_idx: Indice du lieu de référence
        :param k: Nombre de voisins à retourner
        :return: Array des k indices les plus proches
        """
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        
        n = self.matrice_od.shape[0]
        if n <= 1 or k <= 0:
            return np.array([], dtype=int)
        
        k = min(k, n - 1)  # Ne peut pas retourner plus que n-1
        
        # Copie défensive
        d = self.matrice_od[lieu_idx].copy()
        d[lieu_idx] = np.inf  # Exclut soi-même
        
        # Utilise argpartition pour extraire les k plus petits (O(n))
        idx_k = np.argpartition(d, k)[:k]
        
        # Trie ces k indices par distance croissante
        idx_k = idx_k[np.argsort(d[idx_k])]
        
        return idx_k
    
    def calcul_distance_route(self, route):
        """
        Calcule la distance totale d'une route (somme des arêtes successives).
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
        
        # indexation vectorielle de la matrice
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
                 'afficher_top_routes', 'afficher_pheromones')
    
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
        
        self.afficher_top_routes = True  # Renommé pour clarté
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
            self.root.bind("t", self._on_toggle_top_routes)
            self.root.bind("T", self._on_toggle_top_routes)
            self.root.bind("p", self._on_toggle_pheromones)
            self.root.bind("P", self._on_toggle_pheromones)
            
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
        """
        Met à jour les N meilleures routes et redessine.
        Filtre pour n'afficher que des routes uniques et différentes de la meilleure.
        """
        # Déduplication stricte: ne garder que des routes uniques
        routes_uniques = []
        
        # Exclure la meilleure route globale (utilise __eq__)
        for route in routes:
            # Ajouter seulement si unique ET différente de la meilleure
            if self.meilleure_route and route == self.meilleure_route:  # Utilise __eq__ !
                continue
            
            # Vérifie unicité avec __eq__
            deja_presente = any(route == r for r in routes_uniques)  # Utilise __eq__ !
            if not deja_presente:
                routes_uniques.append(route)
                if len(routes_uniques) >= self.n_top_routes:
                    break
        
        self.top_routes = routes_uniques
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
    
    def _on_toggle_top_routes(self, _evt=None):
        """Raccourci 'T': affiche/masque les N meilleures routes grises."""
        self.afficher_top_routes = not self.afficher_top_routes
        status = "affichées" if self.afficher_top_routes else "masquées"
        print(f"Routes grises (top {self.n_top_routes}): {status}")
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
        
        # Dessine les N meilleures routes en gris clair (si activé)
        if self.afficher_top_routes:
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
        Crée une population initiale semi-greedy: 10% PPN + 90% aléatoire.
        Améliore la convergence sans perdre la diversité.
        """
        nb_lieux = len(self.graph.liste_lieux)
        if nb_lieux < 2:
            return
        
        # 10% avec heuristique Plus Proche Voisin (convergence rapide)
        nb_ppn = max(1, int(self.taille_population * 0.10))
        print(f"  Initialisation: {nb_ppn} routes PPN + {self.taille_population - nb_ppn} routes aléatoires")
        
        for i in range(nb_ppn):
            # Différents points de départ pour diversité
            depart = i % nb_lieux
            route_ppn = self.graph.construire_route_plus_proche_voisin(depart)
            if route_ppn:
                self.population.append(route_ppn)
        
        # 90% routes aléatoires (diversité et exploration)
        indices_base = list(range(1, nb_lieux))
        while len(self.population) < self.taille_population:
            indices = indices_base.copy()
            random.shuffle(indices)
            ordre = [0] + indices + [0]
            self.population.append(Route(ordre))
        
        # Évalue la population initiale
        self._evaluer_population()
        print(f"  Meilleure distance initiale (PPN): {self.meilleure_distance:.2f}")
        print(f"  Top 3: {self.population[0]}, {self.population[1]}, {self.population[2]}")  # Utilise __repr__ !
    
    def _evaluer_population(self):
        """
        Calcule la fitness de chaque route et trie directement avec __lt__.
        Utilise les dunders de comparaison pour tri élégant.
        """
        # Calcule et stocke la fitness dans chaque route
        for route in self.population:
            route._fitness = self.graph.calcul_distance_route(route)
        
        # Tri direct avec __lt__ (route1 < route2 si distance1 < distance2)
        self.population.sort()  # Utilise __lt__ automatiquement !
        
        # Met à jour la meilleure solution globale
        if self.population and self.population[0]._fitness < self.meilleure_distance:
            self.meilleure_distance = self.population[0]._fitness
            self.meilleure_route = self.population[0]
            self.iteration_meilleure = self.iteration_courante
    
    def _selection_tournoi(self, taille_tournoi=3):
        """
        Sélection par tournoi: tire K routes aléatoires, retourne la meilleure.
        Utilise __lt__ pour comparaison directe.
        """
        taille = min(taille_tournoi, len(self.population))
        candidats = random.sample(self.population, taille)
        
        # Calcule fitness si nécessaire (pour routes nouvellement créées)
        for route in candidats:
            if route._fitness is None:
                route._fitness = self.graph.calcul_distance_route(route)
        
        # Trouve le meilleur candidat avec __lt__ (min utilise < automatiquement !)
        meilleur = min(candidats)  # Utilise __lt__ !
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
        
        # Mutation plus agressive: au moins 2 swaps pour maintenir diversité
        nb_swaps = max(2, int(len(ordre) * self.taux_mutation))
        for _ in range(nb_swaps):
            i, j = random.sample(range(len(ordre)), 2)
            ordre[i], ordre[j] = ordre[j], ordre[i]
        
        # Reconstruit la route avec les 0
        route.ordre = [0] + ordre + [0]
    
    def _amelioration_2opt_knn(self, route, k=20, max_iterations=50):
        """
        Amélioration 2-opt restreinte aux k plus proches voisins.
        Beaucoup plus rapide que 2-opt complet: O(n×k) au lieu de O(n²).
        
        :param route: Route à améliorer
        :param k: Nombre de voisins à considérer (défaut 20)
        :param max_iterations: Nombre max d'itérations (défaut 50)
        :return: Route améliorée
        """
        if route is None:
            return route
        
        meilleure_route = Route(ordre=route.ordre.copy())
        meilleure_distance = self.graph.calcul_distance_route(meilleure_route)
        
        # Pré-calcul des k plus proches voisins pour chaque lieu
        n = len(self.graph.liste_lieux)
        k_voisins = {}
        for lieu in range(n):
            k_voisins[lieu] = set(self.graph.plus_proches_voisins_k(lieu, k))
        
        ameliore = True
        iterations = 0
        
        while ameliore and iterations < max_iterations:
            ameliore = False
            iterations += 1
            ordre = meilleure_route.ordre[1:-1]  # Sans les 0
            n_ordre = len(ordre)
            
            for i in range(n_ordre - 1):
                lieu_i = ordre[i]
                
                # Ne teste que les j où ordre[j] est dans les k-NN de lieu_i
                for j in range(i + 2, n_ordre):
                    lieu_j = ordre[j]
                    
                    # Optimisation: teste seulement si proche voisin
                    if lieu_j not in k_voisins[lieu_i]:
                        continue
                    
                    # Test 2-opt: inverse le segment [i+1:j+1]
                    nouveau_ordre = ordre[:i+1] + ordre[i+1:j+1][::-1] + ordre[j+1:]
                    nouvelle_route = Route([0] + nouveau_ordre + [0])
                    nouvelle_distance = self.graph.calcul_distance_route(nouvelle_route)
                    
                    if nouvelle_distance < meilleure_distance - 1e-6:
                        meilleure_route = nouvelle_route
                        meilleure_distance = nouvelle_distance
                        ordre = nouveau_ordre
                        ameliore = True
                        break
                
                if ameliore:
                    break
        
        return meilleure_route
    
    def _nouvelle_generation(self):
        """
        Crée une nouvelle génération via élitisme, croisement, mutation et immigrants aléatoires.
        """
        nouvelle_pop = []
        
        # Élitisme: conserve les nb_elites meilleures routes
        nouvelle_pop.extend(self.population[:self.nb_elites])
        
        # Injecter des immigrants aléatoires (10% de la population) pour maintenir diversité
        nb_immigrants = max(2, int(self.taille_population * 0.10))
        nb_lieux = len(self.graph.liste_lieux)
        for _ in range(nb_immigrants):
            indices = list(range(1, nb_lieux))
            random.shuffle(indices)
            nouvelle_pop.append(Route([0] + indices + [0]))
        
        # Génère le reste par croisement et mutation
        while len(nouvelle_pop) < self.taille_population:
            # Sélection des parents
            parent1 = self._selection_tournoi()
            parent2 = self._selection_tournoi()
            
            # Croisement
            enfant = self._croisement_ox(parent1, parent2)
            
            # Mutation (toujours appliquer pour maintenir diversité)
            self._mutation_swap(enfant)
            
            nouvelle_pop.append(enfant)
        
        self.population = nouvelle_pop
    
    def _mettre_a_jour_affichage(self):
        """
        Met à jour l'affichage avec les meilleures routes et informations.
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
        msg += "Raccourcis: 'T' = toggle routes grises, 'P' = matrice OD, 'ESC' = quitter"
        self.affichage.set_message(msg)
    
    def executer(self, nb_iterations=500, delai_ms=50):
        """
        Exécute l'algorithme génétique pour un nombre d'itérations donné.
        
        :param nb_iterations: Nombre de générations à effectuer
        :param delai_ms: Délai en ms entre updates GUI (si pas headless)
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
                    
                    # Amélioration 2-opt k-NN sur la meilleure route (toutes les 20 générations)
                    if self.iteration_courante % 20 == 0 and self.meilleure_route:
                        route_amelioree = self._amelioration_2opt_knn(self.meilleure_route, k=20)
                        distance_amelioree = self.graph.calcul_distance_route(route_amelioree)
                        if distance_amelioree < self.meilleure_distance:
                            print(f"  2-opt k-NN amélioration (gen {self.iteration_courante}): {self.meilleure_distance:.2f} → {distance_amelioree:.2f}")
                            print(f"    Nouvelle meilleure: {route_amelioree}")  # Utilise __repr__ !
                            self.meilleure_distance = distance_amelioree
                            self.meilleure_route = route_amelioree
                            # IMPORTANT: injecter dans la population pour propagation
                            self.population[0] = route_amelioree
                    
                    # Update GUI selon l'intervalle
                    if self.iteration_courante % self.update_interval == 0:
                        self._mettre_a_jour_affichage()
                    
                    # Planifie la prochaine itération
                    self.affichage.root.after(delai_ms, iteration)
                else:
                    # Amélioration finale 2-opt k-NN
                    if self.meilleure_route:
                        print("\nAmélioration 2-opt k-NN finale...")
                        route_amelioree = self._amelioration_2opt_knn(self.meilleure_route, k=30, max_iterations=100)
                        distance_amelioree = self.graph.calcul_distance_route(route_amelioree)
                        if distance_amelioree < self.meilleure_distance:
                            print(f"✓ Amélioration finale: {self.meilleure_distance:.2f} → {distance_amelioree:.2f}")
                            self.meilleure_distance = distance_amelioree
                            self.meilleure_route = route_amelioree
                    
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
    
    g = Graph()
    
    # g.charger_graph("tests/data/graph_5.csv")
    g.charger_graph("tests/data/graph_20.csv")
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
    # Paramètres optimisés pour VM 8GB RAM / 4 cœurs
    tsp_ga = TSP_GA(
        graph=g,
        affichage=ui,
        taille_population=1000,      # Bon compromis qualité/RAM (< 10 MB)
        taux_mutation=0.15,         # 15% pour maintenir diversité
        taux_elitisme=0.05,         # 5% seulement (moins de convergence)
        update_interval=5           # Update plus fréquent pour visualisation
    )

    tsp_ga.executer(nb_iterations=600, delai_ms=0)

