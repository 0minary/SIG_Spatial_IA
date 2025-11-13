import numpy as np
import random
import time
import csv
from tsp_graph_init import Graph, Route, Affichage


class TSP_GA:
    """Classe implémentant un algorithme génétique pour le TSP"""
    
    def __init__(self, graph, taille_population=50, taux_mutation=0.1, taux_crossover=0.8, 
                 taux_elitisme=0.1, nb_meilleures_affichees=5, seed=42):
        """
        Initialise l'algorithme génétique
        
        Parameters:
        - seed: seed pour la reproductibilité (None pour aléatoire)
        """
        # Reproductibilité
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.seed = seed
        
        self.graph = graph
        self.taille_population = taille_population
        self.taux_mutation = taux_mutation
        self.taux_crossover = taux_crossover
        self.taux_elitisme = taux_elitisme
        self.nb_meilleures_affichees = nb_meilleures_affichees
        
        self.nb_lieux = len(graph.liste_lieux)
        
        # Initialiser la population (avec amorçage heuristique)
        self.population = self.initialiser_population()
        
        # Meilleure route trouvée
        self.meilleure_route = None
        self.meilleure_distance = float('inf')
        self.iterations_meilleure = 0
    
    def _route_est_valide(self, route, nb_lieux):
        """Vérifie qu'une route est valide (méthode locale car pas dans Route)"""
        if len(route.ordre) != nb_lieux + 1:
            return False
        if route.ordre[0] != 0 or route.ordre[-1] != 0:
            return False
        lieux_visites = route.ordre[1:-1]
        if len(lieux_visites) != nb_lieux - 1:
            return False
        if set(lieux_visites) != set(range(1, nb_lieux)):
            return False
        return True
    
    def generer_route_aleatoire(self):
        """Génère une route aléatoire valide"""
        lieux_intermediaires = list(range(1, self.nb_lieux))
        random.shuffle(lieux_intermediaires)
        ordre = [0] + lieux_intermediaires + [0]
        r = Route(ordre)
        assert self._route_est_valide(r, self.nb_lieux), "Route invalide générée (aléatoire)"
        return r
    
    def _construire_route_plus_proche_voisin(self, depart_idx=0):
        """Construit une route avec l'heuristique du plus proche voisin (méthode locale)"""
        if self.graph.matrice_od is None:
            self.graph.calcul_matrice_cout_od()
        n = len(self.graph.liste_lieux)
        if n == 0:
            return None
        
        ordre = [depart_idx]
        visites = {depart_idx}
        
        for _ in range(n - 1):
            dernier = ordre[-1]
            distances = np.array(self.graph.matrice_od[dernier], dtype=float)
            for idx in visites:
                distances[idx] = np.inf
            prochain = int(np.argmin(distances))
            ordre.append(prochain)
            visites.add(prochain)
        
        ordre.append(depart_idx)
        return Route(ordre)
    
    def initialiser_population(self):
        """Initialisation mixte optimale : 10% heuristique + 90% aléatoire"""
        population = []

        # 10% avec plus proche voisin pour un bon départ
        nb_heuristiques = max(1, int(self.taille_population * 0.10))
        for i in range(nb_heuristiques):
            route_ppv = self._construire_route_plus_proche_voisin(depart_idx=i % self.nb_lieux)
            if route_ppv:
                population.append(route_ppv)

        # Compléter avec routes aléatoires pour diversité
        while len(population) < self.taille_population:
            population.append(self.generer_route_aleatoire())

        return population
    
    def evaluer_population(self):
        """Évalue la population et retourne les distances"""
        distances = []
        for route in self.population:
            distance = self.graph.calcul_distance_route(route)
            distances.append(distance)
        return distances
    
    def selection_tournoi(self, distances, taille_tournoi=3):
        """Sélection par tournoi"""
        indices = random.sample(range(len(self.population)), taille_tournoi)
        gagnant = min(indices, key=lambda i: distances[i])
        return self.population[gagnant]
    
    # ----- Opérateurs génétiques -----
    
    def _repare_enfant(self, seg):
        """
        Répare un segment en supprimant les doublons et en ajoutant les villes manquantes.
        seg: liste des villes (sans 0)
        """
        attendu = set(range(1, self.nb_lieux))
        vu = set()
        propre = []
        for v in seg:
            if v in attendu and v not in vu:
                propre.append(v)
                vu.add(v)
        manquants = [v for v in range(1, self.nb_lieux) if v not in vu]
        return propre + manquants
    
    def crossover_ordre(self, parent1, parent2):
        """Croisement par ordre (Order Crossover - OX) robuste pour le TSP"""
        seg1 = parent1.ordre[1:-1]
        seg2 = parent2.ordre[1:-1]
        
        taille = len(seg1)
        if taille < 2:
            return Route(parent1.ordre.copy()), Route(parent2.ordre.copy())
        
        p1, p2 = sorted(random.sample(range(taille), 2))
        
        # Enfant 1
        bloc = seg1[p1:p2+1]
        restant = [x for x in seg2 if x not in bloc]
        e1_seg = restant[:p1] + bloc + restant[p1:]
        e1_seg = self._repare_enfant(e1_seg)
        enfant1 = Route([0] + e1_seg + [0])
        
        # Enfant 2
        bloc = seg2[p1:p2+1]
        restant = [x for x in seg1 if x not in bloc]
        e2_seg = restant[:p1] + bloc + restant[p1:]
        e2_seg = self._repare_enfant(e2_seg)
        enfant2 = Route([0] + e2_seg + [0])
        
        assert self._route_est_valide(enfant1, self.nb_lieux), "Enfant 1 invalide (OX)"
        assert self._route_est_valide(enfant2, self.nb_lieux), "Enfant 2 invalide (OX)"
        return enfant1, enfant2
    
    def mutation_swap(self, route):
        """Mutation par échange de deux lieux (hors 0)"""
        seg = route.ordre[1:-1].copy()
        if len(seg) < 2:
            return route
        idx1, idx2 = random.sample(range(len(seg)), 2)
        seg[idx1], seg[idx2] = seg[idx2], seg[idx1]
        r = Route([0] + seg + [0])
        assert self._route_est_valide(r, self.nb_lieux), "Route invalide après swap"
        return r
    
    def mutation_inversion(self, route):
        """Mutation par inversion d'un segment"""
        seg = route.ordre[1:-1].copy()
        if len(seg) < 2:
            return route
        i, j = sorted(random.sample(range(len(seg)), 2))
        seg[i:j+1] = reversed(seg[i:j+1])
        r = Route([0] + seg + [0])
        assert self._route_est_valide(r, self.nb_lieux), "Route invalide après inversion"
        return r
    
    def mutation_2opt(self, route):
        """Mutation locale 2-opt (très efficace pour TSP)"""
        seg = route.ordre[1:-1]
        n = len(seg)
        if n < 4:
            return route
        i, k = sorted(random.sample(range(n), 2))
        seg2 = seg[:i] + list(reversed(seg[i:k+1])) + seg[k+1:]
        r = Route([0] + seg2 + [0])
        assert self._route_est_valide(r, self.nb_lieux), "Route invalide après 2-opt"
        return r
    
    def post_traitement_2opt_local(self, route):
        """Version corrigée avec logging pour debug"""
        if route is None:
            return route
        
        meilleure_route = Route(route.ordre.copy())
        meilleure_distance = self.graph.calcul_distance_route(meilleure_route)
        
        ameliore = True
        iterations = 0
        max_iterations = 100
        
        while ameliore and iterations < max_iterations:
            ameliore = False
            iterations += 1
            ordre = meilleure_route.ordre[1:-1]  # Exclure les 0
            n = len(ordre)
            
            for i in range(n - 1):
                for j in range(i + 2, n):
                    nouveau_ordre = ordre[:i+1] + ordre[i+1:j+1][::-1] + ordre[j+1:]
                    nouvelle_route = Route([0] + nouveau_ordre + [0])
                    nouvelle_distance = self.graph.calcul_distance_route(nouvelle_route)
                    
                    if nouvelle_distance < meilleure_distance - 1e-6:
                        print(f"  2-opt amélioration: {meilleure_distance:.2f} → {nouvelle_distance:.2f}")
                        meilleure_route = nouvelle_route
                        meilleure_distance = nouvelle_distance
                        ordre = nouveau_ordre
                        ameliore = True
                        break
                if ameliore:
                    break
        
        return meilleure_route
    
    def post_traitement_3opt_local(self, route):
        """Post-traitement 3-opt pour explorer un voisinage plus large que 2-opt"""
        if route is None:
            return route
        
        meilleure_route = Route(route.ordre.copy())
        meilleure_distance = self.graph.calcul_distance_route(meilleure_route)
        
        seg = meilleure_route.ordre[1:-1]
        n = len(seg)
        
        if n < 6:
            return meilleure_route
        
        ameliore = True
        iterations_max = 50  # Limiter pour éviter trop de calculs
        
        while ameliore and iterations_max > 0:
            ameliore = False
            iterations_max -= 1
            
            # Tester les swaps 3-opt
            for i in range(n - 4):
                for j in range(i + 2, n - 2):
                    for k in range(j + 2, n):
                        # Tester les différentes combinaisons 3-opt
                        # Type 1: inverser segment j-k
                        nouveau_seg1 = seg[:i+1] + seg[i+1:j+1] + list(reversed(seg[j+1:k+1])) + seg[k+1:]
                        route1 = Route([0] + nouveau_seg1 + [0])
                        dist1 = self.graph.calcul_distance_route(route1)
                        
                        # Type 2: inverser segment i-j et j-k
                        nouveau_seg2 = seg[:i+1] + list(reversed(seg[i+1:j+1])) + list(reversed(seg[j+1:k+1])) + seg[k+1:]
                        route2 = Route([0] + nouveau_seg2 + [0])
                        dist2 = self.graph.calcul_distance_route(route2)
                        
                        # Type 3: réorganiser les segments
                        nouveau_seg3 = seg[:i+1] + seg[j+1:k+1] + seg[i+1:j+1] + seg[k+1:]
                        route3 = Route([0] + nouveau_seg3 + [0])
                        dist3 = self.graph.calcul_distance_route(route3)
                        
                        # Prendre la meilleure amélioration
                        meilleure_dist_candidate = min(dist1, dist2, dist3)
                        if meilleure_dist_candidate < meilleure_distance:
                            if dist1 == meilleure_dist_candidate:
                                meilleure_route = route1
                                seg = nouveau_seg1
                            elif dist2 == meilleure_dist_candidate:
                                meilleure_route = route2
                                seg = nouveau_seg2
                            else:
                                meilleure_route = route3
                                seg = nouveau_seg3
                            meilleure_distance = meilleure_dist_candidate
                            ameliore = True
                            break
                    if ameliore:
                        break
                if ameliore:
                    break
        
        return meilleure_route
    
    def mutation(self, route):
        """Applique une mutation au hasard parmi 3 opérateurs"""
        r = random.random()
        if r < 0.34:
            return self.mutation_swap(route)
        elif r < 0.67:
            return self.mutation_inversion(route)
        else:
            return self.mutation_2opt(route)
    
    # ---------------------------------
    
    def executer(self, nb_iterations=200, affichage=None, delai=0.1, early_stopping=50, log_csv=None):
        """
        Exécute l'algorithme génétique
        
        Parameters:
        - nb_iterations: nombre maximum d'itérations
        - affichage: instance de Affichage pour l'affichage graphique
        - delai: délai entre les itérations (en secondes)
        - early_stopping: nombre de générations sans amélioration avant arrêt (None pour désactiver)
        - log_csv: nom du fichier CSV pour logger les générations (None pour désactiver)
        """
        nb_elites = max(1, int(self.taille_population * self.taux_elitisme))
        
        # Initialiser la meilleure route avec la meilleure de la population initiale
        if self.meilleure_route is None and len(self.population) > 0:
            distances_init = self.evaluer_population()
            meilleur_idx_init = np.argmin(distances_init)
            self.meilleure_distance = distances_init[meilleur_idx_init]
            self.meilleure_route = self.population[meilleur_idx_init]
        
        # Variables pour early stopping
        generations_sans_amelioration = 0
        
        # Stratégie de diversification pour éviter les minima locaux
        dernier_diversification = 0
        periode_diversification = 50  # Diversifier toutes les 50 générations
        
        # Initialiser le log CSV
        log_file = None
        if log_csv:
            try:
                log_file = open(log_csv, 'w', newline='')
                writer = csv.writer(log_file)
                writer.writerow(['generation', 'distance_min', 'distance_max', 'distance_moyenne', 'meilleure_distance'])
            except Exception as e:
                print(f"Erreur lors de l'ouverture du fichier de log: {e}")
                log_file = None
        
        for iteration in range(nb_iterations):
            # Évaluer la population
            distances = self.evaluer_population()
            
            # Trouver la meilleure route de cette génération
            meilleur_idx = np.argmin(distances)
            meilleure_dist_iteration = distances[meilleur_idx]
            meilleure_route_iteration = self.population[meilleur_idx]
            
            # Appliquer 2-opt local pour intensification (config optimale)
            # (toutes les 20 générations sur la meilleure)
            if iteration % 20 == 0:
                route_amelioree = self.post_traitement_2opt_local(meilleure_route_iteration)
                distance_amelioree = self.graph.calcul_distance_route(route_amelioree)
                if distance_amelioree < meilleure_dist_iteration:
                    meilleure_dist_iteration = distance_amelioree
                    meilleure_route_iteration = route_amelioree
                    self.population[meilleur_idx] = route_amelioree
                    distances[meilleur_idx] = distance_amelioree
            
            # Mettre à jour la meilleure route globale
            if meilleure_dist_iteration < self.meilleure_distance:
                self.meilleure_distance = meilleure_dist_iteration
                self.meilleure_route = meilleure_route_iteration
                self.iterations_meilleure = iteration + 1
                generations_sans_amelioration = 0
                dernier_diversification = 0  # Reset diversification
            else:
                generations_sans_amelioration += 1
            
            # Stratégie de diversification si stagnation
            if generations_sans_amelioration > 0 and (generations_sans_amelioration - dernier_diversification) >= periode_diversification:
                print(f"  [Diversification à la génération {iteration + 1}]")
                # Diversifier la population : remplacer 70% des pires individus
                indices_tries = sorted(range(len(distances)), key=lambda i: distances[i])
                nb_a_remplacer = max(1, int(self.taille_population * 0.7))  # 70% pour diversifier plus agressivement

                # Remplacer massivement par des routes aléatoires pour vraiment diversifier
                for idx_pire in indices_tries[-nb_a_remplacer:]:
                    # 70% de routes complètement aléatoires, 30% mutées de la meilleure
                    if random.random() < 0.7:
                        self.population[idx_pire] = self.generer_route_aleatoire()
                    else:
                        # Variante fortement mutée de la meilleure route (plusieurs mutations)
                        route_mutee = Route(self.meilleure_route.ordre.copy())
                        for _ in range(random.randint(5, 10)):  # 5 à 10 mutations successives
                            route_mutee = self.mutation(route_mutee)
                        self.population[idx_pire] = route_mutee
                
                dernier_diversification = generations_sans_amelioration
                # Réévaluer après diversification
                distances = self.evaluer_population()
                meilleur_idx = np.argmin(distances)
                meilleure_dist_iteration = distances[meilleur_idx]
                meilleure_route_iteration = self.population[meilleur_idx]
                
                # Vérifier si la diversification a amélioré
                if meilleure_dist_iteration < self.meilleure_distance:
                    self.meilleure_distance = meilleure_dist_iteration
                    self.meilleure_route = meilleure_route_iteration
                    self.iterations_meilleure = iteration + 1
                    generations_sans_amelioration = 0
                    dernier_diversification = 0
            
            # Créer la nouvelle génération
            nouvelle_population = []
            
            # Élitisme: garder les meilleures routes (copie défensive)
            indices_tries = sorted(range(len(distances)), key=lambda i: distances[i])
            for i in range(nb_elites):
                elite = self.population[indices_tries[i]]
                nouvelle_population.append(Route(elite.ordre))  # copie défensive
            
            # Générer le reste de la population
            while len(nouvelle_population) < self.taille_population:
                # Sélection
                parent1 = self.selection_tournoi(distances)
                parent2 = self.selection_tournoi(distances)
                
                # Croisement
                if random.random() < self.taux_crossover:
                    enfant1, enfant2 = self.crossover_ordre(parent1, parent2)
                else:
                    enfant1, enfant2 = Route(parent1.ordre), Route(parent2.ordre)
                
                # Mutation avec taux adaptatif (plus élevé si stagnation)
                taux_mutation_adapte = self.taux_mutation
                if generations_sans_amelioration > 50:  # Adaptive mutation rate
                    taux_mutation_adapte = min(0.6, self.taux_mutation * (1 + generations_sans_amelioration / 150))  # Jusqu'à 60%
                
                if random.random() < taux_mutation_adapte:
                    enfant1 = self.mutation(enfant1)
                if random.random() < taux_mutation_adapte:
                    enfant2 = self.mutation(enfant2)
                
                nouvelle_population.append(enfant1)
                if len(nouvelle_population) < self.taille_population:
                    nouvelle_population.append(enfant2)
            
            # Limiter à la taille de population
            self.population = nouvelle_population[:self.taille_population]
            
            # Calculer les statistiques
            distance_moyenne = np.mean(distances)
            distance_min = np.min(distances)
            distance_max = np.max(distances)
            
            # Logger dans CSV
            if log_file:
                try:
                    writer = csv.writer(log_file)
                    writer.writerow([
                        iteration + 1,
                        distance_min,
                        distance_max,
                        distance_moyenne,
                        self.meilleure_distance
                    ])
                except Exception as e:
                    print(f"Erreur lors de l'écriture dans le log: {e}")
            
            # Obtenir les N meilleures routes pour l'affichage (exclure la meilleure si elle est dedans)
            indices_tries = sorted(range(len(distances)), key=lambda i: distances[i])
            meilleures_routes = []
            meilleure_ordre_global = self.meilleure_route.ordre if self.meilleure_route else None
            
            for i in indices_tries[:self.nb_meilleures_affichees]:
                route_candidate = self.population[i]
                # Ajouter seulement si différente de la meilleure route globale
                if meilleure_ordre_global is None or route_candidate.ordre != meilleure_ordre_global:
                    meilleures_routes.append(route_candidate)
                elif len(meilleures_routes) < self.nb_meilleures_affichees:
                    # Si la meilleure est dans les top N, prendre la suivante
                    continue
            
            # Si on n'a pas assez de routes différentes, compléter avec d'autres bonnes routes
            idx = self.nb_meilleures_affichees
            while len(meilleures_routes) < self.nb_meilleures_affichees and idx < len(indices_tries):
                route_candidate = self.population[indices_tries[idx]]
                if route_candidate.ordre != meilleure_ordre_global:
                    if route_candidate.ordre not in [r.ordre for r in meilleures_routes]:
                        meilleures_routes.append(route_candidate)
                idx += 1
            
            # Affichage
            if affichage:
                info_texte = (
                    f"Génération: {iteration + 1}/{nb_iterations}\n"
                    f"Meilleure distance trouvée: {self.meilleure_distance:.2f}\n"
                    f"Trouvée à la génération: {self.iterations_meilleure}\n"
                    f"Distance génération courante - Min: {distance_min:.2f}, "
                    f"Max: {distance_max:.2f}, Moyenne: {distance_moyenne:.2f}\n"
                    f"Taille population: {self.taille_population}\n"
                    f"Paramètres: Mutation={self.taux_mutation}, "
                    f"Crossover={self.taux_crossover}, Élitisme={self.taux_elitisme}\n"
                    f"Appuyez sur 'P' pour afficher/masquer les {self.nb_meilleures_affichees} meilleures routes\n"
                    f"Appuyez sur 'S' pour sauvegarder la meilleure route\n"
                    f"Appuyez sur 'ESC' pour quitter"
                )
                
                # Mise à jour de l'affichage avec les méthodes disponibles
                affichage.set_meilleure_route(self.meilleure_route)
                affichage.set_top_routes(meilleures_routes)
                affichage.set_message(info_texte)
                
                time.sleep(delai)
            
            # Restart complet si vraiment bloqué (désactivé pour laisser l'algorithme explorer plus longtemps)
            # if generations_sans_amelioration >= 200:
            #     print("  [RESTART COMPLET - Réinitialisation population]")
            #     self.population = self.initialiser_population()
            #     generations_sans_amelioration = 0
            #     dernier_diversification = 0
            
            # Early stopping
            if early_stopping is not None and generations_sans_amelioration >= early_stopping:
                print(f"\nArrêt anticipé: aucune amélioration depuis {early_stopping} générations")
                break
        
        # Fermer le fichier de log
        if log_file:
            log_file.close()
            print(f"✓ Log sauvegardé dans {log_csv}")
        
        # S'assurer qu'une route a été trouvée
        if self.meilleure_route is None and len(self.population) > 0:
            distances_final = self.evaluer_population()
            meilleur_idx_final = np.argmin(distances_final)
            self.meilleure_distance = distances_final[meilleur_idx_final]
            self.meilleure_route = self.population[meilleur_idx_final]
        
        # Post-traitement 2-opt puis 3-opt local pour améliorer la meilleure solution
        if self.meilleure_route:
            print("\nPost-traitement 2-opt local en cours...")
            route_amelioree = self.post_traitement_2opt_local(self.meilleure_route)
            distance_amelioree = self.graph.calcul_distance_route(route_amelioree)
            
            if distance_amelioree < self.meilleure_distance:
                print(f"✓ Amélioration 2-opt: {self.meilleure_distance:.2f} → {distance_amelioree:.2f}")
                self.meilleure_route = route_amelioree
                self.meilleure_distance = distance_amelioree
            else:
                print(f"  Pas d'amélioration avec 2-opt")
            
            # Essayer 3-opt pour explorer un voisinage plus large
            print("Post-traitement 3-opt local en cours...")
            route_3opt = self.post_traitement_3opt_local(self.meilleure_route)
            distance_3opt = self.graph.calcul_distance_route(route_3opt)
            
            if distance_3opt < self.meilleure_distance:
                print(f"✓ Amélioration 3-opt: {self.meilleure_distance:.2f} → {distance_3opt:.2f}")
                self.meilleure_route = route_3opt
                self.meilleure_distance = distance_3opt
            else:
                print(f"  Pas d'amélioration avec 3-opt")
        
        # Comparaison avec heuristique de référence
        print("\n--- Comparaison avec Plus Proche Voisin ---")
        route_ppv = self._construire_route_plus_proche_voisin(0)
        if route_ppv:
            distance_ppv = self.graph.calcul_distance_route(route_ppv)
            print(f"Distance Plus Proche Voisin: {distance_ppv:.2f}")
            print(f"Votre meilleure solution: {self.meilleure_distance:.2f}")
            if distance_ppv > 0:
                amelioration = ((distance_ppv - self.meilleure_distance) / distance_ppv * 100)
                print(f"Amélioration: {amelioration:.1f}%")
        
        return self.meilleure_route, self.meilleure_distance
