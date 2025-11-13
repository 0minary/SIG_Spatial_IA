"""
Fichier principal pour exécuter l'algorithme génétique (GA)
pour résoudre le problème du voyageur de commerce (TSP)
"""

import os
from tsp_graph_init import Graph, Affichage
from tsp_ga import TSP_GA


def executer_multiple_runs(graph, nb_runs=5, affichage=None):
    """
    Exécute plusieurs runs avec différents seeds et garde le meilleur résultat
    
    Parameters:
    - graph: instance de Graph
    - nb_runs: nombre de runs à effectuer
    - affichage: instance de Affichage (utilisée seulement pour le dernier run)
    """
    meilleure_route_globale = None
    meilleure_distance_globale = float('inf')
    meilleur_seed = None
    
    print(f"=== MODE COMPÉTITION : {nb_runs} runs ===\n")
    
    for run in range(nb_runs):
        seed = 42 + run * 137  # Seeds très différents pour explorer différentes zones (137 est un nombre premier)
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{nb_runs} (seed={seed})")
        print(f"{'='*60}")
        
        # Exécuter (sans affichage pour les runs intermédiaires)
        affichage_run = affichage if run == nb_runs - 1 else None
        
        # CONFIGURATION OPTIMALE POUR COMPÉTITION
        # Équilibre parfait entre vitesse et qualité
        ga = TSP_GA(
            graph,
            taille_population=400,        # Bon compromis
            taux_mutation=0.25,           # Exploration modérée
            taux_crossover=0.85,          # Exploitation forte
            taux_elitisme=0.05,           # Protège les bonnes solutions
            nb_meilleures_affichees=5,
            seed=seed
        )

        meilleure_route, meilleure_distance = ga.executer(
            nb_iterations=1000,           # Suffisant pour converger
            affichage=affichage_run,
            delai=0.01 if run < nb_runs - 1 else 0.02,
            early_stopping=200,           # Arrêt raisonnable
            log_csv="generations_log.csv" if run == nb_runs - 1 else None
        )
        
        print(f"\nRun {run + 1} terminé: distance = {meilleure_distance:.2f}")
        
        # Mettre à jour le meilleur résultat global
        if meilleure_distance < meilleure_distance_globale:
            meilleure_distance_globale = meilleure_distance
            meilleure_route_globale = meilleure_route
            meilleur_seed = seed
            print(f"✓ NOUVEAU MEILLEUR RÉSULTAT !")
        else:
            print(f"  (Meilleur actuel: {meilleure_distance_globale:.2f})")
    
    print(f"\n{'='*60}")
    print(f"RÉSULTAT FINAL (meilleur sur {nb_runs} runs):")
    print(f"Meilleure distance: {meilleure_distance_globale:.2f}")
    print(f"Trouvée avec seed: {meilleur_seed}")
    if meilleure_route_globale is not None:
        print(f"Route: {meilleure_route_globale.ordre}")
    print(f"{'='*60}\n")
    
    return meilleure_route_globale, meilleure_distance_globale


def main():
    """Fonction principale"""
    # Créer le graphe
    graph = Graph()
    
    # Option 1: Charger depuis un fichier CSV (colonnes x,y)
    # Construire le chemin relatif au répertoire du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #graph.charger_graph(os.path.join(script_dir, "tests/data/graph_5.csv"))
    graph.charger_graph(os.path.join(script_dir, "tests/data/graph_20.csv"))
    
    # Option 2: Générer des lieux aléatoirement
    #graph.generer_lieux_aleatoires(nb_lieux=20)
    
    # Calculer la matrice OD
    graph.calcul_matrice_cout_od()
    
    # Créer l'affichage (modifier avec votre groupe)
    affichage = Affichage(graph, titre_fenetre="Groupe 8 — Algorithme Génétique — TSP")
    
    # MODE COMPÉTITION : plusieurs runs pour trouver le meilleur résultat
    MODE_COMPETITION = True  # Mettre à False pour un seul run
    
    if MODE_COMPETITION:
        meilleure_route, meilleure_distance = executer_multiple_runs(
            graph,
            nb_runs=10,  # 10 runs pour maximiser les chances de trouver l'optimum
            affichage=affichage
        )
    else:
        # Mode normal (un seul run)
        ga = TSP_GA(
            graph,
            taille_population=300,
            taux_mutation=0.16,
            taux_crossover=0.93,
            taux_elitisme=0.13,
            nb_meilleures_affichees=5,
            seed=42
        )
        
        print("Démarrage de l'algorithme génétique...")
        print("Appuyez sur 'P' pour afficher/masquer les meilleures routes de la population")
        print("Appuyez sur 'R' pour afficher/masquer les meilleures routes")
        print("Appuyez sur 'ESC' pour quitter")
        
        meilleure_route, meilleure_distance = ga.executer(
            nb_iterations=1500,
            affichage=affichage,
            delai=0.02,
            early_stopping=120,
            log_csv="generations_log.csv"
        )
        
        print(f"\nRésultat final:")
        print(f"Meilleure distance: {meilleure_distance:.2f}")
        if meilleure_route is not None:
            print(f"Route: {meilleure_route.ordre}")
    
    # Démarrer l'interface graphique
    affichage.mainloop()


if __name__ == "__main__":
    main()
