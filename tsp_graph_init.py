
import numpy as np
import random
import time
import tkinter as tk
import csv

# Constantes d'affichage et de génération
LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 10


class Lieu:
    """
    represent a place to visit, with its coordinates (x, y) and a name.
    """

    def __init__(self, x, y, nom):
        self.x = float(x)
        self.y = float(y)
        self.nom = str(nom)

    def distance_to(self, autre_lieu):
        """
        Euclidean distance to another Lieu.
        """
        dx = self.x - autre_lieu.x
        dy = self.y - autre_lieu.y
        return float(np.hypot(dx, dy))


class Route:
    """
    Represent a route (cycle) on the graph: order of the places visited.
    Constraint: starts and ends with 0 (starting place).
    """

    def __init__(self, ordre, nombre_lieux=None):
        # Copie défensive
        self.ordre = list(ordre) if ordre is not None else []
        if nombre_lieux is not None and len(self.ordre) == 0:
            # Ex: route séquentielle 0..N-1..0
            self.ordre = list(range(nombre_lieux)) + [0]
        self._assurer_depart_retour_zero()

    def _assurer_depart_retour_zero(self):
        """
        Force the route to start and end with 0 if possible.
        """
        if len(self.ordre) == 0:
            self.ordre = [0]
        # If the first is not 0 but 0 is present, we make a rotation
        if self.ordre[0] != 0 and 0 in self.ordre:
            idx0 = self.ordre.index(0)
            self.ordre = self.ordre[idx0:] + self.ordre[1:idx0 + 1]
        # If 0 is absent, we add it to the beginning
        if self.ordre[0] != 0:
            self.ordre.insert(0, 0)
        # Ensure the closure
        if self.ordre[-1] != 0:
            self.ordre.append(0)

    def nombre_sommets_uniques(self):
        return len(set(self.ordre)) - 1 if len(self.ordre) > 1 else 0


class Graph:
    """
    Contains the places, the OD cost matrix and calculation utilities.
    """

    def __init__(self):
        self.liste_lieux = []
        self.matrice_od = None

    def generer_lieux_aleatoires(self, nb_lieux=NB_LIEUX, largeur=LARGEUR, hauteur=HAUTEUR, graine=None):
        """
        Generate randomly places in the bounds [0, largeur] x [0, hauteur].
        """
        if graine is not None:
            random.seed(graine)
        self.liste_lieux = []
        for i in range(nb_lieux):
            x = random.uniform(0.0, float(largeur))
            y = random.uniform(0.0, float(hauteur))
            self.liste_lieux.append(Lieu(x, y, f"Lieu {i}"))
        self.matrice_od = None

    def charger_graph(self, chemin_csv):
        """
        Load the list of places from a CSV.
        Expected columns (flexible): 'x','y','nom' (case insensitive).
        If 'nom' is absent, a name will be generated from the index.
        """
        # Try via pandas if available, otherwise csv from the standard
        try:
            import pandas as pd  # autorisé
            df = pd.read_csv(chemin_csv)
            colonnes = {c.lower(): c for c in df.columns}
            if 'x' not in colonnes or 'y' not in colonnes:
                raise ValueError("Le CSV doit contenir au minimum des colonnes 'x' et 'y'.")
            col_x = colonnes['x']
            col_y = colonnes['y']
            col_nom = colonnes.get('nom', None)
            self.liste_lieux = []
            for i, ligne in df.iterrows():
                nom = (str(ligne[col_nom]) if col_nom else f"Lieu {i}")
                self.liste_lieux.append(Lieu(ligne[col_x], ligne[col_y], nom))
        except Exception:
            # Fallback minimal with csv
            self.liste_lieux = []
            with open(chemin_csv, newline='', encoding='utf-8') as f:
                lecteur = csv.DictReader(f)
                champs = {c.lower(): c for c in lecteur.fieldnames} if lecteur.fieldnames else {}
                col_x = champs.get('x')
                col_y = champs.get('y')
                col_nom = champs.get('nom')
                if not col_x or not col_y:
                    raise ValueError("Le CSV doit contenir au minimum des colonnes 'x' et 'y'.")
                for i, ligne in enumerate(lecteur):
                    nom = ligne[col_nom] if col_nom and ligne.get(col_nom) else f"Lieu {i}"
                    self.liste_lieux.append(Lieu(ligne[col_x], ligne[col_y], nom))
        self.matrice_od = None

    # ------------ OD matrix and utilities ------------
    def calcul_matrice_cout_od(self):
        """
        Calculate the matrix of Euclidean distances between all places.
        Store the result in self.matrice_od (numpy.ndarray of shape NxN).
        """
        n = len(self.liste_lieux)
        if n == 0:
            self.matrice_od = np.zeros((0, 0), dtype=float)
            return self.matrice_od
        coords = np.array([[lieu.x, lieu.y] for lieu in self.liste_lieux], dtype=float)
        # Différences vectorisées: ||xi - xj||
        diff = coords[:, None, :] - coords[None, :, :]
        self.matrice_od = np.linalg.norm(diff, axis=2)
        return self.matrice_od

    def plus_proche_voisin(self, lieu_idx):
        """
        Return the index of the place closest to the place 'lieu_idx', via the distance matrix.
        """
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        n = self.matrice_od.shape[0]
        if n == 0:
            return None
        distances = np.array(self.matrice_od[lieu_idx], dtype=float)
        distances[lieu_idx] = np.inf  # Exclude itself
        return int(np.argmin(distances))

    def calcul_distance_route(self, route):
        """
        Calculate the total distance of the route (sum of successive edges).
        Utilise la matrice si disponible, sinon calcule à la volée.
        """
        ordre = route.ordre
        if len(ordre) < 2:
            return 0.0
        if self.matrice_od is None or (self.matrice_od.shape[0] != len(self.liste_lieux)):
            # Direct calculation via coordinates
            total = 0.0
            for i in range(len(ordre) - 1):
                a = self.liste_lieux[ordre[i]]
                b = self.liste_lieux[ordre[i + 1]]
                total += a.distance_to(b)
            return float(total)
        # Use the OD matrix
        indices_src = np.array(ordre[:-1], dtype=int)
        indices_dst = np.array(ordre[1:], dtype=int)
        return float(self.matrice_od[indices_src, indices_dst].sum())


class Affichage:
    """
    Tkinter display of places, the best route, the N best routes and/or a cost matrix.
    """

    COULEUR_LIEU = "#222222"
    COULEUR_TEXTE_LIEU = "#ffffff"
    COULEUR_SELEC = "#1e90ff"
    COULEUR_MEILLEURE_ROUTE = "blue"
    COULEUR_TOP_ROUTES = "#c0c0c0"
    RAYON_LIEU = 10

    def __init__(self, graph, titre_fenetre="SIG Spatial IA — Groupe à renseigner", n_top_routes=5):
        self.graph = graph
        self.n_top_routes = int(n_top_routes)
        self.top_routes = []  # liste de Route
        self.meilleure_route = None  # Route
        self.pheromones = None  # np.ndarray optionnel

        self.root = tk.Tk()
        self.root.title(titre_fenetre)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        self.canvas = tk.Canvas(self.root, width=LARGEUR, height=HAUTEUR, bg="white", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=False)
        self.zone_texte = tk.Text(self.root, height=8, width=100)
        self.zone_texte.pack(side="bottom", fill="x")

        # Raccourcis
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        # 'r' = basculer affichage N meilleures routes
        self.root.bind("r", self._on_toggle_routes)
        # 'p' = basculer affichage matrice (texte) des coûts / phéromones
        self.root.bind("p", self._on_toggle_pheromones)
        self.afficher_routes = True
        self.afficher_pheromones = False

        self._dessiner_graph()
        self._redessiner_routes()

    # ------------ Update of the display data ------------
    def set_meilleure_route(self, route):
        self.meilleure_route = route
        self._redessiner_routes()

    def set_top_routes(self, routes):
        self.top_routes = list(routes)[: self.n_top_routes]
        self._redessiner_routes()

    def set_pheromones(self, matrice):
        self.pheromones = matrice
        self._mettre_a_jour_zone_texte_pheromones()

    def set_message(self, texte):
        self.zone_texte.delete("1.0", tk.END)
        self.zone_texte.insert(tk.END, texte)

    # ------------ Event management ------------
    def _on_toggle_routes(self, _evt=None):
        self.afficher_routes = not self.afficher_routes
        self._redessiner_routes()

    def _on_toggle_pheromones(self, _evt=None):
        self.afficher_pheromones = not self.afficher_pheromones
        self._mettre_a_jour_zone_texte_pheromones()

    # ------------ Drawing ------------
    def _dessiner_graph(self):
        self.canvas.delete("all")
        for idx, lieu in enumerate(self.graph.liste_lieux):
            cx = lieu.x
            cy = lieu.y
            r = self.RAYON_LIEU
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=self.COULEUR_LIEU, outline="")
            self.canvas.create_text(cx, cy, text=str(idx), fill=self.COULEUR_TEXTE_LIEU, font=("Arial", 10, "bold"))
        self.canvas.update()

    def _redessiner_routes(self):
        # Clear all lines
        self.canvas.delete("route")
        self.canvas.delete("ordre")
        if self.afficher_routes:
            # Draw N best routes in light gray
            for route in self.top_routes:
                self._dessiner_route(route, couleur=self.COULEUR_TOP_ROUTES, dash=(2, 4), tag="route")
        # Draw the best route in blue dashed
        if self.meilleure_route is not None:
            self._dessiner_route(self.meilleure_route, couleur=self.COULEUR_MEILLEURE_ROUTE, dash=(6, 4), tag="route")
            self._dessiner_ordre_visite(self.meilleure_route)
        self.canvas.update()

    def _dessiner_route(self, route, couleur, dash, tag):
        ordre = route.ordre
        if len(ordre) < 2:
            return
        points = []
        for idx in ordre:
            lieu = self.graph.liste_lieux[idx]
            points.extend([lieu.x, lieu.y])
        self.canvas.create_line(*points, fill=couleur, dash=dash, width=2, tags=tag, smooth=False)

    def _dessiner_ordre_visite(self, route):
        """
        Display the visit order (0..N-1..) above each visited place for the best route.
        """
        # Clean up old order labels
        self.canvas.delete("ordre")
        # Map place -> visit position (except last 0)
        for position, idx_lieu in enumerate(route.ordre[:-1]):
            lieu = self.graph.liste_lieux[idx_lieu]
            self.canvas.create_text(lieu.x, lieu.y - (self.RAYON_LIEU + 10), text=str(position),
                                    fill="black", font=("Arial", 9), tags="ordre")

    def _mettre_a_jour_zone_texte_pheromones(self):
        """
        Display the cost/pheromones matrix in text form in the text zone.
        """
        if not self.afficher_pheromones:
            return
        self.zone_texte.delete("1.0", tk.END)
        if self.pheromones is None:
            # By default, we display the OD matrix if available
            matrice = self.graph.matrice_od
            if matrice is None:
                self.zone_texte.insert(tk.END, "Aucune matrice à afficher (OD/Phéromones non disponible).")
                return
            titre = "Matrice de coûts (OD):\n"
            self.zone_texte.insert(tk.END, titre)
            self._inserer_matrice_texte(matrice)
        else:
            titre = "Matrice de phéromones:\n"
            self.zone_texte.insert(tk.END, titre)
            self._inserer_matrice_texte(self.pheromones)

    def _inserer_matrice_texte(self, matrice):
        """
        Format a matrix (numpy) in aligned text.
        """
        arr = np.array(matrice, dtype=float)
        with np.printoptions(precision=2, suppress=True):
            texte = str(arr)
        self.zone_texte.insert(tk.END, texte)

    def mainloop(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Simple demonstration: random generation, sequential route and display
    g = Graph()
    g.generer_lieux_aleatoires(nb_lieux=NB_LIEUX, largeur=LARGEUR, hauteur=HAUTEUR, graine=42)
    g.calcul_matrice_cout_od()
    route_demo = Route(ordre=None, nombre_lieux=len(g.liste_lieux))

    ui = Affichage(g, titre_fenetre="SIG Spatial IA — Groupe DEMO", n_top_routes=5)
    ui.set_meilleure_route(route_demo)
    # Example of N best routes (simple variants)
    variantes = []
    if len(g.liste_lieux) >= 5:
        variantes.append(Route([0, 2, 1, 3, 4] + list(range(5, len(g.liste_lieux))) + [0]))
        variantes.append(Route([0, 1, 3, 2, 4] + list(range(5, len(g.liste_lieux))) + [0]))
    ui.set_top_routes(variantes)
    msg = f"Itération: 0 | Meilleure distance: {g.calcul_distance_route(route_demo):.2f}\n"
    msg += "Raccourcis: 'r' = top routes, 'p' = matrice OD/pheromones, 'ESC' = quitter"
    ui.set_message(msg)
    ui.mainloop()

