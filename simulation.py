import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp
from scipy.spatial.distance import cdist
import seaborn as sns
from matplotlib.colors import ListedColormap
import random

class DroneBaseOptimization:
    
    def __init__(self, grid_size=(100, 100), num_candidate_bases=20, max_bases=5, 
                 max_distance=30, gamma=0.01):
        self.grid_size = grid_size
        self.num_candidate_bases = num_candidate_bases
        self.max_bases = max_bases
        self.max_distance = max_distance
        self.gamma = gamma
        
        # Create grid cells
        self.grid_cells = [(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])]
        self.num_cells = len(self.grid_cells)
        
        # Initialize probabilities of search need
        self.search_probs = np.zeros(self.num_cells)
        
        # Generate candidate base locations
        self.candidate_bases = self._generate_candidate_bases()
        
        # Calculate distances between cells and bases
        self.distances = self._calculate_distances()
        
        # Store results
        self.selected_bases = []
        self.cell_coverage = np.zeros(self.num_cells, dtype=bool)
        self.cell_closest_base = np.full(self.num_cells, -1)
    
    def _generate_candidate_bases(self):
        """Generate candidate base locations randomly"""
        candidate_bases = []
        while len(candidate_bases) < self.num_candidate_bases:
            x = random.randint(0, self.grid_size[0] - 1)
            y = random.randint(0, self.grid_size[1] - 1)
            if (x, y) not in candidate_bases:
                candidate_bases.append((x, y))
        return candidate_bases
    
    def _calculate_distances(self):
        """Calculate distances between all cells and all candidate bases"""
        cell_coords = np.array(self.grid_cells)
        base_coords = np.array(self.candidate_bases)
        return cdist(cell_coords, base_coords, 'euclidean')
    
    def generate_search_probabilities(self, elevation=None, population=None, 
                                      flood_areas=None, rescue_locations=None):
        # Initialize with base probability
        self.search_probs = np.random.uniform(0, 0.05, self.num_cells)
        
        # If elevation data provided
        if elevation is not None:
            # Normalize elevation to 0-1 range
            elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
            # Lower elevations have higher probability
            elev_factor = 1 - elev_norm
            # Increase probability for low elevation cells
            self.search_probs += 0.15 * (elev_factor > 0.8)
        
        # If population data provided
        if population is not None:
            # Normalize population to 0-1 range
            pop_norm = (population - population.min()) / (population.max() - population.min())
            # Add population factor
            self.search_probs += 0.25 * pop_norm
        
        # If flood areas provided
        if flood_areas is not None:
            flood_threshold = 3  # 3 km threshold
            for i, cell in enumerate(self.grid_cells):
                for flood_loc in flood_areas:
                    dist = np.sqrt((cell[0] - flood_loc[0])**2 + (cell[1] - flood_loc[1])**2)
                    if dist < flood_threshold:
                        self.search_probs[i] += 0.25
                        break
        
        # If rescue locations provided
        if rescue_locations is not None:
            rescue_threshold = 5  # 5 km threshold
            for i, cell in enumerate(self.grid_cells):
                for rescue_loc in rescue_locations:
                    dist = np.sqrt((cell[0] - rescue_loc[0])**2 + (cell[1] - rescue_loc[1])**2)
                    if dist < rescue_threshold:
                        self.search_probs[i] += 0.3
                        break
        
        # Ensure probabilities are within [0, 1]
        self.search_probs = np.clip(self.search_probs, 0, 1)
        
        return self.search_probs
    
    def optimize_base_locations(self):
        # Create the optimization model
        model = pulp.LpProblem("DroneBaseLocation", pulp.LpMaximize)
        
        # Create decision variables
        # x_j: binary variable indicating if base j is opened
        x = pulp.LpVariable.dicts("Base", range(self.num_candidate_bases), cat=pulp.LpBinary)
        
        # z_i: binary variable indicating if cell i is covered
        z = pulp.LpVariable.dicts("CellCovered", range(self.num_cells), cat=pulp.LpBinary)
        
        # y_ij: binary variable indicating if cell i is covered and has closest base j
        y = {(i, j): pulp.LpVariable(f"Closest_{i}_{j}", cat=pulp.LpBinary)
             for i in range(self.num_cells) for j in range(self.num_candidate_bases)
             if self.distances[i, j] <= self.max_distance}
        
        # Set the objective function
        # First term: maximize coverage weighted by search probabilities
        # Second term: minimize distance to nearest base
        objective_terms = []
        
        # Coverage term
        for i in range(self.num_cells):
            objective_terms.append(self.search_probs[i] * z[i])
        
        # Distance term
        for i in range(self.num_cells):
            for j in range(self.num_candidate_bases):
                if self.distances[i, j] <= self.max_distance:
                    objective_terms.append(-self.gamma * self.search_probs[i] * 
                                           self.distances[i, j] * y[(i, j)])
        
        model += pulp.lpSum(objective_terms)
        
        # Maximum number of bases constraint
        model += pulp.lpSum([x[j] for j in range(self.num_candidate_bases)]) <= self.max_bases
        
        # Cell coverage constraints
        for i in range(self.num_cells):
            eligible_bases = [j for j in range(self.num_candidate_bases) 
                             if self.distances[i, j] <= self.max_distance]
            
            # A cell is covered if at least one eligible base is open
            model += z[i] <= pulp.lpSum([x[j] for j in eligible_bases])
            
            # A cell is covered if it has at least one closest base
            model += z[i] <= pulp.lpSum([y[(i, j)] for j in eligible_bases 
                                        if (i, j) in y])
            
            # Cell can have at most one closest base
            model += pulp.lpSum([y[(i, j)] for j in eligible_bases 
                                if (i, j) in y]) <= 1
            
            # A cell can only be served by an open base
            for j in eligible_bases:
                if (i, j) in y:
                    model += y[(i, j)] <= x[j]
        
        # Solve the model
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract results
        self.selected_bases = [j for j in range(self.num_candidate_bases) if x[j].value() > 0.5]
        self.cell_coverage = np.array([z[i].value() > 0.5 for i in range(self.num_cells)])
        
        # Determine closest base for each cell
        self.cell_closest_base = np.full(self.num_cells, -1)
        for i in range(self.num_cells):
            for j in range(self.num_candidate_bases):
                if (i, j) in y and y[(i, j)].value() > 0.5:
                    self.cell_closest_base[i] = j
                    break
        
        # Calculate metrics
        coverage_pct = np.mean(self.cell_coverage) * 100
        weighted_coverage_pct = np.sum(self.search_probs * self.cell_coverage) / np.sum(self.search_probs) * 100
        
        # Calculate mean distance to nearest base for covered cells
        covered_indices = np.where(self.cell_coverage)[0]
        mean_dnb_covered = np.mean([self.distances[i, self.cell_closest_base[i]] 
                                   for i in covered_indices]) if len(covered_indices) > 0 else 0
        
        print(f"Coverage: {coverage_pct:.1f}%")
        print(f"Weighted Coverage: {weighted_coverage_pct:.1f}%")
        print(f"Mean Distance to Nearest Base (covered cells): {mean_dnb_covered:.1f} km")
        
        return self.selected_bases, self.cell_coverage, self.cell_closest_base
    
    def relocate_bases(self, updated_search_probs=None, max_relocations=2, relocation_radius=30):
        if updated_search_probs is not None:
            self.search_probs = updated_search_probs
        
        if not self.selected_bases:
            raise ValueError("Must run optimize_base_locations first")
            
        # Create the optimization model for relocation
        model = pulp.LpProblem("DroneBaseRelocation", pulp.LpMaximize)
        
        # Set of existing base locations (J1 in the paper)
        J1 = self.selected_bases
        
        # Set of potential new base locations (J2 in the paper)
        J2 = [j for j in range(self.num_candidate_bases) if j not in J1]
        
        # For each existing base, find eligible relocation sites within relocation_radius
        relocation_options = {}
        for j in J1:
            base_j_coords = self.candidate_bases[j]
            relocation_options[j] = []
            for k in J2:
                base_k_coords = self.candidate_bases[k]
                distance = np.sqrt((base_j_coords[0] - base_k_coords[0])**2 + 
                                   (base_j_coords[1] - base_k_coords[1])**2)
                if distance <= relocation_radius:
                    relocation_options[j].append(k)
        
        # Create decision variables
        # x_j: 1 if base j remains open (for j in J1) or is newly opened (for j in J2)
        x = pulp.LpVariable.dicts("Base", range(self.num_candidate_bases), cat=pulp.LpBinary)
        
        # z_i: 1 if cell i is covered
        z = pulp.LpVariable.dicts("CellCovered", range(self.num_cells), cat=pulp.LpBinary)
        
        # y_ij: 1 if cell i is covered and has closest base j
        y = {(i, j): pulp.LpVariable(f"Closest_{i}_{j}", cat=pulp.LpBinary)
             for i in range(self.num_cells) for j in range(self.num_candidate_bases)
             if self.distances[i, j] <= self.max_distance}
        
        # m_jk: 1 if base j is closed and base k is opened ("relocation" from j to k)
        m = {(j, k): pulp.LpVariable(f"Move_{j}_{k}", cat=pulp.LpBinary)
             for j in J1 for k in relocation_options[j]}
        
        # v_jk: binary variable for big-M constraints
        v = {(j, k): pulp.LpVariable(f"V_{j}_{k}", cat=pulp.LpBinary)
             for j in J1 for k in relocation_options[j]}
        
        # Set the objective function (same as baseline)
        objective_terms = []
        
        # Coverage term
        for i in range(self.num_cells):
            objective_terms.append(self.search_probs[i] * z[i])
        
        # Distance term
        for i in range(self.num_cells):
            for j in range(self.num_candidate_bases):
                if self.distances[i, j] <= self.max_distance:
                    objective_terms.append(-self.gamma * self.search_probs[i] * 
                                           self.distances[i, j] * y[(i, j)])
        
        model += pulp.lpSum(objective_terms)
        
        # Maximum number of bases constraint
        model += pulp.lpSum([x[j] for j in range(self.num_candidate_bases)]) <= self.max_bases
        
        # Cell coverage constraints (same as baseline)
        for i in range(self.num_cells):
            eligible_bases = [j for j in range(self.num_candidate_bases) 
                             if self.distances[i, j] <= self.max_distance]
            
            # A cell is covered if at least one eligible base is open
            model += z[i] <= pulp.lpSum([x[j] for j in eligible_bases])
            
            # A cell is covered if it has at least one closest base
            model += z[i] <= pulp.lpSum([y[(i, j)] for j in eligible_bases 
                                        if (i, j) in y])
            
            # Cell can have at most one closest base
            model += pulp.lpSum([y[(i, j)] for j in eligible_bases 
                                if (i, j) in y]) <= 1
            
            # A cell can only be served by an open base
            for j in eligible_bases:
                if (i, j) in y:
                    model += y[(i, j)] <= x[j]
        
        # Relocation constraints
        big_M = 1000
        
        # If m_jk = 1, then v_jk = 1
        for j in J1:
            for k in relocation_options[j]:
                model += m[(j, k)] <= big_M * v[(j, k)]
        
        # If base j is closed for relocation, then x_j = 0
        for j in J1:
            model += x[j] <= big_M * (1 - pulp.lpSum([v[(j, k)] for k in relocation_options[j]]))
        
        # If base k is opened due to relocation, then the corresponding m_jk = 1
        for j in J1:
            for k in relocation_options[j]:
                model += x[k] <= m[(j, k)]
        
        # A new base k can only be opened once
        for k in J2:
            relevent_js = [j for j in J1 if k in relocation_options[j]]
            if relevent_js:
                model += pulp.lpSum([m[(j, k)] for j in relevent_js]) <= 1
        
        # An existing base j can only be relocated once
        for j in J1:
            if relocation_options[j]:
                model += pulp.lpSum([m[(j, k)] for k in relocation_options[j]]) <= 1
        
        # Maximum number of relocations
        model += pulp.lpSum([m[(j, k)] for j in J1 for k in relocation_options[j]]) <= max_relocations
        
        # Solve the model
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract results
        new_selected_bases = [j for j in range(self.num_candidate_bases) if x[j].value() > 0.5]
        cell_coverage = np.array([z[i].value() > 0.5 for i in range(self.num_cells)])
        
        # Determine closest base for each cell
        cell_closest_base = np.full(self.num_cells, -1)
        for i in range(self.num_cells):
            for j in range(self.num_candidate_bases):
                if (i, j) in y and y[(i, j)].value() > 0.5:
                    cell_closest_base[i] = j
                    break
        
        # Determine which bases were relocated
        relocations = []
        for j in J1:
            for k in relocation_options[j]:
                if (j, k) in m and m[(j, k)].value() > 0.5:
                    relocations.append((j, k))
        
        if relocations:
            print(f"Relocated {len(relocations)} bases:")
            for j, k in relocations:
                j_coords = self.candidate_bases[j]
                k_coords = self.candidate_bases[k]
                print(f"  Base at {j_coords} relocated to {k_coords}")
        else:
            print("No bases were relocated")
        
        # Calculate metrics
        coverage_pct = np.mean(cell_coverage) * 100
        weighted_coverage_pct = np.sum(self.search_probs * cell_coverage) / np.sum(self.search_probs) * 100
        
        # Calculate mean distance to nearest base for covered cells
        covered_indices = np.where(cell_coverage)[0]
        mean_dnb_covered = np.mean([self.distances[i, cell_closest_base[i]] 
                                   for i in covered_indices]) if len(covered_indices) > 0 else 0
        
        print(f"New Coverage: {coverage_pct:.1f}%")
        print(f"New Weighted Coverage: {weighted_coverage_pct:.1f}%")
        print(f"New Mean Distance to Nearest Base (covered cells): {mean_dnb_covered:.1f} km")
        
        # Update the object's attributes
        self.selected_bases = new_selected_bases
        self.cell_coverage = cell_coverage
        self.cell_closest_base = cell_closest_base
        
        return new_selected_bases, cell_coverage, cell_closest_base
    
    def calculate_gini_coefficient(self):
        # Calculate distances to nearest base
        dnb_all = np.zeros(self.num_cells)
        for i in range(self.num_cells):
            if self.cell_closest_base[i] >= 0:
                dnb_all[i] = self.distances[i, self.cell_closest_base[i]]
            else:
                # For uncovered cells, use a large value (e.g., 2 * max_distance)
                dnb_all[i] = 2 * self.max_distance
        
        # Sort distances in ascending order
        dnb_all_sorted = np.sort(dnb_all)
        
        # Get distances for covered cells only
        covered_indices = np.where(self.cell_coverage)[0]
        dnb_covered = dnb_all[covered_indices]
        dnb_covered_sorted = np.sort(dnb_covered)
        
        # Calculate Gini coefficient for all cells
        n_all = len(dnb_all)
        gini_all = np.sum((2 * np.arange(1, n_all + 1) - n_all - 1) * dnb_all_sorted) / (n_all * np.sum(dnb_all_sorted))
        
        # Calculate Gini coefficient for covered cells
        if len(dnb_covered) > 0:
            n_covered = len(dnb_covered)
            gini_covered = np.sum((2 * np.arange(1, n_covered + 1) - n_covered - 1) * dnb_covered_sorted) / (n_covered * np.sum(dnb_covered_sorted))
        else:
            gini_covered = 0
        
        print(f"Gini Coefficient (covered cells): {gini_covered:.3f}")
        print(f"Gini Coefficient (all cells): {gini_all:.3f}")
        
        return gini_covered, gini_all
    
    def plot_results(self, ax=None, show_search_probs=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        
        # Convert grid cells to 2D array for visualization
        grid_2d = np.zeros(self.grid_size)
        
        if show_search_probs:
            # Plot search probabilities
            for idx, (x, y) in enumerate(self.grid_cells):
                grid_2d[x, y] = self.search_probs[idx]
            
            im = ax.imshow(grid_2d.T, origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax, label='Probability of Search Need')
            ax.set_title('Search Need Probabilities')
        else:
            # Color cells based on closest base
            unique_bases = np.unique(self.cell_closest_base)
            unique_bases = unique_bases[unique_bases >= 0]
            
            # Create a colormap with unique colors for each base
            cmap = plt.cm.get_cmap('tab10', len(unique_bases))
            
            # Create a masked array for uncovered cells
            grid_2d = np.full(self.grid_size, -1)
            
            for idx, (x, y) in enumerate(self.grid_cells):
                if self.cell_coverage[idx]:
                    # Get the sequential index of the base in unique_bases
                    base_idx = np.where(unique_bases == self.cell_closest_base[idx])[0][0]
                    grid_2d[x, y] = base_idx
            
            # Create a custom colormap with grey for uncovered cells
            colors = [cmap(i) for i in range(len(unique_bases))]
            custom_cmap = ListedColormap(['lightgrey'] + colors)
            
            # Plot the grid
            im = ax.imshow(grid_2d.T, origin='lower', cmap=custom_cmap, 
                          vmin=-1, vmax=len(unique_bases)-1)
            ax.set_title('Coverage Map (Each color represents a different base)')
        
        # Plot base locations
        for j in self.selected_bases:
            x, y = self.candidate_bases[j]
            ax.plot(x, y, 'X', color='white', markersize=10, markeredgecolor='black')
        
        # Plot candidate base locations
        for j in range(self.num_candidate_bases):
            if j not in self.selected_bases:
                x, y = self.candidate_bases[j]
                ax.plot(x, y, 's', color='white', markersize=6, markeredgecolor='black', alpha=0.6)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        
        coverage_pct = np.mean(self.cell_coverage) * 100
        
        # Calculate mean distance to nearest base for covered cells
        covered_indices = np.where(self.cell_coverage)[0]
        if len(covered_indices) > 0:
            mean_dnb = np.mean([self.distances[i, self.cell_closest_base[i]] 
                              for i in covered_indices])
        else:
            mean_dnb = 0
        
        ax.set_title(f'Coverage: {coverage_pct:.1f}%, Mean DNB: {mean_dnb:.1f} km')
        
        return fig
    
    def plot_distance_vs_probability(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        # Calculate distances to nearest base for all cells
        distances_to_nearest = np.zeros(self.num_cells)
        for i in range(self.num_cells):
            if self.cell_closest_base[i] >= 0:
                distances_to_nearest[i] = self.distances[i, self.cell_closest_base[i]]
            else:
                # For uncovered cells, use a value beyond max_distance
                distances_to_nearest[i] = 1.5 * self.max_distance
        
        # Plot distances vs probabilities
        ax.scatter(self.search_probs, distances_to_nearest, 
                   c=self.cell_coverage, cmap='coolwarm', alpha=0.7)
        
        # Draw a horizontal line at max_distance
        ax.axhline(y=self.max_distance, color='r', linestyle='--')
        
        ax.set_xlabel('Probability of Search Need')
        ax.set_ylabel('Distance to Nearest Base (km)')
        ax.set_title('Distance to Nearest Base vs. Probability of Search Need')
        
        # Add a color bar
        sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Covered')
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Not Covered', 'Covered'])
        
        return fig


# Example usage
def demonstration():
    """Demonstrate the use of the DroneBaseOptimization class"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create a grid and optimization object
    grid_size = (50, 50)  # 50x50 km grid
    optimizer = DroneBaseOptimization(
        grid_size=grid_size,
        num_candidate_bases=15,
        max_bases=4,
        max_distance=15,
        gamma=0.01
    )
    
    # Generate synthetic data for search probabilities
    # 1. Elevation data (lower is more flood-prone)
    elevation = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Create a valley in the middle with lower elevation
            elevation[i, j] = 100 + 10 * ((i - grid_size[0]//2)**2 + (j - grid_size[1]//2)**2) ** 0.5
    elevation = elevation.flatten()
    
    # 2. Population data (higher in certain areas)
    population = np.zeros(grid_size)
    # Add a few population centers
    centers = [(10, 10), (30, 40), (40, 15), (20, 30)]
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Sum of proximity to population centers
            population[i, j] = sum(1000 / (1 + ((i-x)**2 + (j-y)**2) ** 0.5) for x, y in centers)
    population = population.flatten()
    
    # 3. Flood-prone areas
    flood_areas = [(15, 5), (25, 15), (35, 25), (45, 35)]
    
    # 4. Previous rescue locations
    rescue_locations = [(12, 8), (22, 18), (32, 28), (42, 38), (15, 25), (25, 35)]
    
    # Generate search probabilities
    search_probs = optimizer.generate_search_probabilities(
        elevation=elevation,
        population=population,
        flood_areas=flood_areas,
        rescue_locations=rescue_locations
    )
    
    # Create a figure to display the search probabilities
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot search probabilities
    optimizer.plot_results(ax=axs[0, 0], show_search_probs=True)
    
    # Optimize base locations
    selected_bases, cell_coverage, cell_closest_base = optimizer.optimize_base_locations()
    
    # Plot the initial optimization results
    optimizer.plot_results(ax=axs[0, 1])
    
    # Plot distance vs probability
    optimizer.plot_distance_vs_probability(ax=axs[1, 0])
    
    # Calculate Gini coefficient
    optimizer.calculate_gini_coefficient()
    
    # Create updated search probabilities (simulating changed conditions)
    updated_search_probs = optimizer.search_probs.copy()
    
    # Shift the probability center to simulate changing disaster conditions
    prob_grid = updated_search_probs.reshape(grid_size)
    
    # Apply a shift/transform to the probabilities
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Shift focus to different areas
            prob_grid[i, j] *= 0.5 + 0.5 * np.sin(i/10) * np.cos(j/10)
    
    updated_search_probs = prob_grid.flatten()
    updated_search_probs = np.clip(updated_search_probs, 0, 1)
    
    # Relocate bases based on updated search probabilities
    optimizer.relocate_bases(
        updated_search_probs=updated_search_probs,
        max_relocations=2,
        relocation_radius=20
    )
    
    # Plot the results after relocation
    optimizer.plot_results(ax=axs[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return optimizer

# Function that implements the Planning for Potential Relocation (PPR) approach
def optimize_with_ppr(grid_size=(50, 50), max_bases=4, max_distance=15, 
                      gamma=0.01, lambda_param=0.5):
    # Create a PPR optimization model
    class PPROptimization(DroneBaseOptimization):
        def __init__(self, grid_size, num_candidate_bases, max_bases, 
                     max_distance, gamma, lambda_param):
            super().__init__(grid_size, num_candidate_bases, max_bases, 
                            max_distance, gamma)
            self.lambda_param = lambda_param
            
        def optimize_base_locations_ppr(self, relocation_radius=30):
            # Create the optimization model
            model = pulp.LpProblem("DroneBaseLocationPPR", pulp.LpMaximize)
            
            # Create decision variables
            # x_j: binary variable indicating if base j is opened
            x = pulp.LpVariable.dicts("Base", range(self.num_candidate_bases), cat=pulp.LpBinary)
            
            # z_i: binary variable indicating if cell i is covered
            z = pulp.LpVariable.dicts("CellCovered", range(self.num_cells), cat=pulp.LpBinary)
            
            # y_ij: binary variable indicating if cell i is covered and has closest base j
            y = {(i, j): pulp.LpVariable(f"Closest_{i}_{j}", cat=pulp.LpBinary)
                 for i in range(self.num_cells) for j in range(self.num_candidate_bases)
                 if self.distances[i, j] <= self.max_distance}
            
            # w_k: binary variable indicating if base k could be relocated to
            w = pulp.LpVariable.dicts("PotentialBase", range(self.num_candidate_bases), cat=pulp.LpBinary)
            
            # u_i: binary variable indicating if cell i is covered by potential relocation
            u = pulp.LpVariable.dicts("PotentialCovered", range(self.num_cells), cat=pulp.LpBinary)
            
            # Set the objective function
            # First term: maximize coverage weighted by search probabilities
            # Second term: minimize distance to nearest base
            # Third term: reward potential future coverage after relocation
            objective_terms = []
            
            # Current coverage term
            for i in range(self.num_cells):
                objective_terms.append(self.search_probs[i] * z[i])
            
            # Distance term
            for i in range(self.num_cells):
                for j in range(self.num_candidate_bases):
                    if self.distances[i, j] <= self.max_distance:
                        objective_terms.append(-self.gamma * self.search_probs[i] * 
                                               self.distances[i, j] * y[(i, j)])
            
            # Potential future coverage term
            for i in range(self.num_cells):
                objective_terms.append(self.lambda_param * self.search_probs[i] * u[i])
            
            model += pulp.lpSum(objective_terms)
            
            # Maximum number of bases constraint
            model += pulp.lpSum([x[j] for j in range(self.num_candidate_bases)]) <= self.max_bases
            
            # Cell coverage constraints
            for i in range(self.num_cells):
                eligible_bases = [j for j in range(self.num_candidate_bases) 
                                 if self.distances[i, j] <= self.max_distance]
                
                # A cell is covered if at least one eligible base is open
                model += z[i] <= pulp.lpSum([x[j] for j in eligible_bases])
                
                # A cell is covered if it has at least one closest base
                model += z[i] <= pulp.lpSum([y[(i, j)] for j in eligible_bases 
                                            if (i, j) in y])
                
                # Cell can have at most one closest base
                model += pulp.lpSum([y[(i, j)] for j in eligible_bases 
                                    if (i, j) in y]) <= 1
                
                # A cell can only be served by an open base
                for j in eligible_bases:
                    if (i, j) in y:
                        model += y[(i, j)] <= x[j]
            
            # Potential relocation constraints
            
            # A base k could be relocated to if there's an open base within relocation_radius
            for k in range(self.num_candidate_bases):
                # Find existing bases that could relocate to k
                relocating_bases = []
                for j in range(self.num_candidate_bases):
                    base_j_coords = self.candidate_bases[j]
                    base_k_coords = self.candidate_bases[k]
                    distance = np.sqrt((base_j_coords[0] - base_k_coords[0])**2 + 
                                       (base_j_coords[1] - base_k_coords[1])**2)
                    if distance <= relocation_radius:
                        relocating_bases.append(j)
                
                # w_k is 1 if at least one existing base could relocate to k
                model += w[k] <= pulp.lpSum([x[j] for j in relocating_bases])
            
            # A cell i is potentially covered if there's a potential base within range
            for i in range(self.num_cells):
                potential_bases = [k for k in range(self.num_candidate_bases) 
                                 if self.distances[i, k] <= self.max_distance]
                
                model += u[i] <= pulp.lpSum([w[k] for k in potential_bases])
            
            # A base can't be both open and a potential relocation site
            for j in range(self.num_candidate_bases):
                model += w[j] <= 1 - x[j]
            
            # Solve the model
            model.solve(pulp.PULP_CBC_CMD(msg=False))
            
            # Extract results
            self.selected_bases = [j for j in range(self.num_candidate_bases) if x[j].value() > 0.5]
            self.cell_coverage = np.array([z[i].value() > 0.5 for i in range(self.num_cells)])
            cell_potential_coverage = np.array([u[i].value() > 0.5 for i in range(self.num_cells)])
            
            # Determine closest base for each cell
            self.cell_closest_base = np.full(self.num_cells, -1)
            for i in range(self.num_cells):
                for j in range(self.num_candidate_bases):
                    if (i, j) in y and y[(i, j)].value() > 0.5:
                        self.cell_closest_base[i] = j
                        break
            
            # Calculate metrics
            coverage_pct = np.mean(self.cell_coverage) * 100
            weighted_coverage_pct = np.sum(self.search_probs * self.cell_coverage) / np.sum(self.search_probs) * 100
            potential_coverage_pct = np.mean(cell_potential_coverage) * 100
            
            # Calculate mean distance to nearest base for covered cells
            covered_indices = np.where(self.cell_coverage)[0]
            mean_dnb_covered = np.mean([self.distances[i, self.cell_closest_base[i]] 
                                       for i in covered_indices]) if len(covered_indices) > 0 else 0
            
            print(f"PPR Coverage: {coverage_pct:.1f}%")
            print(f"PPR Weighted Coverage: {weighted_coverage_pct:.1f}%")
            print(f"PPR Potential Additional Coverage: {potential_coverage_pct:.1f}%")
            print(f"PPR Mean Distance to Nearest Base (covered cells): {mean_dnb_covered:.1f} km")
            
            return self.selected_bases, self.cell_coverage, cell_potential_coverage
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create a PPR optimization object
    optimizer = PPROptimization(
        grid_size=grid_size,
        num_candidate_bases=15,
        max_bases=max_bases,
        max_distance=max_distance,
        gamma=gamma,
        lambda_param=lambda_param
    )
    
    # Generate synthetic data for search probabilities (same as in demonstration)
    # 1. Elevation data (lower is more flood-prone)
    elevation = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Create a valley in the middle with lower elevation
            elevation[i, j] = 100 + 10 * ((i - grid_size[0]//2)**2 + (j - grid_size[1]//2)**2) ** 0.5
    elevation = elevation.flatten()
    
    # 2. Population data (higher in certain areas)
    population = np.zeros(grid_size)
    # Add a few population centers
    centers = [(10, 10), (30, 40), (40, 15), (20, 30)]
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Sum of proximity to population centers
            population[i, j] = sum(1000 / (1 + ((i-x)**2 + (j-y)**2) ** 0.5) for x, y in centers)
    population = population.flatten()
    
    # 3. Flood-prone areas
    flood_areas = [(15, 5), (25, 15), (35, 25), (45, 35)]
    
    # 4. Previous rescue locations
    rescue_locations = [(12, 8), (22, 18), (32, 28), (42, 38), (15, 25), (25, 35)]
    
    # Generate search probabilities
    search_probs = optimizer.generate_search_probabilities(
        elevation=elevation,
        population=population,
        flood_areas=flood_areas,
        rescue_locations=rescue_locations
    )
    
    # Optimize base locations using PPR approach
    selected_bases, cell_coverage, cell_potential_coverage = optimizer.optimize_base_locations_ppr()
    
    return optimizer

# Comparison between baseline and PPR approaches
def compare_approaches():
    """Compare baseline approach with PPR approach"""
    
    # Set parameters
    grid_size = (50, 50)
    max_bases = 4
    max_distance = 15
    gamma = 0.01
    lambda_param = 0.5
    
    # Create baseline optimizer
    baseline_optimizer = DroneBaseOptimization(
        grid_size=grid_size,
        num_candidate_bases=15,
        max_bases=max_bases,
        max_distance=max_distance,
        gamma=gamma
    )
    
    # Generate synthetic data
    elevation = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            elevation[i, j] = 100 + 10 * ((i - grid_size[0]//2)**2 + (j - grid_size[1]//2)**2) ** 0.5
    elevation = elevation.flatten()
    
    population = np.zeros(grid_size)
    centers = [(10, 10), (30, 40), (40, 15), (20, 30)]
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            population[i, j] = sum(1000 / (1 + ((i-x)**2 + (j-y)**2) ** 0.5) for x, y in centers)
    population = population.flatten()
    
    flood_areas = [(15, 5), (25, 15), (35, 25), (45, 35)]
    rescue_locations = [(12, 8), (22, 18), (32, 28), (42, 38), (15, 25), (25, 35)]
    
    # Use the same search probabilities for both approaches
    search_probs = baseline_optimizer.generate_search_probabilities(
        elevation=elevation,
        population=population,
        flood_areas=flood_areas,
        rescue_locations=rescue_locations
    )
    
    # Create PPR optimizer with the same parameters and data
    ppr_optimizer = optimize_with_ppr(
        grid_size=grid_size,
        max_bases=max_bases,
        max_distance=max_distance,
        gamma=gamma,
        lambda_param=lambda_param
    )
    
    # Create updated search probabilities (simulating changed conditions)
    updated_search_probs = search_probs.copy()
    
    # Shift the probability center to simulate changing disaster conditions
    prob_grid = updated_search_probs.reshape(grid_size)
    
    # Apply a shift/transform to the probabilities
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Shift focus to different areas
            prob_grid[i, j] *= 0.5 + 0.5 * np.sin(i/10) * np.cos(j/10)
    
    updated_search_probs = prob_grid.flatten()
    updated_search_probs = np.clip(updated_search_probs, 0, 1)
    
    # Run optimization for baseline approach
    print("BASELINE APPROACH:")
    baseline_optimizer.optimize_base_locations()
    
    # Evaluate baseline metrics
    baseline_coverage_before = baseline_optimizer.cell_coverage.copy()
    baseline_gini_before = baseline_optimizer.calculate_gini_coefficient()
    
    # Relocate bases for baseline with updated probabilities
    print("\nBASELINE AFTER RELOCATION:")
    baseline_optimizer.relocate_bases(
        updated_search_probs=updated_search_probs,
        max_relocations=2,
        relocation_radius=20
    )
    
    # Evaluate baseline metrics after relocation
    baseline_coverage_after = baseline_optimizer.cell_coverage.copy()
    baseline_gini_after = baseline_optimizer.calculate_gini_coefficient()
    
    # Run PPR optimization
    print("\nPPR APPROACH:")
    # PPR optimization was already run in optimize_with_ppr()
    
    # Evaluate PPR metrics
    ppr_coverage_before = ppr_optimizer.cell_coverage.copy()
    ppr_gini_before = ppr_optimizer.calculate_gini_coefficient()
    
    # Relocate bases for PPR with updated probabilities
    print("\nPPR AFTER RELOCATION:")
    ppr_optimizer.relocate_bases(
        updated_search_probs=updated_search_probs,
        max_relocations=2,
        relocation_radius=20
    )
    
    # Evaluate PPR metrics after relocation
    ppr_coverage_after = ppr_optimizer.cell_coverage.copy()
    ppr_gini_after = ppr_optimizer.calculate_gini_coefficient()
    
    # Create comparison figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot baseline before and after relocation
    baseline_optimizer.search_probs = search_probs  # Set to initial probs for visualization
    baseline_optimizer.cell_coverage = baseline_coverage_before
    baseline_optimizer.plot_results(ax=axs[0, 0])
    axs[0, 0].set_title('Baseline Approach (Before Relocation)')
    
    baseline_optimizer.search_probs = updated_search_probs  # Set to updated probs
    baseline_optimizer.cell_coverage = baseline_coverage_after
    baseline_optimizer.plot_results(ax=axs[0, 1])
    axs[0, 1].set_title('Baseline Approach (After Relocation)')
    
    # Plot PPR before and after relocation
    ppr_optimizer.search_probs = search_probs  # Set to initial probs for visualization
    ppr_optimizer.cell_coverage = ppr_coverage_before
    ppr_optimizer.plot_results(ax=axs[1, 0])
    axs[1, 0].set_title('PPR Approach (Before Relocation)')
    
    ppr_optimizer.search_probs = updated_search_probs  # Set to updated probs
    ppr_optimizer.cell_coverage = ppr_coverage_after
    ppr_optimizer.plot_results(ax=axs[1, 1])
    axs[1, 1].set_title('PPR Approach (After Relocation)')
    
    plt.tight_layout()
    plt.show()
    
    # Return comparison results
    return {
        'baseline': {
            'before': {
                'coverage': np.mean(baseline_coverage_before) * 100,
                'gini': baseline_gini_before
            },
            'after': {
                'coverage': np.mean(baseline_coverage_after) * 100,
                'gini': baseline_gini_after
            }
        },
        'ppr': {
            'before': {
                'coverage': np.mean(ppr_coverage_before) * 100,
                'gini': ppr_gini_before
            },
            'after': {
                'coverage': np.mean(ppr_coverage_after) * 100,
                'gini': ppr_gini_after
            }
        }
    }

# Run the demonstration
if __name__ == "__main__":
    print("Running baseline demonstration:")
    optimizer = demonstration()
    
    print("\nComparing baseline and PPR approaches:")
    results = compare_approaches()
    
    # Print comparison summary
    print("\nCOMPARISON SUMMARY:")
    print(f"Baseline coverage before: {results['baseline']['before']['coverage']:.1f}%")
    print(f"Baseline coverage after: {results['baseline']['after']['coverage']:.1f}%")
    print(f"PPR coverage before: {results['ppr']['before']['coverage']:.1f}%")
    print(f"PPR coverage after: {results['ppr']['after']['coverage']:.1f}%")
    
    print(f"\nPercent improvement from baseline to PPR after relocation:")
    improvement = (results['ppr']['after']['coverage'] - results['baseline']['after']['coverage']) / results['baseline']['after']['coverage'] * 100
    print(f"Coverage improvement: {improvement:.1f}%")
