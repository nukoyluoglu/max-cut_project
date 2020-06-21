import util
import numpy as np
from itertools import combinations
import math

class SpinLattice(util.Graph):

    def __init__(self, lattice_X, lattice_Y, lattice_spacing):
        super().__init__()
        self.lattice_spacing = lattice_spacing
        self.num_rows = lattice_X / self.lattice_spacing
        self.num_cols = lattice_Y / self.lattice_spacing
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                self.add_vertex((r, c))

    def turn_on_interactions(self, interaction_fn):
        spins = self.get_vertices()
        for spin1, spin2 in combinations(spins, 2):
            dist = util.euclidean_dist_2D(spin1, spin2, self.lattice_spacing)
            
            # do not create edges corresponding to interactions close to 0
            strength = math.floor(interaction_fn(dist))
            if strength > 0:
                self.add_edge(spin1, spin2, strength)

def inverse_fn(alpha):
    def fn(dist):
        return np.power(dist, -1.0 * alpha)
    return fn

def logistic_decay_fn(amplitude, radius):
    def fn(dist):
        # shifts logistic decay function by radius
        return amplitude / (1.0 + np.exp(dist - radius))
    return fn