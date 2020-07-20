import util
import numpy as np
from itertools import combinations
import math

class SpinLattice(util.Graph):

    def __init__(self, lattice_X, lattice_Y, lattice_spacing, prob_full=1.0, triangular=False):
        super().__init__()
        np.random.seed(0)
        self.lattice_spacing = lattice_spacing
        self.num_rows = int(lattice_X / self.lattice_spacing)
        self.num_cols = int(lattice_Y / self.lattice_spacing)
        if not triangular:
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if np.random.uniform() <= prob_full: # turn off cells randomly
                        self.add_vertex((r, c))
        if triangular:
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if np.random.uniform() <= prob_full: # turn off cells randomly
                        self.add_vertex((r * np.sqrt(3) / 2, c + (r % 2) / 2))

    def turn_on_interactions(self, interaction_fn):
        spins = self.get_vertices()
        for spin1, spin2 in combinations(spins, 2):
            dist = util.euclidean_dist_2D(spin1, spin2, self.lattice_spacing)
            # TODO: cut-off or no cut-off?
            # do not create edges corresponding to interactions close to 0
            # strength = math.floor(interaction_fn(dist))
            strength = interaction_fn(dist)
            if strength > 0:
                self.add_edge(spin1, spin2, strength)

class FreeSpins(util.Graph):

    def __init__(self, num_particles):
        super().__init__()
        np.random.seed(0)
        self.num_particles = num_particles
        for p in range(self.num_particles):
            self.add_vertex(p)

    def turn_on_interactions(self, interaction_fn):
        spins = self.get_vertices()
        for spin1, spin2 in combinations(spins, 2):
            strength = interaction_fn()
            if strength > 0:
                self.add_edge(spin1, spin2, strength)

def power_decay_fn(radius, alpha=6.0):
    def fn(dist):
        return 1.0 / (1.0 + np.power(dist / radius, alpha))
    return fn

def logistic_decay_fn(radius, beta=6.0):
    def fn(dist):
        # shifts logistic decay function by radius
        return 1.0 / (1.0 + np.exp(beta * (dist - radius)))
    return fn

def step_fn(radius, alpha=None):
    def fn(dist):
        return 1 if dist <= radius else 0
    return fn

def random(radius=None, alpha=None):
    np.random.seed(30)
    def fn(dist=None):
        return np.random.uniform()
    return fn