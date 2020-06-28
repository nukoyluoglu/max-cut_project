import numpy as np

INITIAL_TEMPERATURE = 1e-2

class MaxCutAlgorithm:

    def solve(self, problem): raise NotImplementedError("Override me")

class SimulatedAnnealing(MaxCutAlgorithm):

    def solve(self, problem):
        for temp in np.arange(INITIAL_TEMPERATURE, 0, -1e-6):
            v = problem.get_vertices()[np.random.choice(problem.get_num_vertices())]
            delta = problem.get_switch_change(v)
            if delta > 0 or np.random.uniform() <= np.exp(delta / temp):
                problem.switch(v)
    # TODO: goal: efficient for short-range interaction
    
    # nearest-neighbor interactions on square lattice with constant temperature low enough
    # visualization
    # 10 x 10

    # metric - probability to find absolute ground state
    # check if you find state with the same energy each time - stability of the solution

    # making ground state
    # making approximately thermal state - distribution of final states - how close to Boltzman - hard to prepare in quantum system

    # energy scale of hamiltonian/final energy constant
    # ask energy fluctuations in infinite temperature state - should be constant as we scale hamiltonian
    # energy fluctuations scale as sqrt of number of spins

class SemiDefinite(MaxCutAlgorithm):

    def solve(self, problem):
        print("to implement")

class Greedy(MaxCutAlgorithm):

    def solve(self, problem):
        print("to implement")

# glauber algorithm - for short range
# cluster algorithms - flip many spins at once

