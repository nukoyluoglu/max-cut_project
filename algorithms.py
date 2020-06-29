import numpy as np

INITIAL_TEMPERATURE = 1e-2
MAX_NUM_ITER_COOLING = 10000
NO_CHANGE_TRESHOLD_COOLING = 5
MAX_NUM_ITER_EQUILIBRIUM = 5000
EQUILIBRIUM_TRESHOLD = 5

class MaxCutAlgorithm:

    def solve(self, problem): raise NotImplementedError("Override me")

class SimulatedAnnealing(MaxCutAlgorithm):

    def __init__(self):
        self.initial_temp = None
        self.cooling = None

    def solve(self, problem):
        num_temp_no_change = 0
        for k in range(MAX_NUM_ITER_COOLING):
            temp = self.initial_temp * np.power(self.cooling, k)
            energy_change_at_temp = 0
            num_equil = 0
            for _ in range(MAX_NUM_ITER_EQUILIBRIUM):
                v = problem.get_vertices()[np.random.choice(problem.get_num_vertices())]
                delta = problem.get_switch_energy_change(v)
                if delta < 0 or np.random.uniform() <= np.exp(- delta / temp):
                    problem.switch(v)
                    energy_change_at_temp += delta
                    num_equil = 0
                else:
                    num_equil += 1
                if num_equil >= EQUILIBRIUM_TRESHOLD and problem.get_best_energy() == problem.get_energy():
                    break          
            if energy_change_at_temp == 0:
                num_temp_no_change += 1
            else:
                num_temp_no_change = 0
            if num_temp_no_change >= NO_CHANGE_TRESHOLD_COOLING and problem.get_best_energy() == problem.get_energy():
                break
            

    def set_cooling_schedule(self, acceptance, cooling):
        if acceptance != 0:
            self.initial_temp = 1.0 / np.log(1.0 / acceptance)
        else:
            self.initial_temp = 0.001
        self.cooling = cooling

    # TODO: goal: efficient for short-range interaction

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

