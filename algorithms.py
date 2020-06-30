import numpy as np
import itertools
import copy

INITIAL_TEMPERATURE = 1e-2
MAX_NUM_ITER_COOLING = 10000
NO_CHANGE_TRESHOLD_COOLING = 20
MAX_NUM_ITER_EQUILIBRIUM = 5000
EQUILIBRIUM_TRESHOLD = 10

class MaxCutAlgorithm:

    def solve(self, problem): raise NotImplementedError("Override me")

class SimulatedAnnealing(MaxCutAlgorithm):

    def __init__(self):
        self.initial_temp = None
        self.cooling = None
        self.temp_history = []

    def solve(self, problem):
        num_temp_no_change = 0
        self.temp_history = [self.initial_temp]
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
                problem.get_partition_history().append(copy.deepcopy(problem.get_partition()))
                problem.get_objective_history().append(copy.deepcopy(problem.get_objective()))
                self.temp_history.append(temp)
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
            self.initial_temp = 0.01
        self.cooling = cooling

    def get_temp_history(self):
        return self.temp_history

    # TODO: making approximately thermal state - distribution of final states - how close to Boltzman - hard to prepare in quantum system

    # energy scale of hamiltonian/final energy constant
    # ask energy fluctuations in infinite temperature state - should be constant as we scale hamiltonian
    # energy fluctuations scale as sqrt of number of spins

class BruteForce(MaxCutAlgorithm):
    
    def solve(self, problem):
        best_energy = 0
        best_partitions = []
        vertices = problem.get_vertices()
        num_partitions_explored = 0
        for configuration in itertools.product([-1, 1], repeat=problem.get_num_vertices()):
            problem.set_partition({v: s for v, s in zip(vertices, configuration)})
            problem.set_objective()
            if problem.get_energy() < best_energy:
                best_energy = problem.get_energy()
                best_partitions = [copy.deepcopy(problem.get_partition())]
            elif problem.get_energy() == best_energy:
                best_partitions.append(copy.deepcopy(problem.get_partition()))
            num_partitions_explored += 1
        return best_energy, best_partitions, num_partitions_explored
        




class SemiDefinite(MaxCutAlgorithm):

    def solve(self, problem):
        print("to implement")

class Greedy(MaxCutAlgorithm):

    def solve(self, problem):
        print("to implement")

# glauber algorithm - for short range
# cluster algorithms - flip many spins at once

