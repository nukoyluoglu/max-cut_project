import numpy as np
import itertools
import copy

# MAX_NUM_ITER_COOLING = 100
MAX_NUM_ITER_COOLING = 100000
NO_CHANGE_TRESHOLD_COOLING = 5
NO_CHANGE_TRESHOLD_COOLING_ENSEMBLE = 50
# MAX_NUM_ITER_EQUILIBRIUM = 1000
MAX_NUM_ITER_EQUILIBRIUM = 1
BATCH_SIZE = 1000

class MaxCutAlgorithm:

    def solve(self, problem): raise NotImplementedError("Override me")

class SimulatedAnnealing(MaxCutAlgorithm):

    def __init__(self, ensemble=1):
        self.init_temp = None
        self.cool_rate = None
        self.temp_history = []
        self.ensemble = ensemble
        self.max_num_iter_equilibrium = MAX_NUM_ITER_EQUILIBRIUM
        self.max_num_iter_cooling = MAX_NUM_ITER_COOLING
        self.no_change_treshold_cooling = NO_CHANGE_TRESHOLD_COOLING

    def solve(self, problem):
        # num_temp_no_change = 0
        num_batch_no_change = 0
        batch_change = 0
        self.temp_history = [self.init_temp]
        for k in range(self.max_num_iter_cooling):
            temp = self.init_temp * np.power(self.cool_rate, k)
            # energy_change_at_temp = 0
            for _ in range(self.max_num_iter_equilibrium):
                v = problem.get_vertices()[np.random.choice(problem.get_num_vertices())]
                delta = problem.get_switch_energy_change(v)
                ratio = delta / temp if temp >= 1e-300 else float('inf')
                if delta <= 0 or np.random.uniform() <= np.exp(- ratio):
                    problem.switch(v)
                    # energy_change_at_temp += delta
                    batch_change += delta
                problem.get_partition_history().append(copy.copy(problem.get_partition()))
                problem.get_objective_history().append(copy.copy(problem.get_objective()))
                self.temp_history.append(temp)
            if k % BATCH_SIZE == BATCH_SIZE - 1 and batch_change == 0:
                num_batch_no_change += 1
                batch_change = 0
            elif k % BATCH_SIZE == BATCH_SIZE - 1:
                num_batch_no_change = 0
                batch_change = 0
            if num_batch_no_change >= self.no_change_treshold_cooling:
                break

            # if energy_change_at_temp == 0:
            #     num_temp_no_change += 1
            # else:
            #     num_temp_no_change = 0
            # if num_temp_no_change >= self.no_change_treshold_cooling:
            #     break
            
    def set_cooling_schedule(self, init_temp, cool_rate):
        self.init_temp = init_temp
        self.cool_rate = cool_rate

    def set_cooling_iter(self, total_num_iter_cooling):
        self.max_num_iter_cooling = total_num_iter_cooling
        self.no_change_treshold_cooling = float('inf')

    def get_temp_history(self):
        return self.temp_history

    # TODO: making approximately thermal state - distribution of final states - how close to Boltzman - hard to prepare in quantum system

    # energy scale of hamiltonian/final energy constant
    # ask energy fluctuations in infinite temperature state - should be constant as we scale hamiltonian
    # energy fluctuations scale as sqrt of number of spins

class SimulatedAnnealingEnsemble(MaxCutAlgorithm):

    def __init__(self, ensemble=1):
        self.init_temp = None
        self.cool_rate = None
        self.temp_history = []
        self.ensemble = ensemble
        self.max_num_iter_equilibrium = MAX_NUM_ITER_EQUILIBRIUM
        self.max_num_iter_cooling = MAX_NUM_ITER_COOLING
        self.no_change_treshold_cooling = NO_CHANGE_TRESHOLD_COOLING

    def solve(self, problem):
        num_temp_no_change = 0
        self.temp_history = [self.init_temp]
        for k in range(self.max_num_iter_cooling):
            temp = self.init_temp * np.power(self.cool_rate, k)
            energy_change_at_temp = 0
            temp_iter = 0
            while temp_iter < self.max_num_iter_equilibrium:
                delta_ensemble = 0
                ensemble = self.generate_cluster(problem, temp)
                for v in ensemble:
                    delta = problem.get_switch_energy_change(v)
                    problem.switch(v)
                    delta_ensemble += delta
                problem.get_partition_history().append(copy.copy(problem.get_partition()))
                problem.get_objective_history().append(copy.copy(problem.get_objective()))
                self.temp_history.append(temp) 
                temp_iter += len(ensemble)    
                energy_change_at_temp += delta_ensemble
            if energy_change_at_temp == 0:
                num_temp_no_change += 1
            else:
                num_temp_no_change = 0
            if num_temp_no_change >= self.no_change_treshold_cooling:
                break
            
    def set_cooling_schedule(self, init_temp, cool_rate):
        self.init_temp = init_temp
        self.cool_rate = cool_rate

    def get_temp_history(self):
        return self.temp_history

    def generate_cluster(self, problem, temp):
        # Wolff algorithm
        v_cluster = []
        v_stack = [problem.get_vertices()[np.random.choice(problem.get_num_vertices())]] 
        while v_stack:
            v = v_stack.pop()
            v_cluster.append(v)
            for n in problem.get_neighbors(v):
                ratio = problem.get_edge(v, n) / temp if temp >= 1e-300 else float('inf')
                # using how energy is symmetric w.r.t. partition
                # p = 1 - np.exp(-2 * ratio)
                p = np.exp(-2 * ratio)
                if problem.get_partition()[v] == problem.get_partition()[n] and np.random.uniform() <= p:
                    v_stack.append(n)
        return v_cluster

class BruteForce(MaxCutAlgorithm):
    # TODO: how number of partitions scale with system size as well as radius
    # does fixed radius lead to same number of ground states for each system size
    # phil anderson - spin glasses
    def solve(self, problem, allGroundStates=False):
        best_energy = 0
        sample_best_partition = copy.deepcopy(problem.get_partition())
        vertices = problem.get_vertices()
        num_best_partitions = 0
        num_partitions_explored = 0
        all_energies = []
        if allGroundStates:
            best_partitions = [sample_best_partition]
        for configuration in itertools.product([-1, 1], repeat=problem.get_num_vertices()):
            problem.set_partition({v: s for v, s in zip(vertices, configuration)})
            problem.set_objective()
            if problem.get_energy() < best_energy:
                best_energy = problem.get_energy()
                sample_best_partition = copy.deepcopy(problem.get_partition())
                num_best_partitions = 1
                if allGroundStates:
                    best_partitions = [sample_best_partition]
            elif problem.get_energy() == best_energy:
                num_best_partitions += 1
                if allGroundStates:
                    best_partitions.append(copy.deepcopy(problem.get_partition()))
            num_partitions_explored += 1
            all_energies.append(problem.get_energy())
        if allGroundStates:
            return best_energy, num_best_partitions, best_partitions
        return best_energy, num_best_partitions, sample_best_partition, num_partitions_explored, all_energies

class SemiDefinite(MaxCutAlgorithm):

    def solve(self, problem):
        print("to implement")

class Greedy(MaxCutAlgorithm):

    def solve(self, problem):
        print("to implement")

# glauber algorithm - for short range
# cluster algorithms - flip many spins at once

