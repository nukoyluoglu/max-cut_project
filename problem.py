import util
from itertools import combinations
import numpy as np
import copy

class MaxCutProblem(util.Graph):

    def __init__(self, setup):
        super().__init__(setup.get_vertex_dict())
        np.random.seed()
        self.partition = {v: 2 * np.random.randint(2) - 1 for v in self.get_vertices()}
        self.objective = 0
        self.set_objective()
        self.switch_change = self.calc_objective_change_per_switch()
        self.partition_history = [copy.copy(self.partition)]
        self.objective_history = [copy.copy(self.objective)]
        self.best_partition = self.partition
        self.best_objective = self.objective
    
    def set_objective(self):
        objective = 0
        for v1, v2 in combinations(self.get_vertices(), 2):
            if self.partition[v1] != self.partition[v2]:
                objective += self.get_edge(v1, v2)
        self.objective = objective

    def calc_objective_change_per_switch(self):
        change = {v: 0 for v in self.get_vertices()}
        for v in self.get_vertices():
            for n in self.get_neighbors(v):
                w = self.get_edge(v, n)
                if self.partition[v] == self.partition[n]:
                    change[v] += w
                else:
                    change[v] -= w
        return change

    def switch(self, v):
        self.partition[v] = - self.partition[v]
        self.objective += self.switch_change[v]
        if self.objective > self.best_objective:
            self.best_objective = self.objective
            self.best_partition = self.partition
        self.switch_change[v] = - self.switch_change[v]
        for n in self.get_neighbors(v):
            w = self.get_edge(v, n)
            if self.partition[v] == self.partition[n]:
                self.switch_change[n] += 2.0 * w
            else:
                self.switch_change[n] -= 2.0 * w

    def switch_ensemble(self, v_ensemble):
        for v in v_ensemble:
            self.partition[v] = - self.partition[v]
            self.objective += self.switch_change[v]
            self.switch_change[v] = - self.switch_change[v]
            for n in self.get_neighbors(v):
                w = self.get_edge(v, n)
                if n in v_ensemble:
                    if self.partition[v] == self.partition[n]:
                        self.objective -= w
                        self.switch_change[n] += w
                    else:
                        self.objective += w
                        self.switch_change[n] -= w
                else:
                    if self.partition[v] == self.partition[n]:
                        self.switch_change[n] += 2.0 * w
                    else:
                        self.switch_change[n] -= 2.0 * w
        if self.objective > self.best_objective:
            self.best_objective = self.objective
            self.best_partition = self.partition
 
    def get_switch_objective_change(self, v):
        return self.switch_change[v]

    def get_switch_energy_change(self, v):
        return - self.switch_change[v]

    def get_switch_ensemble_objective_change(self, v_ensemble):
        delta = 0.0
        for v in v_ensemble:
            delta += self.switch_change[v]
            for n in self.get_neighbors(v):
                w = self.get_edge(v, n)
                if n in v_ensemble:
                    if self.partition[v] == self.partition[n]:
                        delta -= w
                    else:
                        delta += w
        return delta

    def get_switch_ensemble_energy_change(self, v_ensemble):
        return - self.get_switch_ensemble_objective_change(v_ensemble)

    def get_partition(self):
        return self.partition
    
    def get_objective(self):
        return self.objective

    def get_energy(self):
        return - self.objective

    def get_best_partition(self):
        return self.best_partition
    
    def get_best_objective(self):
        return self.best_objective

    def get_best_energy(self):
        return - self.best_objective

    def get_partition_history(self):
        return self.partition_history
    
    def get_objective_history(self):
        return self.objective_history

    def get_energy_history(self):
        return [- objective for objective in self.objective_history]

    def set_partition(self, partition):
        self.partition = partition




        


        
