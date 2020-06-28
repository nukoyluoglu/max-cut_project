import util
from itertools import combinations
import numpy as np
import copy

class MaxCutProblem(util.Graph):

    def __init__(self, setup):
        super().__init__(setup.get_vertex_dict())
        self.partition = {v: 2 * np.random.randint(2) - 1 for v in self.get_vertices()}
        self.objective = self.calc_objective()
        self.switch_change = self.calc_objective_change_per_switch()
        self.partition_history = [copy.deepcopy(self.partition)]
        self.objective_history = [copy.deepcopy(self.objective)]
        self.best_partition = self.partition
        self.best_objective = self.objective
    
    def calc_objective(self):
        objective = 0
        for v1, v2 in combinations(self.get_vertices(), 2):
            if self.partition[v1] != self.partition[v2]:
                objective += self.get_edge(v1, v2)
        return objective

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
        self.partition_history.append(copy.deepcopy(self.partition))
        self.objective_history.append(copy.deepcopy(self.objective))
        if self.objective > self.best_objective:
            self.best_objective = self.objective
            self.best_partition = self.partition
        self.switch_change[v] = - self.switch_change[v]
        for n in self.get_neighbors(v):
            w = self.get_edge(v, n)
            if self.partition[v] == self.partition[n]:
                self.switch_change[n] += 2 * w
            else:
                self.switch_change[n] -= 2 * w
    
    def get_switch_change(self, v):
        return self.switch_change[v]

    def get_partition(self):
        return self.partition
    
    def get_objective(self):
        return self.objective

    def get_best_partition(self):
        return self.best_partition
    
    def get_best_objective(self):
        return self.best_objective

    def get_partition_history(self):
        return self.partition_history
    
    def get_objective_history(self):
        return self.objective_history





        


        
