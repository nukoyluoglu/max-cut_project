import util
from itertools import combinations
import numpy as np

class MaxCutProblem(util.Graph):

    def __init__(self, setup):
        super().__init__(setup.get_vertex_dict())
        self.partition = np.random.randint(2, size=self.num_vertices)
        self.cur_obj = self.get_objective()
        self.best_obj = self.cur_obj
        self.switch_change = self.get_objective_change_per_switch()
    
    def get_objective(self):
        obj = 0
        for v1, v2 in combinations(self.get_vertices(), 2):
            if self.partition[v1] != self.partition[v2]:
                obj += self.get_edge(v1, v2)
        return obj

    def get_objective_change_per_switch(self):
        change = np.zeros(self.num_vertices)
        for v in self.get_vertices():
            for n in self.get_neighbors(v):
                w = self.get_edge(v, n)
                if self.partition[v] == self.partition[n]:
                    change[v] += w
                else:
                    change[v] -= w
        return change

    def switch(self, v):
        self.partition[v] = 1 - self.partition[v]
        self.cur_obj += self.switch_change[v]
        self.switch_change[v] = - self.switch_change[v]
        for n in self.get_neighbors(v):
            w = self.get_edge(v, n)
            if self.partition[v] == self.partition[n]:
                self.switch_change[n] += 2 * w
            else:
                self.switch_change[n] -= 2 * w
        self.best_obj = max(self.cur_obj, self.best_obj)
    
    def get_switch_change(self, v):
        return self.switch_change[v]





        


        
