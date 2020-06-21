import numpy as np

INITIAL_TEMPERATURE = 10000

class MaxCutAlgorithm:

    def solve(self, problem): raise NotImplementedError("Override me")

class SimulatedAnnealing(MaxCutAlgorithm):

    def solve(self, problem):
        for temp in range(INITIAL_TEMPERATURE, 0, -2e-6):
            v = np.random.choice(problem.get_vertices())
            delta = problem.get_switch_change(v)
            if np.random.uniform() <= np.exp(delta / temp):
                problem.switch(v)
        return problem.partition, problem.cur_obj, problem.best_obj

class SemiDefinite(MaxCutAlgorithm):

    def solve(self, problem):
        print("to implement")

class Greedy(MaxCutAlgorithm):

    def solve(self, problem):
        print("to implement")

