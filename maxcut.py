import experiment
import algorithms
import timeit
import util
import numpy as np

LATTICE_X = 100
LATTICE_Y = 100
LATTICE_SPACING = 5
INTERACTION_AMPLITUDE = 50

def runtime(algorithm, radius):
    SETUP_CODE = '''
setup = experiment.SpinLattice(LATTICE_X, LATTICE_Y, LATTICE_SPACING)
interaction_fn = experiment.logistic_decay_fn(INTERACTION_AMPLITUDE, radius)
setup.turn_on_interactions(interaction_fn)
prob = problem.MaxCutProblem(setup)'''

    TEST_CODE = '''
algorithm.solve(prob)'''
    
    times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=5)
    return np.mean(times)

def algorithm_performance(algorithm):
    radius = np.arange(0, 100, 5)
    time_complexity = [lambda r : runtime(algorithm, r) for r in radius]
    util.plot_runtime(radius, time_complexity) 

def main(plot=True):
    # sample plot of radius vs. interaction strength, with amplitude=50 and radius=30
    util.plot_interaction(experiment.logistic_decay_fn(INTERACTION_AMPLITUDE, 30), 30)

    algorithm_performance(algorithms.SimulatedAnnealing())
    algorithm_performance(algorithms.SemiDefinite())

if __name__ == '__main__':
    main()
