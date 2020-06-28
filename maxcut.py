import experiment
import problem
import algorithms
import timeit
import util
import numpy as np

LATTICE_X = 5
LATTICE_Y = 5
LATTICE_SPACING = 1

def run(algorithm, interaction_fn, radius, trials, plot=False):
    if plot:
        # plot of radius vs. interaction strength
        util.plot_interaction(interaction_fn(radius), radius, LATTICE_X, LATTICE_Y)
    partition_history_trials = []
    objective_history_trials = []
    runtime_trials = []
    for _ in range(trials):
        setup = experiment.SpinLattice(LATTICE_X, LATTICE_Y, LATTICE_SPACING)
        setup.turn_on_interactions(interaction_fn(radius))
        prob = problem.MaxCutProblem(setup)
        myGlobals = globals()
        myGlobals.update({'algorithm': algorithm, 'prob': prob})
        algorithm.solve(prob)
        runtime = timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
        partition_history_trials.append(prob.get_partition_history)
        objective_history_trials.append(prob.get_objective_history())
        runtime_trials.append(runtime)
        if plot:
            # sample plot of radius vs. interaction strength
            util.plot_spin_lattice(prob.get_partition_history(), LATTICE_X, LATTICE_Y)
            # plot of objective vs. time
            util.plot_objective_in_time(prob.get_objective_history())
    return partition_history_trials, objective_history_trials, runtime_trials

def algorithm_performance(algorithm):
    # nearest-neighbor interaction
    partition_history_trials, objective_history_trials, runtime_trials = run(algorithm, experiment.step_fn, 1, 1, plot=True)

    # radius = np.arange(0, 5, 1)
    # time_complexity = [lambda r : run(algorithm, r) for r in radius]
    # util.plot_runtime(radius, time_complexity) 
    # print(run(algorithm, 1))
    # TODO: check scaling with system size
    # e.g. random coupling - exponential with system size
    # random coupling - how it scales with dimension
    # completely random graph
    # TODO: faster for short-range case: mechanism for knowing if it converges

def main(plot=True):
    algorithm_performance(algorithms.SimulatedAnnealing())
    # algorithm_performance(algorithms.SemiDefinite())

if __name__ == '__main__':
    main()
