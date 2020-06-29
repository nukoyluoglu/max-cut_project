import experiment
import problem
import algorithms
import timeit
import util
import numpy as np

LATTICE_X = 5
LATTICE_Y = 5
LATTICE_SPACING = 1
NUM_PERFORMANCE_TRIALS = 100
NUM_PARAM_TRIALS = 20

def run(algorithm, interaction_fn, radius, plot=False):
    if plot:
        # plot of radius vs. interaction strength
        util.plot_interaction(interaction_fn(radius), radius, LATTICE_X, LATTICE_Y)
    partition_history_trials = []
    energy_history_trials = []
    runtime_trials = []
    for i in range(NUM_PERFORMANCE_TRIALS):
        setup = experiment.SpinLattice(LATTICE_X, LATTICE_Y, LATTICE_SPACING)
        setup.turn_on_interactions(interaction_fn(radius))
        prob = problem.MaxCutProblem(setup)
        myGlobals = globals()
        myGlobals.update({'algorithm': algorithm, 'prob': prob})
        runtime = timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
        partition_history_trials.append(prob.get_partition_history)
        energy_history_trials.append(prob.get_energy_history())
        runtime_trials.append(runtime)
        if plot and i == 0: # display plot for 1 trial only
            # sample visualization of spin dynamics
            util.plot_spin_lattice(prob.get_partition_history(), LATTICE_X, LATTICE_Y, radius)
            # sample plot of energy vs. time
            util.plot_energy_in_time(prob.get_energy_history(), radius)
    return {'partition_histories': partition_history_trials, 'energy_histories': energy_history_trials, 'runtimes': runtime_trials}

def get_simulated_annealing_optimal_params(algorithm, interaction_fn, radius, plot=False):
    acceptances = np.linspace(0, 1, 11)
    coolings = np.linspace(0.8, 0.95, 4)
    best_energy = 0
    best_runtime = float('inf')
    best_params = {'acceptance': None, 'cooling': None}
    runtimes = np.full((len(acceptances), len(coolings)), float('nan'))
    prev_best_energy = 0
    for i in range(len(acceptances)):
        for j in range(len(coolings)):
            runtime = 0
            energy = 0
            for _ in range(NUM_PARAM_TRIALS):
                setup = experiment.SpinLattice(LATTICE_X, LATTICE_Y, LATTICE_SPACING)
                setup.turn_on_interactions(interaction_fn(radius))
                prob = problem.MaxCutProblem(setup)
                algorithm.set_cooling_schedule(acceptances[i], coolings[j])
                myGlobals = globals()
                myGlobals.update({'algorithm': algorithm, 'prob': prob})
                runtime += timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
                energy += prob.get_best_energy()
            runtime /= NUM_PARAM_TRIALS
            energy /= NUM_PARAM_TRIALS
            if energy < best_energy:
                best_energy = energy
                runtimes = np.full((len(acceptances), len(coolings)), float('nan'))
                best_runtime = runtime
                best_params['acceptance'] = acceptances[i]
                best_params['cooling'] = coolings[j]
            elif runtime < best_runtime:
                best_runtime = runtime
                best_params['acceptance'] = acceptances[i]
                best_params['cooling'] = coolings[j]
            runtimes[i][j] = runtime
        if best_energy == prev_best_energy:
            break
        prev_best_energy = best_energy
    if plot:
        util.plot_params_performance(runtimes, acceptances, coolings, 'runtime', 'acceptance probability', 'cooling coefficient', radius, best_params)
    return best_params

def algorithm_performance(algorithm_name):
    algorithm_performance_by_radius = {}
    if algorithm_name == 'simulated annealing':
        for radius in np.arange(1, util.euclidean_dist_2D((0, 0), (LATTICE_X, LATTICE_Y), 1)):
            algorithm = algorithms.SimulatedAnnealing()
            plot = True
            best_params = get_simulated_annealing_optimal_params(algorithm, experiment.step_fn, radius, plot=plot)
            algorithm.set_cooling_schedule(best_params['acceptance'], best_params['cooling'])
            algorithm_performance_by_radius[radius] = run(algorithm, experiment.step_fn, radius, plot=plot)
        radii = []
        runtimes = []
        for radius, performance in algorithm_performance_by_radius.items():
            radii.append(radius)
            runtimes.append(np.mean((performance['runtimes'])))
        util.plot_runtime(radii, runtimes)
        # print(run(algorithm, 1))
        # TODO: check scaling with system size
        # e.g. random coupling - exponential with system size
        # random coupling - how it scales with dimension
        # completely random graph
        # TODO: faster for short-range case: mechanism for knowing if it converges
        # TODO: filling of the lattice - randomly turn off corners
        # TODO: different interaction functions

def main(plot=True):
    algorithm_performance('simulated annealing')
    # algorithm_performance(algorithms.SemiDefinite())

if __name__ == '__main__':
    main()
