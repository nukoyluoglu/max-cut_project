import experiment
import problem
import algorithms
import timeit
import util
import numpy as np
import copy
import csv

LATTICE_X = 5
LATTICE_Y = 5
LATTICE_SPACING = 1
NUM_PERFORMANCE_TRIALS = 100
NUM_PARAM_TRIALS = 20  

def run(algorithm, interaction_fn, radius, exact_best_energy=None, plot=False):
    if plot:
        # plot of radius vs. interaction strength
        util.plot_interaction(interaction_fn(radius), radius, LATTICE_X, LATTICE_Y)
    partition_history_trials = []
    energy_history_trials = []
    temp_history_trials = []
    runtime_trials = []
    step_trials = []
    best_trial = 0
    best_energy = 0
    if exact_best_energy:
        best_energy = exact_best_energy
    for i in range(NUM_PERFORMANCE_TRIALS):
        setup = experiment.SpinLattice(LATTICE_X, LATTICE_Y, LATTICE_SPACING)
        setup.turn_on_interactions(interaction_fn(radius))
        prob = problem.MaxCutProblem(setup)
        myGlobals = globals()
        myGlobals.update({'algorithm': algorithm, 'prob': prob})
        runtime = timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
        partition_history_trials.append(copy.deepcopy(prob.get_partition_history()))
        energy_history_trials.append(copy.deepcopy(prob.get_energy_history()))
        temp_history_trials.append(copy.deepcopy(algorithm.get_temp_history()))
        runtime_trials.append(runtime)
        step_trials.append(len(algorithm.get_temp_history()))
        if not exact_best_energy and prob.get_best_energy() < best_energy:
            best_trial = i
            best_energy = prob.get_best_energy()
        elif exact_best_energy and prob.get_best_energy() == best_energy:
            best_trial = i
        elif exact_best_energy and prob.get_best_energy() < best_energy:
            raise RuntimeError("Energy cannot be lower than exact minimum energy")
    if plot: # display plot for best trial only
        # sample visualization of spin dynamics
        util.plot_spin_lattice(partition_history_trials[best_trial], LATTICE_X, LATTICE_Y, radius)
        # sample plot of energy vs. time
        # util.plot_energy_in_time(energy_history_trials[best_trial], radius)
        util.plot_energy_temp_in_time(energy_history_trials[best_trial], temp_history_trials[best_trial], radius)
    return {'partition_histories': partition_history_trials, 'energy_histories': energy_history_trials, 'runtimes': runtime_trials, 'temp_histories': temp_history_trials, 'steps': step_trials}

def get_simulated_annealing_optimal_params(algorithm, interaction_fn, radius, prev_best_acceptance, exact_best_energy=None, plot=False):
    acceptances = np.linspace(0, 0.9, 10)
    # coolings = np.linspace(0.8, 0.95, 4)
    coolings = np.linspace(0.8, 0.9, 3)
    best_runtime = float('inf')
    best_step = float('inf')
    best_params = {'acceptance': None, 'cooling': None}
    runtimes = np.full((len(acceptances), len(coolings)), float('nan'))
    steps = np.full((len(acceptances), len(coolings)), float('nan'))
    best_energy = 0
    if exact_best_energy:
        best_energy = exact_best_energy
    best_energy_prev = 0
    no_improvement = 0
    for i in range(int(10 * prev_best_acceptance), len(acceptances)): # consider acceptances starting from optimal value for previous radius
        best_energy_now = 0
        for j in range(len(coolings)):
            runtime = 0
            energy = 0
            step = 0
            for _ in range(NUM_PARAM_TRIALS):
                setup = experiment.SpinLattice(LATTICE_X, LATTICE_Y, LATTICE_SPACING)
                setup.turn_on_interactions(interaction_fn(radius))
                prob = problem.MaxCutProblem(setup)
                algorithm.set_cooling_schedule(acceptances[i], coolings[j])
                myGlobals = globals()
                myGlobals.update({'algorithm': algorithm, 'prob': prob})
                runtime += timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
                energy += prob.get_best_energy()
                step += len(algorithm.get_temp_history())
            runtime /= NUM_PARAM_TRIALS
            energy /= NUM_PARAM_TRIALS
            step /= NUM_PARAM_TRIALS
            if not exact_best_energy and energy < best_energy:
                best_energy = energy
                runtimes = np.full((len(acceptances), len(coolings)), float('nan'))
                runtimes[i][j] = runtime
                steps = np.full((len(acceptances), len(coolings)), float('nan'))
                steps[i][j] = step
                best_runtime = runtime
                best_step = step
                best_params['acceptance'] = acceptances[i]
                best_params['cooling'] = coolings[j]
            elif energy == best_energy:
                runtimes[i][j] = runtime
                if runtime < best_runtime:
                    best_runtime = runtime
                steps[i][j] = step
                if step < best_step:
                    best_step = step
                    best_params['acceptance'] = acceptances[i]
                    best_params['cooling'] = coolings[j]
            elif exact_best_energy and energy < best_energy:
                raise RuntimeError("Energy cannot be lower than exact minimum energy")
            best_energy_now = min(energy, best_energy_now)
        if exact_best_energy and best_energy_now == best_energy: # minimum possible energy is reached
            break
        if not exact_best_energy and best_energy_now >= best_energy_prev: # minimum energy did not decrease
            no_improvement += 1
        if no_improvement >= 3:
            break
        best_energy_prev = best_energy_now
    if plot:
        util.plot_params_performance(steps, acceptances, coolings, 'steps', 'acceptance probability', 'cooling coefficient', radius, best_params)
    return best_params

def get_exact_solutions(interaction_fn):
    exact_best_solution_by_radius = {}
    for radius in np.arange(1, util.euclidean_dist_2D((0, 0), (LATTICE_X, LATTICE_Y), 1)):
        setup = experiment.SpinLattice(LATTICE_X, LATTICE_Y, LATTICE_SPACING)
        setup.turn_on_interactions(interaction_fn(radius))
        prob = problem.MaxCutProblem(setup)
        exact_best_energy, exact_best_partitions, total_num_partitions = algorithms.BruteForce().solve(prob)
        sample_optimal_partition = [exact_best_partitions[0]]
        filename = 'step_fn_exact_sols_{}x{}/optimal_spin_lattice_radius_{}.html'.format(LATTICE_X, LATTICE_Y, radius)
        util.plot_spin_lattice(sample_optimal_partition, LATTICE_X, LATTICE_Y, radius, filename=filename)
        exact_best_solution_by_radius[radius] = {'minimum_energy': exact_best_energy, 'state_space_size': total_num_partitions}
    w = csv.writer(open('exact_solutions_{}x{}.csv'.format(LATTICE_X, LATTICE_Y), 'w'))
    for radius, solution in exact_best_solution_by_radius.items():
        w.writerow([radius, solution['minimum_energy'], solution['state_space_size']])
    return exact_best_solution_by_radius

def algorithm_performance(algorithm_name, interaction_fn, brute_force=False):
    if brute_force:
        exact_best_energy_by_radius = {}
        r = csv.reader(open('exact_solutions_{}x{}/exact_solutions_{}x{}.csv'.format(LATTICE_X, LATTICE_Y, LATTICE_X, LATTICE_Y), 'r'))
        for row in r:
            exact_best_energy_by_radius[float(row[0])] = float(row[1])
    if algorithm_name == 'simulated annealing':
        algorithm_performance_by_radius = {}
        prev_best_acceptance = 0
        for radius in np.arange(1, util.euclidean_dist_2D((0, 0), (LATTICE_X, LATTICE_Y), 1)):
            exact_best_energy = None
            if brute_force:
                exact_best_energy = exact_best_energy_by_radius[radius]
            algorithm = algorithms.SimulatedAnnealing()
            plot = True
            best_params = get_simulated_annealing_optimal_params(algorithm, interaction_fn, radius, prev_best_acceptance, exact_best_energy=exact_best_energy, plot=plot)
            algorithm.set_cooling_schedule(best_params['acceptance'], best_params['cooling'])
            algorithm_performance_by_radius[radius] = run(algorithm, interaction_fn, radius, exact_best_energy=exact_best_energy, plot=plot)
            prev_best_acceptance = best_params['acceptance']
        radii = []
        runtimes = []
        steps = []
        for radius, performance in algorithm_performance_by_radius.items():
            radii.append(radius)
            runtimes.append(np.mean(performance['runtimes']))
            steps.append(np.mean(performance['steps']))
        util.plot_runtime(radii, runtimes)
        util.plot_steps(radii, steps)
        # print(run(algorithm, 1))
        # TODO: check scaling with system size
        # e.g. random coupling - exponential with system size
        # random coupling - how it scales with dimension
        # completely random graph
        # TODO: faster for short-range case: mechanism for knowing if it converges
        # TODO: filling of the lattice - randomly turn off corners
        # TODO: different interaction functions

def main(plot=True):
    # get_exact_solutions(experiment.step_fn)
    algorithm_performance('simulated annealing', experiment.step_fn)
    # algorithm_performance(algorithms.SemiDefinite())

if __name__ == '__main__':
    main()
