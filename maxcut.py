import sys
import experiment
import problem
import algorithms
import timeit
import util
import collections
import numpy as np
import copy
import csv

LATTICE_SPACING = 1
NUM_PERFORMANCE_TRIALS = 100
NUM_PARAM_TRIALS = 50

def initialize_problem(interaction_fn, radius, lattice_X, lattice_Y, fill=1.0, triangular=False):
    setup = experiment.SpinLattice(lattice_X, lattice_Y, LATTICE_SPACING, prob_full=fill, triangular=triangular)
    setup.turn_on_interactions(interaction_fn(radius))
    return problem.MaxCutProblem(setup)

def run_trials_for_param(algorithm, interaction_fn, radius, lattice_X, lattice_Y, init_temp, cool_rate, fill=1.0, triangular=False):
    ave_runtime = 0
    ave_step = 0
    ave_final_energy = 0
    best_final_energy = 0
    ave_energy_vs_temp = {}
    ave_energy_hist = {}
    ave_temp_hist = {}
    sample_best_partition_hist = None
    for _ in range(NUM_PARAM_TRIALS):
        prob = initialize_problem(interaction_fn, radius, lattice_X, lattice_Y, fill=fill, triangular=triangular)
        algorithm.set_cooling_schedule(init_temp, cool_rate)
        myGlobals = globals()
        myGlobals.update({'algorithm': algorithm, 'prob': prob})
        runtime = timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
        # debug jumps
        # if prob.objective < 34.446:
        #     print('LOCALMIN')
        #     for t in range(999, len(prob.objective_history), 1000):
        #         print(prob.get_objective_history()[t], algorithm.get_temp_history()[t])
        #     util.plot_temporary(prob.get_energy_history(), algorithm.get_temp_history(), radius, lattice_X, lattice_Y, interaction_shape, init_temp, cool_rate, _)
        step = len(algorithm.get_temp_history())
        # TODO: look at min energy rather than final energy (if it touched min once it's enough)
        # TODO: random coupling
        final_energy = prob.get_energy()
        energy_hist = prob.get_energy_history()
        temp_hist = algorithm.get_temp_history()
        ave_runtime += runtime
        ave_step += step
        ave_final_energy += final_energy
        if final_energy < best_final_energy:
            best_final_energy = final_energy
            sample_best_partition_hist = prob.get_partition_history()
        if len(temp_hist) != len(energy_hist):
            raise RuntimeError('Length of temperature and energy histories must match')
        temp_start = 0
        for t in range(len(temp_hist)):
            temp = temp_hist[t]
            if t != 0 and temp != temp_hist[t - 1]:
                temp_start = t
            if t - temp_start < algorithm.max_num_iter_equilibrium / 2: # consider second half at a temperature as equilibrium
                ave_energy_vs_temp.setdefault(temp, []).append(energy_hist[t])
            ave_energy_hist.setdefault(t, []).append(energy_hist[t])
            ave_temp_hist.setdefault(t, []).append(temp)
    for temp in ave_energy_vs_temp.keys():
        energies_at_temp = ave_energy_vs_temp[temp]
        ave_energy_vs_temp[temp] = (np.mean(energies_at_temp), np.std(energies_at_temp))
    for t in ave_energy_hist.keys():
        energies_at_t = ave_energy_hist[t]
        ave_energy_hist[t] = np.mean(energies_at_t)
        temps_at_t = ave_temp_hist[t]
        ave_temp_hist[t] = np.mean(temps_at_t)
    ave_final_energy /= NUM_PARAM_TRIALS
    ave_step /= NUM_PARAM_TRIALS
    ave_runtime /= NUM_PARAM_TRIALS
    # plot of energy vs. time
    util.plot_params_energy_temp_vs_step(ave_energy_hist, ave_temp_hist, radius, lattice_X, lattice_Y, interaction_shape, init_temp, cool_rate)
    return {'ave_final_energy': ave_final_energy, 'ave_runtime': ave_runtime, 'ave_step': ave_step, 'best_final_energy': best_final_energy, 'ave_energy_vs_temp': ave_energy_vs_temp, 'ave_energy_hist': ave_energy_hist, 'ave_temp_hist': ave_temp_hist, 'sample_best_partition_hist': sample_best_partition_hist}

def search_params(algorithm_name, interaction_fn, radius, lattice_X, lattice_Y, interaction_shape, ensemble=1, exact_best_energy=None, fill=1.0, triangular=False, plot=True):
    if algorithm_name == 'simulated annealing':
        algorithm = algorithms.SimulatedAnnealing(ensemble=ensemble)
        init_temps = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
        cool_rates = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        best_solution = {}
        best_params = {'init_temp': None, 'cool_rate': None}
        ave_energy_vs_temp_by_params = collections.defaultdict(dict)
        for i in range(len(init_temps)): # consider acceptances starting from optimal value for previous radius
            for j in range(len(cool_rates)):
                param_solution = run_trials_for_param(algorithm, interaction_fn, radius, lattice_X, lattice_Y, init_temps[i], cool_rates[j], fill=fill, triangular=triangular)
                ave_energy_vs_temp_by_params[init_temps[i]][cool_rates[j]] = param_solution['ave_energy_vs_temp']
                if not best_solution or param_solution['best_final_energy'] < best_solution['best_final_energy'] or (param_solution['best_final_energy'] == best_solution['best_final_energy'] and param_solution['ave_step'] < best_solution['ave_step']):
                    best_solution = param_solution
                    best_params['init_temp'] = init_temps[i]
                    best_params['cool_rate'] = cool_rates[j]
                elif best_solution == param_solution:
                    raise RuntimeError('Best solution must be preserved throughout trials')
                # if exact_best_energy and best_solution['best_final_energy'] < exact_best_energy:
                #     raise RuntimeError('Energy cannot be lower than exact minimum energy')
        if plot:
            util.plot_params_energy_vs_temp(ave_energy_vs_temp_by_params, best_params, radius, lattice_X, lattice_Y, interaction_shape, exact_best_energy=exact_best_energy)
            util.plot_params_energy_vs_temp_heatmap(ave_energy_vs_temp_by_params, best_params, radius, lattice_X, lattice_Y, interaction_shape, exact_best_energy=exact_best_energy)
            # sample visualization of spin dynamics for best params
            util.plot_spin_lattice([best_solution['sample_best_partition_hist'][-1]], radius, lattice_X, lattice_Y, interaction_shape, triangular=triangular)
        return best_solution, best_params

def get_exact_solutions(lattice_X, lattice_Y, interaction_shape, fill=1.0, triangular=False):
    radius_range = np.arange(1.0, util.euclidean_dist_2D((0.0, 0.0), (lattice_X, lattice_Y), 1.0))
    if interaction_shape == 'step_fn':
        interaction_fn = experiment.step_fn
    elif interaction_shape == 'power_decay_fn':
        interaction_fn = experiment.power_decay_fn
    elif interaction_shape == 'random':
        interaction_fn = experiment.random
        radius_range = np.zeros(1)
    w1 = csv.writer(open('{}_exact_sols_{}x{}/optimal_sols.csv'.format(interaction_shape, lattice_X, lattice_Y), 'w'))
    w1.writerow(['radius', 'min energy', '# ground states', '# states'])
    for radius in radius_range:
        prob = prob = initialize_problem(interaction_fn, radius, lattice_X, lattice_Y, fill=fill, triangular=triangular)
        exact_best_energy, num_best_partitions, sample_best_partition, total_num_partitions, all_partition_energies = algorithms.BruteForce().solve(prob)
        filename = '{}_exact_sols_{}x{}/optimal_spin_lattice_radius_{}'.format(interaction_shape, lattice_X, lattice_Y, radius)
        util.plot_spin_lattice([sample_best_partition], radius, lattice_X, lattice_Y, interaction_shape, filename=filename, triangular=triangular)
        w1.writerow([radius, exact_best_energy, num_best_partitions, total_num_partitions])
        w2 = csv.writer(open('{}_exact_sols_{}x{}/energies_radius_{}.csv'.format(interaction_shape, lattice_X, lattice_Y, radius), 'w'))
        w2.writerow(['energy'])
        for energy in all_partition_energies:
            w2.writerow([energy])

def algorithm_performance(algorithm_name, lattice_X, lattice_Y, interaction_shape, fill=1.0, triangular=False, ensemble=1, brute_force_solution=None, plot=True):
    algorithm_performance_by_radius = {}
    radius_range = np.arange(1.0, util.euclidean_dist_2D((0.0, 0.0), (lattice_X, lattice_Y), 1.0))
    if interaction_shape == 'step_fn':
        interaction_fn = experiment.step_fn
    elif interaction_shape == 'power_decay_fn':
        interaction_fn = experiment.power_decay_fn
    elif interaction_shape == 'random':
        interaction_fn = experiment.random
        radius_range = np.zeros(1)
    for radius in radius_range:
        if plot and interaction_shape != 'random':
            # plot of radius vs. interaction strength
            util.plot_interaction(interaction_fn(radius), radius, lattice_X, lattice_Y, interaction_shape)
        exact_best_energy = None
        if brute_force_solution:
            exact_best_energy = brute_force_solution[radius]
        algorithm_performance_by_radius[radius] = search_params(algorithm_name, interaction_fn, radius, lattice_X, lattice_Y, interaction_shape, ensemble=ensemble, exact_best_energy=exact_best_energy, fill=fill, triangular=triangular)
    if plot and interaction_shape != 'random':
        util.plot_runtimes_steps_vs_radius(algorithm_performance_by_radius, lattice_X, lattice_Y, interaction_shape)
        # TODO: check scaling with system size
        # e.g. random coupling - exponential with system size
        # random coupling - how it scales with dimension
        # completely random graph
        # TODO: faster for short-range case: mechanism for knowing if it converges
        # TODO: filling of the lattice - randomly turn off corners
        # TODO: different interaction functions
        # TODO: find convergence without imposing cutoff for decaying interaction functions
        # TODO: try ensemble switch
    return algorithm_performance_by_radius

def main(interaction_shape='step_fn', plot=True):
    algorithm_performance_by_system = {}
    num_ground_states_by_system = {}
    for system_size in range(3, 6):
        lattice_X = system_size
        lattice_Y = system_size
        exact_min_energy_by_radius = None
        if system_size < 5:
            # brute force, step function
            get_exact_solutions(lattice_X, lattice_Y, interaction_shape, fill=1.0, triangular=True)
        if system_size < 5:
            exact_min_energy_by_radius = {}
            num_ground_states_by_radius = {}
            r = csv.reader(open('{}_exact_sols_{}x{}/optimal_sols.csv'.format(interaction_shape, lattice_X, lattice_Y), 'r'))
            next(r)
            for row in r:
                radius = float(row[0])
                exact_min_energy_by_radius[radius] = float(row[1])
                num_ground_states_by_radius[radius] = float(row[2])
            num_ground_states_by_system[system_size] = num_ground_states_by_radius
        # simulated annealing
        algorithm_performance_by_radius = algorithm_performance('simulated annealing', lattice_X, lattice_Y, interaction_shape, fill=1.0, triangular=True, ensemble=1, brute_force_solution=exact_min_energy_by_radius)
        algorithm_performance_by_system[system_size] = algorithm_performance_by_radius
    util.plot_runtimes_steps_vs_system_size(algorithm_performance_by_system, interaction_shape)
    if num_ground_states_by_system:
        util.plot_num_ground_states_vs_system_size(num_ground_states_by_system, interaction_shape)


    # get_exact_solutions(experiment.step_fn, 'step_fn_0.8_filled', fill=0.8)
    # since system is probabilistic, brute force is not accurate
    # algorithm_performance('simulated annealing', experiment.step_fn, 'step_fn_0.8_filled', brute_force=False, fill=0.8)

if __name__ == '__main__':
    for i in range(1, len(sys.argv)):    
        interaction_shape = str(sys.argv[i])
        main(interaction_shape)
