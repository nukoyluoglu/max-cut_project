import sys
import os
import experiment
import problem
import algorithms
import maxcut_free
import timeit
import util
import collections
import numpy as np
import copy
import csv
import argparse

LATTICE_SPACING = 1
NUM_PERFORMANCE_TRIALS = 100
NUM_PARAM_TRIALS = 50

def initialize_problem(interaction_fn, radius, lattice_X, lattice_Y, fill=1.0, triangular=False):
    setup = experiment.SpinLattice(lattice_X, lattice_Y, LATTICE_SPACING, prob_full=fill, triangular=triangular)
    setup.turn_on_interactions(interaction_fn(radius))
    return problem.MaxCutProblem(setup)

def run_trials_for_param(algorithm, interaction_fn, radius, lattice_X, lattice_Y, init_temp, cool_rate, interaction_shape, fill, triangular, path):
    ave_runtime = 0
    ave_step = 0
    ave_best_energy = 0
    min_best_energy = 0
    ave_energy_vs_temp = {}
    ave_energy_hist = {}
    ave_temp_hist = {}
    sample_best_prob = None
    for _ in range(NUM_PARAM_TRIALS):
        prob = initialize_problem(interaction_fn, radius, lattice_X, lattice_Y, fill=fill, triangular=triangular)
        algorithm.set_cooling_schedule(init_temp, cool_rate)
        myGlobals = globals()
        myGlobals.update({'algorithm': algorithm, 'prob': prob})
        runtime = timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
        step = len(algorithm.get_temp_history())
        best_energy = prob.get_best_energy()
        energy_hist = prob.get_energy_history()
        temp_hist = algorithm.get_temp_history()
        ave_runtime += runtime
        ave_step += step
        ave_best_energy += best_energy
        if best_energy < min_best_energy:
            min_best_energy = best_energy
            sample_best_prob = prob
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
    ave_best_energy /= NUM_PARAM_TRIALS
    ave_step /= NUM_PARAM_TRIALS
    ave_runtime /= NUM_PARAM_TRIALS
    # plot of energy and temperature vs. time
    util.plot_energy_temp_vs_step(ave_energy_hist, ave_temp_hist, radius, lattice_X ** 2, interaction_shape, init_temp, cool_rate, path, None)
    # return {'ave_final_energy': ave_final_energy, 'ave_runtime': ave_runtime, 'ave_step': ave_step, 'best_final_energy': best_final_energy, 'ave_energy_vs_temp': ave_energy_vs_temp, 'ave_energy_hist': ave_energy_hist, 'ave_temp_hist': ave_temp_hist, 'sample_best_prob': sample_best_prob}
    return {'ave_best_energy': ave_best_energy, 'ave_runtime': ave_runtime, 'ave_step': ave_step, 'best_energy': min_best_energy, 'ave_energy_vs_temp': ave_energy_vs_temp, 'ave_energy_hist': ave_energy_hist, 'ave_temp_hist': ave_temp_hist, 'sample_best_prob': sample_best_prob}

def search_params(algorithm_name, interaction_fn, radius, lattice_X, lattice_Y, interaction_shape, ensemble, exact_best_energy, fill, triangular, path, exact_path, plot=True):
    if algorithm_name == 'simulated annealing':
        algorithm = algorithms.SimulatedAnnealing()
        init_temps = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
        cool_rates = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        if ensemble:
            algorithm = algorithms.SimulatedAnnealingEnsemble()
            init_temps = np.array([1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
        best_solution = {}
        best_params = {'init_temp': None, 'cool_rate': None}
        ave_energy_vs_temp_by_params = collections.defaultdict(dict)
        for i in range(len(init_temps)):
            for j in range(len(cool_rates)):
                param_solution = run_trials_for_param(algorithm, interaction_fn, radius, lattice_X, lattice_Y, init_temps[i], cool_rates[j], interaction_shape, fill, triangular, path)
                ave_energy_vs_temp_by_params[init_temps[i]][cool_rates[j]] = param_solution['ave_energy_vs_temp']
                if not best_solution or param_solution['best_energy'] < best_solution['best_energy'] or (param_solution['best_energy'] == best_solution['best_energy'] and param_solution['ave_step'] < best_solution['ave_step']):
                    best_solution = param_solution
                    best_params['init_temp'] = init_temps[i]
                    best_params['cool_rate'] = cool_rates[j]
                elif best_solution == param_solution:
                    raise RuntimeError('Best solution must be preserved throughout trials')
                if exact_best_energy and best_solution['best_energy'] < exact_best_energy:
                    print('exact best energy: {}'.format(exact_best_energy))
                    print('algo best energy: {}'.format(best_solution['best_energy']))
                    print('algo best partition: {}'.format(best_solution['sample_best_prob'].get_partition()))
        if plot:
            util.plot_params_energy_vs_temp(ave_energy_vs_temp_by_params, best_params, radius, lattice_X ** 2, interaction_shape, path, exact_best_energy, exact_path)
            util.plot_params_energy_vs_temp_heatmap(ave_energy_vs_temp_by_params, best_params, radius, lattice_X ** 2, interaction_shape, path, exact_best_energy, exact_path)
            # sample visualization of spin dynamics for best params
            sample_best_prob = best_solution['sample_best_prob']
            util.plot_spin_lattice(sample_best_prob.get_partition_history(), sample_best_prob.get_energy_history(), radius, lattice_X, lattice_Y, interaction_shape, triangular, path)
        return best_solution, best_params  

def algorithm_performance(algorithm_name, lattice_X, lattice_Y, interaction_shape, fill, triangular, ensemble, path, brute_force_solution=None, brute_force_path=None, plot=True):
    algorithm_performance_by_radius = {}
    radius_range = np.arange(1.0, util.euclidean_dist_2D((0.0, 0.0), (lattice_X, lattice_Y), 1.0))
    if interaction_shape == 'step_fn':
        interaction_fn = experiment.step_fn
    elif interaction_shape == 'power_decay_fn':
        interaction_fn = experiment.power_decay_fn
    elif interaction_shape == 'random':
        interaction_fn = experiment.random
        radius_range = np.zeros(1)
    if ensemble:
        radius_range = np.ones(1)
    for radius in radius_range:
        path += '/radius_{}'.format(radius)
        if plot and interaction_shape != 'random':
            # plot of radius vs. interaction strength
            util.plot_interaction(interaction_fn(radius), radius, lattice_X, lattice_Y, interaction_shape, path)
        exact_best_energy = None
        if brute_force_solution:
            exact_best_energy = brute_force_solution[radius]
        algorithm_performance_by_radius[radius] = search_params(algorithm_name, interaction_fn, radius, lattice_X, lattice_Y, interaction_shape, ensemble, exact_best_energy, fill, triangular, path, brute_force_path)
    if plot and interaction_shape != 'random':
        util.plot_runtimes_steps_vs_radius(algorithm_performance_by_radius, lattice_X, lattice_Y, interaction_shape, path)
    return algorithm_performance_by_radius

def get_exact_solutions(lattice_X, lattice_Y, interaction_shape, fill, triangular, path):
    radius_range = np.arange(1.0, util.euclidean_dist_2D((0.0, 0.0), (lattice_X, lattice_Y), 1.0))
    if interaction_shape == 'step_fn':
        interaction_fn = experiment.step_fn
    elif interaction_shape == 'power_decay_fn':
        interaction_fn = experiment.power_decay_fn
    elif interaction_shape == 'random':
        interaction_fn = experiment.random
        radius_range = np.zeros(1)
    w1 = csv.writer(open('{}/optimal_sols.csv'.format(path), 'w'))
    w1.writerow(['radius', 'min energy', '# ground states', '# states'])
    for radius in radius_range:
        prob = prob = initialize_problem(interaction_fn, radius, lattice_X, lattice_Y, fill=fill, triangular=triangular)
        exact_best_energy, num_best_partitions, sample_best_partition, total_num_partitions, all_partition_energies = algorithms.BruteForce().solve(prob)
        util.plot_spin_lattice([sample_best_partition], [exact_best_energy], radius, lattice_X, lattice_Y, interaction_shape, triangular, path)
        w1.writerow([radius, exact_best_energy, num_best_partitions, total_num_partitions])
        w2 = csv.writer(open('{}/energies_radius_{}.csv'.format(path, radius), 'w'))
        w2.writerow(['energy'])
        for energy in all_partition_energies:
            w2.writerow([energy])

def simulate(structure, fill, interaction_shape, ensemble, plot=True):
    triangular = True if structure == 'triangular' else False
    title = '{}_{}_{}'.format(structure, interaction_shape, fill)
    if ensemble:
        title += '_ensemble'
    try:
        os.mkdir(title)
    except FileExistsError:
        print('Directory {} already exists'.format(title))
    algorithm_performance_by_system = {}
    num_ground_states_by_system = {}
    for system_size in range(3, 6):
        lattice_X = system_size
        lattice_Y = system_size
        exact_min_energy_by_radius = None
        exact_sols_dir = None
        if system_size < 5:
            # brute force
            exact_sols_dir = '{}/exact_sols_{}x{}'.format(title, lattice_X, lattice_Y)
            util.make_dir(exact_sols_dir)
            get_exact_solutions(lattice_X, lattice_Y, interaction_shape, fill, triangular, exact_sols_dir)
            exact_min_energy_by_radius = {}
            num_ground_states_by_radius = {}
            r = csv.reader(open('{}/optimal_sols.csv'.format(exact_sols_dir), 'r'))
            next(r)
            for row in r:
                radius = float(row[0])
                exact_min_energy_by_radius[radius] = float(row[1])
                num_ground_states_by_radius[radius] = float(row[2])
            num_ground_states_by_system[system_size] = num_ground_states_by_radius
        # simulated annealing
        algo_sols_dir = '{}/{}x{}'.format(title, lattice_X, lattice_Y)
        util.make_dir(algo_sols_dir)
        algorithm_performance_by_radius = algorithm_performance('simulated annealing', lattice_X, lattice_Y, interaction_shape, fill, triangular, ensemble, algo_sols_dir, brute_force_solution=exact_min_energy_by_radius, brute_force_path=exact_sols_dir)
        algorithm_performance_by_system[system_size] = algorithm_performance_by_radius
    algo_summary_dir = '{}/summary'.format(title)
    util.make_dir(algo_summary_dir)
    util.plot_runtimes_steps_vs_system_size(algorithm_performance_by_system, interaction_shape, algo_summary_dir)
    if num_ground_states_by_system:
        exact_summary_dir = '{}/exact_sols_summary'.format(title)
        util.make_dir(exact_summary_dir)
        util.plot_num_ground_states_vs_system_size(num_ground_states_by_system, interaction_shape, exact_summary_dir)

def main(structure, fill, interaction_shape, ensemble, plot=True):
    if structure == 'free':
        maxcut_free.simulate(structure, interaction_shape, ensemble)
    else:
        simulate(structure, fill, interaction_shape, ensemble)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--structure", default='square', help="System Structure (options: square, triangular, free)")
    parser.add_argument("-f", "--fill", default=1.0, help="Fill (options: float in range [0, 1])", type=float)
    parser.add_argument("-i", "--interaction", default='power_decay_fn', help="Interaction Shape (options: step_fn, power_decay_fn, random)")
    parser.add_argument("-e", "--ensemble", default=False, help="Ensemble (options: True, False)", type=bool)
    args = parser.parse_args()
    structure = args.structure
    fill = args.fill
    interaction_shape = args.interaction
    ensemble = args.ensemble
    main(structure, fill, interaction_shape, ensemble)
            
