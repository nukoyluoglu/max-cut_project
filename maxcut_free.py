import sys
import os
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

def initialize_problem(interaction_fn, num_particles):
    setup = experiment.FreeSpins(num_particles)
    setup.turn_on_interactions(interaction_fn())
    return problem.MaxCutProblem(setup)

def run_trials_for_param(algorithm, interaction_fn, num_particles, exact_best_energy, min_energy_gap, init_temp, cool_rate, interaction_shape, path):
    ave_runtime = 0
    ave_step = 0
    min_best_energy = 0
    ave_energy_vs_temp = {}
    ave_energy_hist = {}
    ave_temp_hist = {}
    if exact_best_energy:
        prob_ground_state_hist = {}
        prob_ground_state_vs_temp = {}
    sample_best_prob = None
    if not min_energy_gap:
        min_energy_gap = 0.0
    for _ in range(NUM_PARAM_TRIALS):
        prob = initialize_problem(interaction_fn, num_particles)
        algorithm.set_cooling_schedule(init_temp, cool_rate)
        myGlobals = globals()
        myGlobals.update({'algorithm': algorithm, 'prob': prob})
        runtime = timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
        step = len(algorithm.get_temp_history())
        # debug each run
        # util.plot_temporary(prob.get_energy_history(), algorithm.get_temp_history(), init_temp, cool_rate, _, path, exact_best_energy)
        best_energy = prob.get_best_energy()
        energy_hist = prob.get_energy_history()
        ground_state_found = float('inf')
        if abs(best_energy - exact_best_energy) <= min_energy_gap:
            ground_state_found = energy_hist.index(best_energy)
        temp_hist = algorithm.get_temp_history()
        ave_runtime += runtime
        ave_step += step
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
                if exact_best_energy:
                    prob_ground_state_vs_temp.setdefault(temp, []).append(float(t >= ground_state_found))
            ave_energy_hist.setdefault(t, []).append(energy_hist[t])
            ave_temp_hist.setdefault(t, []).append(temp)
            if exact_best_energy:
                prob_ground_state_hist.setdefault(t, []).append(float(t >= ground_state_found))
    for temp in ave_energy_vs_temp.keys():
        energies_at_temp = ave_energy_vs_temp[temp]
        ave_energy_vs_temp[temp] = (np.mean(energies_at_temp), np.std(energies_at_temp))
        if exact_best_energy:
            ground_states_at_temp = prob_ground_state_vs_temp[temp]
            prob_ground_state_vs_temp[temp] = (np.mean(ground_states_at_temp), np.std(ground_states_at_temp))
    for t in ave_energy_hist.keys():
        energies_at_t = ave_energy_hist[t]
        ave_energy_hist[t] = np.mean(energies_at_t)
        temps_at_t = ave_temp_hist[t]
        ave_temp_hist[t] = np.mean(temps_at_t)
        if exact_best_energy:
            ground_states_found_at_t = prob_ground_state_hist[t]
            prob_ground_state_hist[t] = np.mean(ground_states_found_at_t)
    ave_step /= NUM_PARAM_TRIALS
    ave_runtime /= NUM_PARAM_TRIALS
    step = ave_step
    runtime = ave_runtime
    # plot of energy and temperature vs. time
    util.plot_energy_temp_vs_step(ave_energy_hist, ave_temp_hist, 'NA', num_particles, interaction_shape, init_temp, cool_rate, path, exact_best_energy)
    if exact_best_energy:
        optimize_step = {}
        to_optimize = {}
        for t, prob_ground_state in prob_ground_state_hist.items():
            if t == 0: continue
            optimize_step[t] = np.divide(t, np.abs(np.log(1.0 - prob_ground_state))) # absolute value to avoid sign dependence
            # if prob_ground_state < 1.0 and prob_ground_state > 0.5:
            if prob_ground_state > 0.9:
                to_optimize[t] = optimize_step[t]
        # optimal_step = min(optimize_step, key=optimize_step.get)
        step = float('inf')
        optimal_step = None
        if len(to_optimize) > 0:
            optimal_step = min(to_optimize, key=to_optimize.get)
            step = optimal_step
        # plot of T / log(1 - P_T) vs. T
        util.plot_step_optimization(optimize_step, ave_temp_hist, optimal_step, 'NA', num_particles, interaction_shape, init_temp, cool_rate, path)
        # plot of probability of reaching ground state and temperature vs. time
        util.plot_prob_ground_state_temp_vs_step(prob_ground_state_hist, ave_temp_hist, optimal_step, 'NA', num_particles, interaction_shape, init_temp, cool_rate, path)
    return {'best_energy': min_best_energy, 'step': step, 'runtime': runtime, 'ave_energy_vs_temp': ave_energy_vs_temp, 'prob_ground_state_vs_temp': prob_ground_state_vs_temp, 'sample_best_prob': sample_best_prob}

def search_params(algorithm_name, interaction_fn, num_particles, interaction_shape, ensemble, exact_best_energy, min_energy_gap, path, exact_path, plot=True):
    if algorithm_name == 'simulated annealing':
        algorithm = algorithms.SimulatedAnnealing()
        init_temps = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
        cool_rates = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        if ensemble:
            algorithm = algorithms.SimulatedAnnealingEnsemble()
            init_temps = np.array([10.0, 50.0, 100.0])
        best_solution = {}
        best_params = {'init_temp': None, 'cool_rate': None}
        ave_energy_vs_temp_by_params = collections.defaultdict(dict)
        prob_ground_state_vs_temp_by_params = collections.defaultdict(dict)
        for i in range(len(init_temps)):
            for j in range(len(cool_rates)):
                param_solution = run_trials_for_param(algorithm, interaction_fn, num_particles, exact_best_energy, min_energy_gap, init_temps[i], cool_rates[j], interaction_shape, path)
                ave_energy_vs_temp_by_params[init_temps[i]][cool_rates[j]] = param_solution['ave_energy_vs_temp']
                prob_ground_state_vs_temp_by_params[init_temps[i]][cool_rates[j]] = param_solution['prob_ground_state_vs_temp']
                if not best_solution or param_solution['best_energy'] < best_solution['best_energy'] or (param_solution['best_energy'] == best_solution['best_energy'] and param_solution['step'] < best_solution['step']):
                    best_solution = param_solution
                    best_params['init_temp'] = init_temps[i]
                    best_params['cool_rate'] = cool_rates[j]
                elif best_solution == param_solution:
                    raise RuntimeError('Best solution must be preserved throughout trials')
                if exact_best_energy and best_solution['best_energy'] < exact_best_energy - min_energy_gap:
                    print('exact best energy: {}'.format(exact_best_energy))
                    print('algo best energy: {}'.format(best_solution['best_energy']))
                    print('algo best partition: {}'.format(best_solution['sample_best_prob'].get_partition()))
        if plot:
            util.plot_params_energy_vs_temp(ave_energy_vs_temp_by_params, best_params, 'NA', num_particles, interaction_shape, path, exact_best_energy, exact_path)
            util.plot_params_energy_vs_temp_heatmap(ave_energy_vs_temp_by_params, best_params, 'NA', num_particles, interaction_shape, path, exact_best_energy, exact_path)
            if exact_best_energy:
                util.plot_params_prob_ground_state_vs_temp(prob_ground_state_vs_temp_by_params, best_params, 'NA', num_particles, interaction_shape, path, exact_best_energy, exact_path)
                util.plot_params_prob_ground_state_vs_temp_heatmap(prob_ground_state_vs_temp_by_params, best_params, 'NA', num_particles, interaction_shape, path, exact_best_energy, exact_path)
        return best_solution, best_params

def algorithm_performance(algorithm_name, num_particles, interaction_shape, ensemble, path, exact_best_energy=None, min_energy_gap=None, brute_force_path=None, plot=True):
    return search_params(algorithm_name, experiment.random, num_particles, interaction_shape, ensemble, exact_best_energy, min_energy_gap, path, brute_force_path)

def get_exact_solutions(num_particles, interaction_shape, path):
    interaction_fn = experiment.random
    w1 = csv.writer(open('{}/optimal_sols.csv'.format(path), 'w'))
    w1.writerow(['min energy', '# ground states', '# states'])
    prob = prob = initialize_problem(interaction_fn, num_particles)
    exact_best_energy, num_best_partitions, sample_best_partition, total_num_partitions, all_partition_energies = algorithms.BruteForce().solve(prob)
    w1.writerow([exact_best_energy, num_best_partitions, total_num_partitions])
    w2 = csv.writer(open('{}/energies_radius_NA.csv'.format(path), 'w'))
    w2.writerow(['energy'])
    min_gap = float('inf')
    prev_energy = 0.0
    for energy in sorted(all_partition_energies):
        w2.writerow([energy])
        if energy != prev_energy:
            min_gap = min(abs(energy - prev_energy), min_gap)
            prev_energy = energy
    return min_gap

def simulate(structure, interaction_shape, ensemble, plot=True):
    title = '{}_{}'.format(structure, interaction_shape)
    if ensemble:
        title += '_ensemble'
    try:
        os.mkdir(title)
    except FileExistsError:
        print('Directory {} already exists'.format(title))
    algorithm_performance_by_system = {}
    num_ground_states_by_system = {}
    for system_size in range(9, 17):
        exact_sols_dir = None
        # brute force
        exact_sols_dir = '{}/exact_sols_{}'.format(title, system_size)
        util.make_dir(exact_sols_dir)
        min_gap = get_exact_solutions(system_size, interaction_shape, exact_sols_dir)
        r = csv.reader(open('{}/optimal_sols.csv'.format(exact_sols_dir), 'r'))
        next(r)
        for row in r:
            exact_min_energy = float(row[0])
            num_ground_states = float(row[1])
        num_ground_states_by_system[system_size] = num_ground_states
        # simulated annealing
        algo_sols_dir = '{}/{}'.format(title, system_size)
        util.make_dir(algo_sols_dir)
        algorithm_performance_by_system[system_size] = algorithm_performance('simulated annealing', system_size, interaction_shape, ensemble, algo_sols_dir, exact_best_energy=exact_min_energy, min_energy_gap=min_gap, brute_force_path=exact_sols_dir)
    algo_summary_dir = '{}/summary'.format(title)
    util.make_dir(algo_summary_dir)
    util.plot_runtimes_steps_vs_system_size(algorithm_performance_by_system, interaction_shape, algo_summary_dir)
    if num_ground_states_by_system:
        exact_summary_dir = '{}/exact_sols_summary'.format(title)
        util.make_dir(exact_summary_dir)
        util.plot_num_ground_states_vs_system_size(num_ground_states_by_system, interaction_shape, exact_summary_dir)
            
