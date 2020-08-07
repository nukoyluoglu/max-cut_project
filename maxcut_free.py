import sys
import os
import experiment
import problem
import classical_algorithms
import timeit
import time
import util
import collections
import numpy as np
import copy
import csv

LATTICE_SPACING = 1
NUM_PARAM_TRIALS = 1000

def initialize_problem(interaction_fn, num_particles):
    setup = experiment.FreeSpins(num_particles)
    setup.turn_on_interactions(interaction_fn())
    return problem.MaxCutProblem(setup)

def run_trials_for_param(algorithm, interaction_fn, num_particles, exact_best_energy, min_energy_gap, init_temp, cool_rate, interaction_shape, path):
    ave_runtime = 0
    ave_step = 0
    min_best_energy = 0
    temp_hists = []
    energy_hists = []
    partition_hists = []
    ground_states_found = []
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
        best_energy = prob.get_best_energy()
        energy_hist = prob.get_energy_history()
        ground_state_found = float('inf')
        if abs(best_energy - exact_best_energy) <= min_energy_gap:
            ground_state_found = energy_hist.index(best_energy)
        elif best_energy - exact_best_energy < - min_energy_gap:
            print('WARNING: Cannot reach energy below ground state energy')
        temp_hist = algorithm.get_temp_history()
        partition_hist = prob.get_partition_history()
        ave_runtime += runtime
        ave_step += step
        if best_energy < min_best_energy:
            min_best_energy = best_energy
            sample_best_prob = prob
        if len(temp_hist) != len(energy_hist):
            raise RuntimeError('Length of temperature and energy histories must match')
        temp_hists.append(temp_hist)
        energy_hists.append(energy_hist)
        ground_states_found.append(ground_state_found)
        partition_hists.append(copy.copy(partition_hist))
    longest_temps = max(temp_hists, key=lambda hist: len(hist))
    longest_run = len(longest_temps)
    ave_energy_vs_temp = {}
    ave_energy_hist = {}
    ave_temp_hist = {}
    entropy_vs_temp = {}
    entropy_hist = {}
    temp_hist = longest_temps
    for i in range(NUM_PARAM_TRIALS):
        energy_hist = energy_hists[i]
        extend_iter = longest_run - len(energy_hist)
        energy_extend = np.mean(energy_hist[-1000:])
        energy_hist.extend([energy_extend for _ in range(extend_iter)])
        ground_state_found = ground_states_found[i]
        partition_hist = [list(d.values()) for d in partition_hists[i]]
        partition_hist.extend([partition_hist[-1] for _ in range(extend_iter)])
        for t in range(len(temp_hist)):
            temp = temp_hist[t]
            ave_energy_vs_temp.setdefault(temp, []).append(energy_hist[t])
            if exact_best_energy:
                prob_ground_state_vs_temp.setdefault(temp, []).append(float(t >= ground_state_found))
            ave_energy_hist.setdefault(t, []).append(energy_hist[t])
            ave_temp_hist.setdefault(t, []).append(temp)
            if exact_best_energy:
                prob_ground_state_hist.setdefault(t, []).append(float(t >= ground_state_found))
            entropy_vs_temp.setdefault(temp, []).append(partition_hist[t])
            entropy_hist.setdefault(t, []).append(partition_hist[t])
    for temp in ave_energy_vs_temp.keys():
        energies_at_temp = ave_energy_vs_temp[temp]
        ave_energy_vs_temp[temp] = (np.mean(energies_at_temp), np.divide(np.std(energies_at_temp), np.sqrt(len(energies_at_temp))))
        if exact_best_energy:
            ground_states_at_temp = prob_ground_state_vs_temp[temp]
            prob_ground_state_vs_temp[temp] = (np.mean(ground_states_at_temp), np.divide(np.std(ground_states_at_temp), np.sqrt(len(ground_states_at_temp))))
        u, counts_partitions_at_temp = np.unique(entropy_vs_temp[temp], return_counts=True, axis=0)
        total_partitions_at_temp = np.sum(counts_partitions_at_temp)
        probs_partitions_at_temp = counts_partitions_at_temp / total_partitions_at_temp
        entropy_vs_temp[temp] = np.sum(np.multiply(- probs_partitions_at_temp, np.log(probs_partitions_at_temp)))
    for t in ave_energy_hist.keys():
        energies_at_t = ave_energy_hist[t]
        ave_energy_hist[t] = (np.mean(energies_at_t), np.divide(np.std(energies_at_t), np.sqrt(len(energies_at_t))))
        temps_at_t = ave_temp_hist[t]
        ave_temp_hist[t] = np.mean(temps_at_t)
        if exact_best_energy:
            ground_states_found_at_t = prob_ground_state_hist[t]
            prob_ground_state_hist[t] = (np.mean(ground_states_found_at_t), np.divide(np.std(ground_states_found_at_t), np.sqrt(len(ground_states_found_at_t))))
        u, counts_partitions_at_t = np.unique(entropy_hist[t], return_counts=True, axis=0)
        total_partitions_at_t = np.sum(counts_partitions_at_t)
        probs_partitions_at_t = counts_partitions_at_t / total_partitions_at_t
        entropy_hist[t] = np.sum(np.multiply(- probs_partitions_at_t, np.log(probs_partitions_at_t)))
    ave_step /= NUM_PARAM_TRIALS
    ave_runtime /= NUM_PARAM_TRIALS
    step = ave_step
    runtime = ave_runtime
    # plot of energy and temperature vs. time
    util.plot_energy_temp_vs_step(ave_energy_hist, ave_temp_hist, 'NA', num_particles, interaction_shape, init_temp, cool_rate, path, exact_best_energy)
    # plot of energy, entropy, and derivative of entropy w.r.t. energy vs. inverse temperature
    util.plot_energy_entropy_vs_temp(ave_energy_vs_temp, entropy_vs_temp, 'NA', num_particles, interaction_shape, init_temp, cool_rate, path, exact_best_energy)
    P_optimal = 1.0 - 1.0 / np.exp(1)
    if exact_best_energy:
        total_iter = {}
        optimize_t = {}
        for t, (prob_ground_state, error) in prob_ground_state_hist.items():
            if t == 0: continue
            if prob_ground_state < P_optimal:
                total_iter[t] = np.divide(t, np.abs(np.log(1.0 - prob_ground_state))) # absolute value to avoid sign change due to dropping P_* term
            else: 
                total_iter[t] = t
            optimize_t[t] = total_iter[t]
        optimal_t = min(optimize_t, key=optimize_t.get)
        step = total_iter[optimal_t]
        # plot of T / |log(1 - P_T)| vs. T
        util.plot_step_optimization(total_iter, ave_temp_hist, optimal_t, step, 'NA', num_particles, interaction_shape, init_temp, cool_rate, path)
        # plot of probability of reaching ground state and temperature vs. time
        util.plot_prob_ground_state_temp_vs_step(prob_ground_state_hist, ave_temp_hist, optimal_t, step, 'NA', num_particles, interaction_shape, init_temp, cool_rate, path)
    return {'best_energy': min_best_energy, 'prob_ground_state': prob_ground_state_hist[optimal_t][0], 'step': step, 'runtime': runtime, 'ave_energy_vs_temp': ave_energy_vs_temp, 'sample_best_prob': sample_best_prob}

def search_params(algorithm_name, interaction_fn, num_particles, interaction_shape, ensemble, exact_best_energy, min_energy_gap, path, exact_path, plot=True):
    if algorithm_name == 'simulated annealing':
        algorithm = classical_algorithms.SimulatedAnnealing()
        # init_temps = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        # cool_rates = np.array([0.9979, 0.9984, 0.9989, 0.9994, 0.9999])
        init_temps = np.array([0.1, 1.0, 10.0])
        cool_rates = np.array([0.9979, 0.9989, 0.9999])
        if ensemble:
            algorithm = classical_algorithms.SimulatedAnnealingEnsemble()
            init_temps = np.array([10.0, 50.0, 100.0])
        best_solution = {}
        best_params = {'init_temp': None, 'cool_rate': None}
        ave_energy_vs_temp_by_params = collections.defaultdict(dict)
        for i in range(len(init_temps)):
            for j in range(len(cool_rates)):
                param_solution = run_trials_for_param(algorithm, interaction_fn, num_particles, exact_best_energy, min_energy_gap, init_temps[i], cool_rates[j], interaction_shape, path)
                ave_energy_vs_temp_by_params[init_temps[i]][cool_rates[j]] = param_solution['ave_energy_vs_temp']
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
            boltzmann_temps, boltzmann_energies = util.plot_params_energy_vs_temp_heatmap(ave_energy_vs_temp_by_params, best_params, 'NA', num_particles, interaction_shape, path, exact_best_energy, exact_path)
            util.plot_params_energy_vs_temp(ave_energy_vs_temp_by_params, best_params, 'NA', num_particles, interaction_shape, path, exact_best_energy, boltzmann_temps, boltzmann_energies, exact_path)
        return best_solution, best_params

def algorithm_performance(algorithm_name, num_particles, interaction_shape, ensemble, path, exact_best_energy=None, min_energy_gap=None, brute_force_path=None, plot=True):
    return search_params(algorithm_name, experiment.random, num_particles, interaction_shape, ensemble, exact_best_energy, min_energy_gap, path, brute_force_path)

def get_exact_solutions(num_particles, interaction_shape, path):
    interaction_fn = experiment.random
    w1 = csv.writer(open('{}/optimal_sols.csv'.format(path), 'w'))
    w1.writerow(['min energy', '# ground states', '# states'])
    prob = prob = initialize_problem(interaction_fn, num_particles)
    exact_best_energy, num_best_partitions, sample_best_partition, total_num_partitions, all_partition_energies = classical_algorithms.BruteForce().solve(prob)
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
    optimal_t_random_select_by_system = {}
    w0 = csv.writer(open('{}/runtimes_25_params_100_iters.csv'.format(title), 'w'))
    w0.writerow(['system size', 'runtime (s)'])
    # for system_size in range(4, 26):
    for system_size in range(9, 17):
        start = time.time()
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
            num_total_states = float(row[2])
        num_ground_states_by_system[system_size] = num_ground_states
        prob_ground_state_random = num_ground_states / num_total_states
        optimal_t_random = 1 / np.abs(np.log(1 - prob_ground_state_random))
        optimal_t_random_select_by_system[system_size] = optimal_t_random
        # simulated annealing
        algo_sols_dir = '{}/{}'.format(title, system_size)
        util.make_dir(algo_sols_dir)
        algorithm_performance_by_system[system_size] = algorithm_performance('simulated annealing', system_size, interaction_shape, ensemble, algo_sols_dir, exact_best_energy=exact_min_energy, min_energy_gap=min_gap, brute_force_path=exact_sols_dir)
        end = time.time()
        w0.writerow([system_size, end - start])
    algo_summary_dir = '{}/summary'.format(title)
    util.make_dir(algo_summary_dir)
    util.plot_runtimes_steps_vs_system_size(algorithm_performance_by_system, interaction_shape, algo_summary_dir, steps_random_select_by_system=optimal_t_random_select_by_system)
    if num_ground_states_by_system:
        exact_summary_dir = '{}/exact_sols_summary'.format(title)
        util.make_dir(exact_summary_dir)
        util.plot_num_ground_states_vs_system_size(num_ground_states_by_system, interaction_shape, exact_summary_dir)
            