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
NUM_PERFORMANCE_TRIALS = 100
# NUM_PARAM_TRIALS = 50
NUM_PARAM_TRIALS = 100

def initialize_problem(interaction_fn, num_particles):
    setup = experiment.FreeSpins(num_particles)
    setup.turn_on_interactions(interaction_fn())
    return problem.MaxCutProblem(setup)

def run_trials_for_param(algorithm, interaction_fn, num_particles, init_temp, cool_rate, interaction_shape, path):
    ave_runtime = 0
    ave_step = 0
    min_best_energy = 0
    temp_hists = []
    energy_hists = []
    partition_hists = []
    sample_best_prob = None
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
        partition_hists.append(partition_hist)
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
        partition_hist = partition_hists[i]
        partition_hist.extend([partition_hist[-1] for _ in range(extend_iter)])
        for t in range(len(temp_hist)):
            temp = temp_hist[t]
            ave_energy_vs_temp.setdefault(temp, []).append(energy_hist[t])
            ave_energy_hist.setdefault(t, []).append(energy_hist[t])
            ave_temp_hist.setdefault(t, []).append(temp)
            entropy_vs_temp.setdefault(temp, []).append(partition_hist[t])
            entropy_hist.setdefault(t, []).append(partition_hist[t])
    for temp in ave_energy_vs_temp.keys():
        energies_at_temp = ave_energy_vs_temp[temp]
        ave_energy_vs_temp[temp] = (np.mean(energies_at_temp), np.divide(np.std(energies_at_temp), np.sqrt(len(energies_at_temp))))
        unique_partitions_at_temp, counts_partitions_at_temp = np.unique(entropy_vs_temp[temp], return_counts=True)
        total_partitions_at_temp = np.sum(counts_partitions_at_temp)
        probs_partitions_at_temp = counts_partitions_at_temp / total_partitions_at_temp
        entropy_vs_temp[temp] = np.sum(np.multiply(- probs_partitions_at_temp, np.log(probs_partitions_at_temp)))
    for t in ave_energy_hist.keys():
        energies_at_t = ave_energy_hist[t]
        ave_energy_hist[t] = (np.mean(energies_at_t), np.divide(np.std(energies_at_t), np.sqrt(len(energies_at_t))))
        temps_at_t = ave_temp_hist[t]
        ave_temp_hist[t] = np.mean(temps_at_t)
        unique_partitions_at_t, counts_partitions_at_t = np.unique(entropy_hist[t], return_counts=True)
        total_partitions_at_t = np.sum(counts_partitions_at_t)
        probs_partitions_at_t = counts_partitions_at_t / total_partitions_at_t
        entropy_hist[t] = np.sum(np.multiply(- probs_partitions_at_t, np.log(probs_partitions_at_t)))
    ave_step /= NUM_PARAM_TRIALS
    ave_runtime /= NUM_PARAM_TRIALS
    step = ave_step
    runtime = ave_runtime
    # plot of energy and temperature vs. time
    util.plot_energy_temp_vs_step(ave_energy_hist, ave_temp_hist, 'NA', num_particles, interaction_shape, init_temp, cool_rate, path, None)
    converged_t = np.where(entropy_hist == np.log(2))
    optimal_t = min(converged_t) if len(converged_t) > 0 else longest_run
    util.plot_entropy_temp_vs_step(entropy_hist, ave_temp_hist, optimal_t, step, 'NA', num_particles, interaction_shape, init_temp, cool_rate, path)
    step = optimal_t
    return {'best_energy': min_best_energy, 'entropy': entropy_hist[step], 'step': step, 'runtime': runtime, 'ave_energy_vs_temp': ave_energy_vs_temp, 'sample_best_prob': sample_best_prob}

def search_params(algorithm_name, interaction_fn, num_particles, interaction_shape, ensemble, path, plot=True):
    if algorithm_name == 'simulated annealing':
        algorithm = classical_algorithms.SimulatedAnnealing()
        init_temps = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        cool_rates = np.array([0.9975, 0.998, 0.9985, 0.999, 0.9995])
        if ensemble:
            algorithm = classical_algorithms.SimulatedAnnealingEnsemble()
            init_temps = np.array([10.0, 50.0, 100.0])
        best_solution = {}
        best_params = {'init_temp': None, 'cool_rate': None}
        ave_energy_vs_temp_by_params = collections.defaultdict(dict)
        for i in range(len(init_temps)):
            for j in range(len(cool_rates)):
                param_solution = run_trials_for_param(algorithm, interaction_fn, num_particles, init_temps[i], cool_rates[j], interaction_shape, path)
                ave_energy_vs_temp_by_params[init_temps[i]][cool_rates[j]] = param_solution['ave_energy_vs_temp']
                if not best_solution or param_solution['best_energy'] < best_solution['best_energy'] or (param_solution['best_energy'] == best_solution['best_energy'] and param_solution['step'] < best_solution['step']):
                    best_solution = param_solution
                    best_params['init_temp'] = init_temps[i]
                    best_params['cool_rate'] = cool_rates[j]
                elif best_solution == param_solution:
                    raise RuntimeError('Best solution must be preserved throughout trials')
        if plot:
            util.plot_params_energy_vs_temp(ave_energy_vs_temp_by_params, best_params, 'NA', num_particles, interaction_shape, path)
            util.plot_params_energy_vs_temp_heatmap(ave_energy_vs_temp_by_params, best_params, 'NA', num_particles, interaction_shape, path)
        return best_solution, best_params

def algorithm_performance(algorithm_name, num_particles, interaction_shape, ensemble, path, plot=True):
    return search_params(algorithm_name, experiment.random, num_particles, interaction_shape, ensemble, path)

def simulate(structure, interaction_shape, ensemble, plot=True):
    start = time.time()
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
        # simulated annealing
        algo_sols_dir = '{}/{}'.format(title, system_size)
        util.make_dir(algo_sols_dir)
        algorithm_performance_by_system[system_size] = algorithm_performance('simulated annealing', system_size, interaction_shape, ensemble, algo_sols_dir)
    algo_summary_dir = '{}/summary'.format(title)
    util.make_dir(algo_summary_dir)
    util.plot_runtimes_steps_vs_system_size(algorithm_performance_by_system, interaction_shape, algo_summary_dir)
    if num_ground_states_by_system:
        exact_summary_dir = '{}/exact_sols_summary'.format(title)
        util.make_dir(exact_summary_dir)
        util.plot_num_ground_states_vs_system_size(num_ground_states_by_system, interaction_shape, exact_summary_dir)
    end = time.time()
            
