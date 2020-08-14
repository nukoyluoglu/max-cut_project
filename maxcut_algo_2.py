import sys
import os
import maxcut
import experiment
import problem
import classical_algorithms
import timeit
import time
import util
import collections
import itertools
import numpy as np
import copy
import csv
import argparse
import multiprocessing as mp

NUM_PARAM_TRIALS = 1000

def collect_param_stats(temp_hists, energy_hists, ground_states_found, partition_hists, exact_min_energy):
    all_temps_hist = {}
    all_energies_hist = {}
    all_partitions_hist = {}
    all_ground_states_found_hist = None
    all_energies_vs_temp = {}
    all_partitions_vs_temp = {}
    all_ground_states_found_vs_temp = None
    if exact_min_energy:
        all_ground_states_found_hist = {}
        all_ground_states_found_vs_temp = {}
    longest_temps = max(temp_hists, key=lambda hist: len(hist))
    longest_run = len(longest_temps)
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
            all_energies_vs_temp.setdefault(temp, []).append(energy_hist[t])
            if exact_min_energy:
                all_ground_states_found_vs_temp.setdefault(temp, []).append(float(t >= ground_state_found))
            all_energies_hist.setdefault(t, []).append(energy_hist[t])
            all_temps_hist.setdefault(t, []).append(temp)
            if exact_min_energy:
                all_ground_states_found_hist.setdefault(t, []).append(float(t >= ground_state_found))
            all_partitions_vs_temp.setdefault(temp, []).append(partition_hist[t])
            all_partitions_hist.setdefault(t, []).append(partition_hist[t])
    return all_temps_hist, all_energies_hist, all_partitions_hist, all_ground_states_found_hist, all_energies_vs_temp, all_partitions_vs_temp, all_ground_states_found_vs_temp

def get_param_stats_per_temp(temp, all_energies_vs_temp, all_partitions_vs_temp, all_ground_states_found_vs_temp):
    stat_vs_temp = {'temp': temp}
    energies_at_temp = all_energies_vs_temp[temp]
    stat_vs_temp['ave_energy'] = np.mean(energies_at_temp)
    stat_vs_temp['err_energy'] = np.divide(np.std(energies_at_temp), np.sqrt(len(energies_at_temp)))
    u, counts_partitions_at_temp = np.unique(all_partitions_vs_temp[temp], return_counts=True, axis=0)
    total_partitions_at_temp = np.sum(counts_partitions_at_temp)
    probs_partitions_at_temp = counts_partitions_at_temp / total_partitions_at_temp
    stat_vs_temp['entropy'] = np.sum(np.multiply(- probs_partitions_at_temp, np.log(probs_partitions_at_temp)))
    stat_vs_temp['ave_prob_ground_state'] = None
    stat_vs_temp['err_prob_ground_state'] = None
    if all_ground_states_found_vs_temp:
        probs_ground_state_at_temp = all_ground_states_found_vs_temp[temp]
        stat_vs_temp['ave_prob_ground_state'] = np.mean(probs_ground_state_at_temp)
        stat_vs_temp['err_prob_ground_state'] = np.divide(np.std(probs_ground_state_at_temp), np.sqrt(len(probs_ground_state_at_temp)))
    return stat_vs_temp

def get_param_stats_per_t(t, all_temps_hist, all_energies_hist, all_partitions_hist, all_ground_states_found_hist):
    stat_vs_t = {'t': t}
    temps_at_t = all_temps_hist[t]
    stat_vs_t['ave_temp'] = np.mean(temps_at_t)
    energies_at_t = all_energies_hist[t]
    stat_vs_t['ave_energy'] = np.mean(energies_at_t)
    stat_vs_t['err_energy'] = np.divide(np.std(energies_at_t), np.sqrt(len(energies_at_t)))
    u, counts_partitions_at_t = np.unique(all_partitions_hist[t], return_counts=True, axis=0)
    total_partitions_at_t = np.sum(counts_partitions_at_t)
    probs_partitions_at_t = counts_partitions_at_t / total_partitions_at_t
    stat_vs_t['entropy'] = np.sum(np.multiply(- probs_partitions_at_t, np.log(probs_partitions_at_t)))
    return stat_vs_t

def get_total_iters(stats_vs_t, energy_hists, all_ground_states_found_hist, exact_min_energy, entropy_approx=True):
    all_ground_states_found_hist_from_entropy = None
    min_energy_from_entropy = None
    if interaction_shape == 'random' and not entropy_approx: # 2 ground states, ground state entropy = ln(2)
        t_converge = [stat_vs_t['t'] for stat_vs_t in stats_vs_t if np.round(stat_vs_t['entropy'], 8) <= np.round(np.log(2), 8)]
        if len(t_converge) > 0:
            min_energy_from_entropy = min([stat_vs_t['ave_energy'] for stat_vs_t in stats_vs_t if stat_vs_t['t'] in t_converge])
            all_ground_states_found_hist_from_entropy = {}
            for energy_hist in energy_hists:
                ground_state_found = False
                for t in range(len(energy_hist)):
                    if np.round(energy_hist[t], 10) == np.round(min_energy_from_entropy, 10):
                        ground_state_found = True
                    all_ground_states_found_hist_from_entropy.setdefault(t, []).append(float(ground_state_found))
    for stat_vs_t in stats_vs_t:
        stat_vs_t['ave_prob_ground_state'] = None
        stat_vs_t['err_prob_ground_state'] = None
        stat_vs_t['total_iter'] = None
        stat_vs_t['ave_prob_ground_state_from_entropy'] = None
        stat_vs_t['err_prob_ground_state_from_entropy'] = None
        stat_vs_t['total_iter_from_entropy'] = None
        t = stat_vs_t['t']
        if exact_min_energy and all_ground_states_found_hist:
            probs_ground_state_at_t = all_ground_states_found_hist[t]
            stat_vs_t['ave_prob_ground_state'] = np.mean(probs_ground_state_at_t)
            stat_vs_t['err_prob_ground_state'] = np.divide(np.std(probs_ground_state_at_t), np.sqrt(len(probs_ground_state_at_t)))
            # compute M * T
            P_opt = 1.0 - 1.0 / np.exp(1)
            P = stat_vs_t['ave_prob_ground_state']
            if P < P_opt:
                stat_vs_t['total_iter'] = np.divide(t, np.abs(np.log(1.0 - P))) # absolute value to avoid sign change due to dropping P_* term
            else: 
                stat_vs_t['total_iter'] = t
        if all_ground_states_found_hist_from_entropy:
            probs_ground_state_at_t = all_ground_states_found_hist_from_entropy[t]
            stat_vs_t['ave_prob_ground_state_from_entropy'] = np.mean(probs_ground_state_at_t)
            stat_vs_t['err_prob_ground_state_from_entropy'] = np.divide(np.std(probs_ground_state_at_t), np.sqrt(len(probs_ground_state_at_t)))
            # compute M * T
            P_opt = 1.0 - 1.0 / np.exp(1)
            P = stat_vs_t['ave_prob_ground_state_from_entropy']
            if P < P_opt:
                stat_vs_t['total_iter_from_entropy'] = np.divide(t, np.abs(np.log(1.0 - P))) # absolute value to avoid sign change due to dropping P_* term
            else: 
                stat_vs_t['total_iter_from_entropy'] = t
        else:
            # compute M * T using entropy as heuristic: S = E[- log P] = - log P
            P_opt = 1.0 - 1.0 / np.exp(1)
            P = np.exp(- stat_vs_t['entropy'])
            if P < P_opt:
                stat_vs_t['total_iter_from_entropy'] = np.divide(t, np.abs(np.log(1.0 - P))) # absolute value to avoid sign change due to dropping P_* term
            else: 
                stat_vs_t['total_iter_from_entropy'] = t

    return min_energy_from_entropy

def simulate(structure, system_size, fill, interaction_shape, interaction_radius, algorithm, init_temp, cool_rate, exact_min_energy, exact_min_gap):
    prob = maxcut.initialize_problem(structure, system_size, fill, interaction_shape, interaction_radius)
    algorithm.set_cooling_schedule(init_temp, cool_rate)
    algorithm.solve(prob)
    step = len(algorithm.get_temp_history())
    min_energy = prob.get_best_energy()
    temp_hist = algorithm.get_temp_history()
    energy_hist = prob.get_energy_history()
    partition_hist = prob.get_partition_history()
    if len(temp_hist) != len(energy_hist):
        raise RuntimeError('Length of temperature and energy histories must match')
    ground_state_found = None
    if exact_min_energy:
        ground_state_found = float('inf')
        if abs(min_energy - exact_min_energy) <= exact_min_gap:
            ground_state_found = np.min(np.where(np.round(energy_hist, 10) == (np.round(min_energy, 10))))
        elif min_energy - exact_min_energy < - exact_min_gap:
            print('WARNING: Cannot reach energy below ground state energy')
    return [step, temp_hist, energy_hist, partition_hist, ground_state_found, min_energy, prob]

def run_trials(structure, system_size, fill, interaction_shape, interaction_radius, algorithm, ensemble, path, exact_min_energy, exact_min_gap, init_temp, cool_rate, sample_best_probs):
    param_sols = []
    for i in range(NUM_PARAM_TRIALS):
        param_sols.append(simulate(structure, system_size, fill, interaction_shape, interaction_radius, algorithm, init_temp, cool_rate, exact_min_energy, exact_min_gap))

    steps, temp_hists, energy_hists, partition_hists, ground_states_found, min_energies, probs = np.array(param_sols).T
    min_energy = min(min_energies)
    sample_best_probs[(init_temp, cool_rate)] = probs[np.argmin(min_energies)]
    
    all_temps_hist, all_energies_hist, all_partitions_hist, all_ground_states_found_hist, all_energies_vs_temp, all_partitions_vs_temp, all_ground_states_found_vs_temp = collect_param_stats(temp_hists, energy_hists, ground_states_found, partition_hists, exact_min_energy)

    stats_vs_temp = []
    for temp in all_energies_vs_temp.keys():
        stats_vs_temp.append(get_param_stats_per_temp(temp, all_energies_vs_temp, all_partitions_vs_temp, all_ground_states_found_vs_temp))
    
    stats_vs_t = []
    for t in all_energies_hist.keys():
        stats_vs_t.append(get_param_stats_per_t(t, all_temps_hist, all_energies_hist, all_partitions_hist, all_ground_states_found_hist))

    min_energy_from_entropy = get_total_iters(stats_vs_t, energy_hists, all_ground_states_found_hist, exact_min_energy)

    step_ave = np.mean(steps)
    step_from_exact = None
    step_from_entropy = None
    prob_ground_state_per_run = None
    t_opt_exact = None
    
    MT_exact = {}
    MT_from_entropy = {}
    for stat_vs_t in stats_vs_t:
        t = stat_vs_t['t']
        if t == 0: continue
        if stat_vs_t['total_iter']:
            MT_exact[t] = stat_vs_t['total_iter']
        if stat_vs_t['total_iter_from_entropy']:
            MT_from_entropy[t] = stat_vs_t['total_iter_from_entropy']
    if len(MT_exact) > 0:
        t_opt_exact = min(MT_exact, key=MT_exact.get)
        step_from_exact = MT_exact[t_opt_exact]
        stat_vs_t_opt_exact = [stat_vs_t for stat_vs_t in stats_vs_t if stat_vs_t['t'] == t_opt_exact][0]
        prob_ground_state_per_run = stat_vs_t_opt_exact['ave_prob_ground_state']
    if len(MT_from_entropy) > 0:
        t_opt_from_entropy = min(MT_from_entropy, key=MT_from_entropy.get)
        step_from_entropy = MT_from_entropy[t_opt_from_entropy]

    with open('{}/stats_vs_temp_T_0_{}_r_{}.csv'.format(path, init_temp, cool_rate), 'w') as output_file:
        header = stats_vs_temp[0].keys()
        dict_writer = csv.DictWriter(output_file, header)
        dict_writer.writeheader()
        dict_writer.writerows(stats_vs_temp)
    
    with open('{}/stats_vs_t_T_0_{}_r_{}.csv'.format(path, init_temp, cool_rate), 'w') as output_file:
        header = stats_vs_t[0].keys()
        dict_writer = csv.DictWriter(output_file, header)
        dict_writer.writeheader()
        dict_writer.writerows(stats_vs_t)

    return dict(init_temp=init_temp, cool_rate=cool_rate, min_energy=min_energy, min_energy_from_entropy=min_energy_from_entropy, step_from_exact=step_from_exact, step_from_entropy=step_from_entropy, step_per_run=t_opt_exact ,prob_ground_state_per_run=prob_ground_state_per_run)

def param_search(structure, system_size, fill, interaction_shape, interaction_radius, ensemble, path, exact_min_energy, exact_min_gap):
    radius_dir_path = '{}/radius_{}'.format(path, interaction_radius)
    util.make_dir(radius_dir_path)
    start = time.time()
    algorithm = classical_algorithms.SimulatedAnnealing()
    cooling_schedules = maxcut.get_cooling_schedules(problem=maxcut.initialize_problem(structure, system_size, fill, interaction_shape, interaction_radius))
    radius_sols = []
    sample_best_probs = {}
    for init_temp, cool_rate in cooling_schedules:
        radius_sols.append(run_trials(structure, system_size, fill, interaction_shape, interaction_radius, algorithm, ensemble, radius_dir_path, exact_min_energy, exact_min_gap, init_temp, cool_rate, sample_best_probs))

    with open('{}/param_results.csv'.format(radius_dir_path), 'w') as output_file:
        header = radius_sols[0].keys()
        dict_writer = csv.DictWriter(output_file, header)
        dict_writer.writeheader()
        dict_writer.writerows(radius_sols)

    opt_sol = min(radius_sols, key=lambda sol: sol['step_from_exact'])
    opt_init_temp = opt_sol['init_temp']
    opt_cool_rate = opt_sol['cool_rate']
    opt_step_from_exact = opt_sol['step_from_exact']
    opt_step_from_entropy = opt_sol['step_from_entropy']
    opt_prob_ground_state_per_run = opt_sol['prob_ground_state_per_run']
    
    if structure != 'free':
        sample_best_prob = sample_best_probs[(opt_init_temp, opt_cool_rate)]
        sample_best_partition_hist = sample_best_prob.get_partition_history()
        sample_best_energy_hist = sample_best_prob.get_energy_history()
        with open('{}/ground_state_partitions.csv'.format(radius_dir_path), 'w') as output_file:
            header = sample_best_partition_hist[0].keys()
            dict_writer = csv.DictWriter(output_file, header)
            dict_writer.writeheader()
            dict_writer.writerows(sample_best_partition_hist)
        with open('{}/ground_state_energies.csv'.format(radius_dir_path), 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['energy'])
            for energy in sample_best_energy_hist:
                writer.writerow([energy])
    end = time.time()
    return dict(interaction_radius=interaction_radius, init_temp=opt_init_temp, cool_rate=opt_cool_rate, step_from_exact=opt_step_from_exact, step_from_entropy=opt_step_from_entropy, prob_ground_state_per_run=opt_prob_ground_state_per_run, search_runtime=(end - start), exact_min_energy=exact_min_energy)

if __name__ == '__main__':
    structure, system_size, fill, interaction_shape, ensemble = maxcut.configure()
    
    parent_dir_path = '{}_{}'.format(structure, interaction_shape)
    util.make_dir(parent_dir_path)
    algo_dir_path = '{}/algo_sols_{}'.format(parent_dir_path, system_size)
    if ensemble:
        algo_dir_path += '_ensemble'
    util.make_dir(algo_dir_path)
    exact_dir_path = '{}/exact_sols_{}'.format(parent_dir_path, system_size)

    exact_min_energy = {}
    exact_min_gap = {}
    if os.path.exists(exact_dir_path):
        with open('{}/exact_sols.csv'.format(exact_dir_path), 'r') as input_file:
            reader = csv.reader(input_file)
            next(reader)
            for row in reader:
                radius = 'NA' if row[0] == 'NA' else float(row[0])
                exact_min_energy[radius] = float(row[1])
                exact_min_gap[radius] = float(row[4])
    
    if interaction_shape != 'random':
        assert isinstance(system_size, (list, tuple, np.ndarray))
        radius_range = np.arange(1.0, util.euclidean_dist_2D((0.0, 0.0), (system_size[0] * experiment.LATTICE_SPACING, system_size[1] * experiment.LATTICE_SPACING)), experiment.LATTICE_SPACING)
    else:
        radius_range = ['NA']

    system_sols = [param_search(structure, system_size, fill, interaction_shape, interaction_radius, ensemble, algo_dir_path, exact_min_energy.get(interaction_radius), exact_min_gap.get(interaction_radius)) for interaction_radius in radius_range]
    
    with open('{}/system_sols.csv'.format(algo_dir_path), 'w') as output_file:
        header = system_sols[0].keys()
        dict_writer = csv.DictWriter(output_file, header)
        dict_writer.writeheader()
        dict_writer.writerows(sorted(system_sols, key=lambda d: d['interaction_radius']))