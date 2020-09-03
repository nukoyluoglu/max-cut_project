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
import numpy as np
import copy
import csv
import argparse
import multiprocessing as mp

def exact_solve_prob(prob): 
    start = time.time()
    min_energy, num_ground_states, sample_ground_state, num_total_states, all_energies = classical_algorithms.BruteForce().solve(prob)
    min_gap = float('inf')
    prev_energy = 0.0
    for energy in sorted(all_energies):
        if energy != prev_energy:
            min_gap = min(abs(energy - prev_energy), min_gap)
            prev_energy = energy
    end = time.time()
    return {'min energy': min_energy, 'min energy gap': min_gap, 'runtime': end - start}

def exact_solve(structure, system_size, fill, interaction_shape, interaction_radius, path): 
    radius_dir_path = '{}/radius_{}'.format(path, interaction_radius)
    util.make_dir(radius_dir_path)
    start = time.time()
    prob = maxcut.initialize_problem(structure, system_size, fill, interaction_shape, interaction_radius)
    min_energy, num_ground_states, sample_ground_state, num_total_states, all_energies = classical_algorithms.BruteForce().solve(prob)
    if structure != 'free':
        with open('{}/ground_state_partitions.csv'.format(radius_dir_path), 'w') as output_file:
            header = sample_ground_state.keys()
            dict_writer = csv.DictWriter(output_file, header)
            dict_writer.writeheader()
            dict_writer.writerow(sample_ground_state)
    with open('{}/energies.csv'.format(radius_dir_path), 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['energy'])
        min_gap = float('inf')
        prev_energy = 0.0
        for energy in sorted(all_energies):
            writer.writerow([energy])
            if energy != prev_energy:
                min_gap = min(abs(energy - prev_energy), min_gap)
                prev_energy = energy
    end = time.time()
    return {'radius': interaction_radius, 'min energy': min_energy, '# ground states': num_ground_states, '# states': num_total_states, 'min energy gap': min_gap, 'runtime': end - start}

if __name__ == '__main__':
    structure, system_size, fill, interaction_shape, ensemble = maxcut.configure()
    
    parent_dir_path = '{}_{}'.format(structure, interaction_shape)
    util.make_dir(parent_dir_path)
    exact_dir_path = '{}/exact_sols_{}'.format(parent_dir_path, system_size)
    util.make_dir(exact_dir_path)

    if interaction_shape != 'random':
        assert isinstance(system_size, (list, tuple, np.ndarray))
        radius_range = np.arange(1.0, util.euclidean_dist_2D((0.0, 0.0), (system_size[0] * experiment.LATTICE_SPACING, system_size[1] * experiment.LATTICE_SPACING)), experiment.LATTICE_SPACING)
    else:
        radius_range = ['NA']

    exact_sols = [exact_solve(structure, system_size, fill, interaction_shape, interaction_radius, exact_dir_path) for interaction_radius in radius_range]
    
    with open('{}/exact_sols.csv'.format(exact_dir_path), 'w') as output_file:
        header = exact_sols[0].keys()
        dict_writer = csv.DictWriter(output_file, header)
        dict_writer.writeheader()
        dict_writer.writerows(sorted(exact_sols, key=lambda d: d['radius']))