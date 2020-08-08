import sys
import os
import experiment
import problem
import classical_algorithms
import maxcut_free
import timeit
import util
import collections
import numpy as np
import copy
import csv
import argparse
import itertools

LATTICE_SPACING = 1
NUM_PARAM_TRIALS = 50

def get_interaction_fn(interaction_shape):
    if interaction_shape == 'random':
        interaction_fn = experiment.random
    elif interaction_shape == 'step_fn':
        interaction_fn = experiment.step_fn
    elif interaction_shape == 'power_decay_fn':
        interaction_fn = experiment.power_decay_fn
    elif interaction_shape == 'random':
        interaction_fn = experiment.random
    return interaction_fn

def get_cooling_schedules():
    # init_temps = np.array([0.1, 1.0, 10.0])
    # cool_rates = np.array([0.9979, 0.9989, 0.9999])
    init_temps = np.array([1.0])
    cool_rates = np.array([0.9999])
    return list(itertools.product(init_temps, cool_rates))

def initialize_problem(structure, system_size, fill, interaction_shape, interaction_radius):
    interaction_fn = get_interaction_fn(interaction_shape)

    if structure == 'free':
        assert isinstance(system_size, int)
        assert interaction_shape == 'random'
        assert interaction_radius == 'NA'
        setup = experiment.FreeSpins(system_size)
        setup.turn_on_interactions(interaction_fn())
    else:
        assert isinstance(system_size, (list, tuple, np.ndarray))
        setup = experiment.SpinLattice(system_size[0], system_size[1], experiment.LATTICE_SPACING, prob_full=fill, triangular=(structure=='triangular'))
        setup.turn_on_interactions(interaction_fn(interaction_radius))
    return problem.MaxCutProblem(setup)

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--structure", default='square', help="System structure (options: square, triangular, free)")
    parser.add_argument("-n", "--size", help="System size (options: integer N or integers L_x, L_y, depending on structure)", nargs='+', type=int)
    parser.add_argument("-f", "--fill", help="Fill (options: float in range [0, 1])", type=float)
    parser.add_argument("-i", "--interaction", default='power_decay_fn', help="Interaction shape (options: step_fn, power_decay_fn, random)")
    parser.add_argument("-e", "--ensemble", default=False, help="Ensemble (options: True, False)", type=bool)
    args = parser.parse_args()
    structure = args.structure
    system_size = args.size
    if system_size:
        system_size = tuple(system_size) if len(system_size) > 1 else system_size[0]
    fill = args.fill
    interaction_shape = args.interaction
    ensemble = args.ensemble
    return structure, system_size, fill, interaction_shape, ensemble