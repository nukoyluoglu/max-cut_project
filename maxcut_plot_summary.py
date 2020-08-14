import maxcut
import util
import numpy as np
import csv
import os

if __name__ == '__main__':
    structure, system_size, fill, interaction_shape, ensemble = maxcut.configure()
    assert system_size == None and fill == None

    parent_dir_path = '{}_{}'.format(structure, interaction_shape)
    algo_summary_dir_path = '{}/algo_sols_summary'.format(parent_dir_path)
    util.make_dir(algo_summary_dir_path)
    exact_summary_dir_path = '{}/exact_sols_summary'.format(parent_dir_path)
    util.make_dir(exact_summary_dir_path)

    algo_sols = {}
    exact_sols = {}
    random_sols = {}
    for dir_name in os.listdir(parent_dir_path):
        dir_path = os.path.join(parent_dir_path, dir_name)
        if dir_name.startswith('algo_sols_') and not dir_name.endswith('summary') and os.path.isdir(dir_path):
            system_size = dir_name[10:]
            if ',' not in system_size:
                system_size = int(system_size)

            algo_system_sols = {}
            with open('{}/system_sols.csv'.format(dir_path), 'r') as input_file:
                dict_reader = csv.DictReader(input_file)
                for row in dict_reader:
                    interaction_radius = 'NA' if row['interaction_radius'] == 'NA' else float(row['interaction_radius'])
                    algo_system_sols[interaction_radius] = {}
                    algo_system_sols[interaction_radius]['init_temp'] = float(row['init_temp'])
                    algo_system_sols[interaction_radius]['cool_rate'] = float(row['cool_rate'])
                    algo_system_sols[interaction_radius]['step_from_exact'] = float(row['step_from_exact']) if row['step_from_exact'] else None
                    algo_system_sols[interaction_radius]['step_from_entropy'] = float(row['step_from_entropy']) if row['step_from_entropy'] else None
                    algo_system_sols[interaction_radius]['prob_ground_state_per_run'] = float(row['prob_ground_state_per_run']) if row['prob_ground_state_per_run'] else None
                    algo_system_sols[interaction_radius]['search_runtime'] = float(row['search_runtime'])
                    algo_system_sols[interaction_radius]['exact_min_energy'] = float(row['exact_min_energy']) if row['exact_min_energy'] else None
            algo_sols[system_size] = algo_system_sols

            exact_dir_path = '{}/exact_sols_{}'.format(parent_dir_path, system_size)
            exact_system_sols = {}
            with open('{}/exact_sols.csv'.format(exact_dir_path), 'r') as input_file:
                dict_reader = csv.DictReader(input_file)
                for row in dict_reader:
                    interaction_radius = 'NA' if row['radius'] == 'NA' else float(row['radius'])
                    min_energy = float(row['min energy'])
                    num_ground_states = int(row['# ground states'])
                    num_states = int(row['# states'])
                    min_energy_gap = float(row['min energy gap'])
                    search_runtime = float(row['runtime'])
                    exact_system_sols[interaction_radius] = (dict(min_energy=min_energy, num_ground_states=num_ground_states, num_states=num_states, min_energy_gap=min_energy_gap, search_runtime=search_runtime))
            exact_sols[system_size] = exact_system_sols

            random_system_sols = {}
            for interaction_radius, sol in exact_system_sols.items():
                prob_ground_state_random = sol['num_ground_states'] / sol['num_states']
                optimal_t_random = 1 / np.abs(np.log(1 - prob_ground_state_random))
                random_system_sols[interaction_radius] = optimal_t_random
            random_sols[system_size] = random_system_sols

    util.plot_steps_vs_system_size(algo_sols, interaction_shape, algo_summary_dir_path, random_sols=random_sols)
    util.plot_num_ground_states_vs_system_size(exact_sols, interaction_shape, exact_summary_dir_path)
        