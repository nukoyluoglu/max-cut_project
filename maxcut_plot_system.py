import experiment
import maxcut
import util
import csv
import collections

if __name__ == '__main__':
    structure, system_size, fill, interaction_shape, ensemble = maxcut.configure()
    
    parent_dir_path = '{}_{}'.format(structure, interaction_shape)
    exact_dir_path = '{}/exact_sols_{}'.format(parent_dir_path, system_size)
    algo_dir_path = '{}/algo_sols_{}'.format(parent_dir_path, system_size)

    system_sols = {}
    with open('{}/system_sols.csv'.format(algo_dir_path), 'r') as input_file:
        dict_reader = csv.DictReader(input_file)
        for row in dict_reader:
            interaction_radius = 'NA' if row['interaction_radius'] == 'NA' else float(row['interaction_radius'])
            system_sols[interaction_radius] = {}
            system_sols[interaction_radius]['init_temp'] = float(row['init_temp'])
            system_sols[interaction_radius]['cool_rate'] = float(row['cool_rate'])
            if row['step_from_exact']:
                system_sols[interaction_radius]['step_from_exact'] = float(row['step_from_exact'])
            if row['step_from_entropy']:
                system_sols[interaction_radius]['step_from_entropy'] = float(row['step_from_entropy'])
            if row['prob_ground_state_per_run']:
                system_sols[interaction_radius]['prob_ground_state_per_run'] = float(row['prob_ground_state_per_run'])
            system_sols[interaction_radius]['search_runtime'] = float(row['search_runtime'])
            if row['exact_min_energy']:
                system_sols[interaction_radius]['exact_min_energy'] = float(row['exact_min_energy'])

    if structure != 'free' and interaction_shape != 'random':
        util.plot_steps_vs_radius(system_sols, system_size, interaction_shape, algo_dir_path)

    for interaction_radius, system_sol in system_sols.items():
        exact_radius_dir_path = '{}/radius_{}'.format(exact_dir_path, interaction_radius)
        algo_radius_dir_path = '{}/radius_{}'.format(algo_dir_path, interaction_radius)

        prob = maxcut.initialize_problem(structure, system_size, fill, interaction_shape, interaction_radius)
        if structure != 'free' and interaction_shape != 'random':
            util.plot_interaction(maxcut.get_interaction_fn(interaction_shape), interaction_radius, system_size, experiment.LATTICE_SPACING, algo_radius_dir_path)

        param_results = collections.defaultdict(dict)
        with open('{}/param_results.csv'.format(algo_radius_dir_path), 'r') as input_file:
            dict_reader = csv.DictReader(input_file)
            for row in dict_reader:
                init_temp = float(row['init_temp'])
                cool_rate = float(row['cool_rate'])
                # print(cool_rate)
                param_results[init_temp][cool_rate] = {}
                param_results[init_temp][cool_rate]['min_energy'] = float(row['min_energy'])
                # print(param_results[init_temp][cool_rate])
                if row['min_energy_from_entropy']:
                    param_results[init_temp][cool_rate]['min_energy_from_entropy'] = float(row['min_energy_from_entropy'])
                if row['step_from_exact']:
                    param_results[init_temp][cool_rate]['step_from_exact'] = float(row['step_from_exact'])
                if row['step_from_entropy']:
                    param_results[init_temp][cool_rate]['step_from_entropy'] = float(row['step_from_entropy'])
                if row['step_per_run']:
                    param_results[init_temp][cool_rate]['step_per_run'] = int(row['step_per_run'])
                if row['prob_ground_state_per_run']:
                    param_results[init_temp][cool_rate]['prob_ground_state_per_run'] = float(row['prob_ground_state_per_run'])
                # print(param_results[init_temp].keys())
         
        for init_temp, cool_rate in maxcut.get_cooling_schedules():
            # print(init_temp, cool_rate)
            # print(param_results.keys(), param_results.values())
            # print(int(cool_rate in param_results[init_temp]))
            param_result = param_results[init_temp][cool_rate]

            stats_vs_temp = []
            with open('{}/stats_vs_temp_T_0_{}_r_{}.csv'.format(algo_radius_dir_path, init_temp, cool_rate), 'r') as input_file:
                dict_reader = csv.DictReader(input_file)
                for row in dict_reader:
                    stat_vs_temp = {}
                    stat_vs_temp['temp'] = float(row['temp'])
                    stat_vs_temp['ave_energy'] = float(row['ave_energy'])
                    stat_vs_temp['err_energy'] = float(row['err_energy'])
                    stat_vs_temp['entropy'] = float(row['entropy'])
                    if row['ave_prob_ground_state']:
                        stat_vs_temp['ave_prob_ground_state'] = float(row['ave_prob_ground_state'])
                    if row['err_prob_ground_state']:
                        stat_vs_temp['err_prob_ground_state'] = float(row['err_prob_ground_state'])
                    stats_vs_temp.append(stat_vs_temp)
            param_result['stats_vs_temp'] = stats_vs_temp

            stats_vs_t = []
            with open('{}/stats_vs_t_T_0_{}_r_{}.csv'.format(algo_radius_dir_path, init_temp, cool_rate), 'r') as input_file:
                dict_reader = csv.DictReader(input_file)
                for row in dict_reader:
                    stat_vs_t = {}
                    stat_vs_t['t'] = float(row['t'])
                    stat_vs_t['ave_temp'] = float(row['ave_temp'])
                    stat_vs_t['ave_energy'] = float(row['ave_energy'])
                    stat_vs_t['err_energy'] = float(row['err_energy'])
                    stat_vs_t['entropy'] = float(row['entropy'])
                    if row['ave_prob_ground_state']:
                        stat_vs_t['ave_prob_ground_state'] = float(row['ave_prob_ground_state'])
                    if row['err_prob_ground_state']:
                        stat_vs_t['err_prob_ground_state'] = float(row['err_prob_ground_state'])
                    if row['total_iter']:
                        stat_vs_t['total_iter'] = float(row['total_iter'])
                    if row['ave_prob_ground_state_from_entropy']:
                        stat_vs_t['ave_prob_ground_state_from_entropy'] = float(row['ave_prob_ground_state_from_entropy'])
                    if row['err_prob_ground_state_from_entropy']:
                        stat_vs_t['err_prob_ground_state_from_entropy'] = float(row['err_prob_ground_state_from_entropy'])
                    if row['total_iter_from_entropy']:
                        stat_vs_t['total_iter_from_entropy'] = float(row['total_iter_from_entropy'])
                    stats_vs_t.append(stat_vs_t)
            param_result['stats_vs_t'] = stats_vs_t
            
            util.plot_energy_temp_vs_step(stats_vs_t, system_size, interaction_shape, interaction_radius, init_temp, cool_rate, algo_radius_dir_path, system_sol.get('exact_min_energy'))
            util.plot_step_optimization(stats_vs_t, param_result['step_per_run'], param_result['step_from_exact'], system_size, interaction_shape, interaction_radius, init_temp, cool_rate, algo_radius_dir_path)
            util.plot_prob_ground_state_temp_vs_step(stats_vs_t, param_result['step_per_run'], param_result['step_from_exact'], system_size, interaction_shape, interaction_radius, init_temp, cool_rate, algo_radius_dir_path)
            util.plot_energy_entropy_vs_temp(stats_vs_temp, system_size, interaction_shape, interaction_radius, init_temp, cool_rate, algo_radius_dir_path, system_sol.get('exact_min_energy'))
            
        boltzmann_temps, boltzmann_energies = util.plot_params_energy_vs_temp_heatmap(param_results, system_sol['init_temp'], system_sol['cool_rate'], system_size, interaction_shape, interaction_radius, algo_radius_dir_path, system_sol.get('exact_min_energy'), exact_radius_dir_path)
        util.plot_params_energy_vs_temp(param_results, system_sol['init_temp'], system_sol['cool_rate'], system_size, interaction_shape, interaction_radius, algo_radius_dir_path, system_sol.get('exact_min_energy'), boltzmann_temps, boltzmann_energies)
        
        if structure != 'free':
            ground_state_partition_hist = []
            with open('{}/ground_state_partitions.csv'.format(algo_dir_path), 'r') as input_file:  
                dict_reader = csv.DictReader(input_file)
                for row in dict_reader:
                    partition = {}
                    for atom, spin in partition.values():
                        partition[int(atom)] = int(spin)
                    ground_state_partition_hist.append(partition)
            ground_state_energy_hist = []
            with open('{}/ground_state_energies.csv'.format(algo_dir_path), 'r') as input_file:  
                reader = csv.reader(input_file)
                next(reader)
                for row in reader:
                    ground_state_energy_hist.append(float(row[0]))
            util.plot_spin_lattice(prob, ground_state_partition_hist, ground_state_energy_hist, interaction_shape, interaction_radius, system_size, algo_radius_dir_path)
            
            exact_ground_state_partition_hist = []
            with open('{}/ground_state_partitions.csv'.format(exact_dir_path), 'r') as input_file:  
                dict_reader = csv.DictReader(input_file)
                for row in dict_reader:
                    partition = {}
                    for atom, spin in partition.values():
                        partition[int(atom)] = int(spin)
                    ground_state_partition_hist.append(partition)
            util.plot_spin_lattice(prob, exact_ground_state_partition_hist, [system_sol['exact_min_energy']], interaction_shape, interaction_radius, system_size, algo_radius_dir_path)        
