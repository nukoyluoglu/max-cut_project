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
            init_temp = float(row['init_temp'])
            cool_rate = float(row['cool_rate'])
            step_from_exact = float(row['step_from_exact'])
            step_from_entropy = float(row['step_from_entropy'])
            prob_ground_state_per_run = float(row['prob_ground_state_per_run'])
            search_runtime = float(row['search_runtime'])
            exact_min_energy = float(row['exact_min_energy'])
            system_sols[interaction_radius] = (dict(init_temp=init_temp, cool_rate=cool_rate, step_from_exact=step_from_exact, step_from_entropy=step_from_entropy, prob_ground_state_per_run=prob_ground_state_per_run, search_runtime=search_runtime, exact_min_energy=exact_min_energy))

    if structure != 'free' and interaction_shape != 'random':
        util.plot_steps_vs_radius(system_sols, system_size[0], system_size[1], interaction_shape, algo_dir_path)

    for interaction_radius, system_sol in system_sols.items():
        exact_radius_dir_path = '{}/radius_{}'.format(exact_dir_path, interaction_radius)
        algo_radius_dir_path = '{}/radius_{}'.format(algo_dir_path, interaction_radius)

        prob = maxcut.initialize_problem(structure, system_size, fill, interaction_shape, interaction_radius)
        if structure != 'free' and interaction_shape != 'random':
            util.plot_interaction(maxcut.get_interaction_fn(interaction_shape), interaction_radius, system_size[0], system_size[1], experiment.LATTICE_SPACING, algo_radius_dir_path)

        param_results = collections.defaultdict(dict)
        with open('{}/param_results.csv'.format(algo_radius_dir_path), 'r') as input_file:
            dict_reader = csv.DictReader(input_file)
            for row in dict_reader:
                init_temp = float(row['init_temp'])
                cool_rate = float(row['cool_rate'])
                min_energy = float(row['min_energy'])
                min_energy_from_entropy = float(row['min_energy_from_entropy'])
                step_from_exact = float(row['step_from_exact'])
                step_from_entropy = float(row['step_from_entropy'])
                step_per_run = int(row['step_per_run'])
                prob_ground_state_per_run = float(row['prob_ground_state_per_run'])
                param_results[init_temp][cool_rate] = dict(min_energy=min_energy, min_energy_from_entropy=min_energy_from_entropy, step_from_exact=step_from_exact, step_from_entropy=step_from_entropy, step_per_run=step_per_run, prob_ground_state_per_run=prob_ground_state_per_run)
    
        for init_temp, cool_rate in maxcut.get_cooling_schedules():
            param_result = param_results[init_temp][cool_rate]

            stats_vs_temp = []
            with open('{}/stats_vs_temp_T_0_{}_r_{}.csv'.format(algo_radius_dir_path, init_temp, cool_rate), 'r') as input_file:
                dict_reader = csv.DictReader(input_file)
                for row in dict_reader:
                    temp = float(row['temp'])
                    ave_energy = float(row['ave_energy'])
                    err_energy = float(row['err_energy'])
                    entropy = float(row['entropy'])
                    ave_prob_ground_state = float(row['ave_prob_ground_state'])
                    err_prob_ground_state = float(row['err_prob_ground_state'])
                    stats_vs_temp.append(dict(temp=temp, ave_energy=ave_energy, err_energy=err_energy, entropy=entropy, ave_prob_ground_state=ave_prob_ground_state, err_prob_ground_state=err_prob_ground_state))
            param_result['stats_vs_temp'] = stats_vs_temp

            stats_vs_t = []
            with open('{}/stats_vs_t_T_0_{}_r_{}.csv'.format(algo_radius_dir_path, init_temp, cool_rate), 'r') as input_file:
                dict_reader = csv.DictReader(input_file)
                for row in dict_reader:
                    t = float(row['t'])
                    ave_temp = float(row['ave_temp'])
                    ave_energy = float(row['ave_energy'])
                    err_energy = float(row['err_energy'])
                    entropy = float(row['entropy'])
                    ave_prob_ground_state = float(row['ave_prob_ground_state'])
                    err_prob_ground_state = float(row['err_prob_ground_state'])
                    total_iter = float(row['total_iter'])
                    ave_prob_ground_state_from_entropy = float(row['ave_prob_ground_state_from_entropy'])
                    err_prob_ground_state_from_entropy = float(row['err_prob_ground_state_from_entropy'])
                    total_iter_from_entropy = float(row['total_iter_from_entropy'])
                    stats_vs_t.append(dict(t=t, ave_temp=ave_temp, ave_energy=ave_energy, err_energy=err_energy, entropy=entropy, min_energy=min_energy, ave_prob_ground_state=ave_prob_ground_state, err_prob_ground_state=err_prob_ground_state, total_iter=total_iter, min_energy_from_entropy=min_energy_from_entropy, ave_prob_ground_state_from_entropy=ave_prob_ground_state_from_entropy, err_prob_ground_state_from_entropy=err_prob_ground_state_from_entropy, total_iter_from_entropy=total_iter_from_entropy))
            param_result['stats_vs_t'] = stats_vs_t
            
            util.plot_energy_temp_vs_step(stats_vs_t, system_size, interaction_shape, interaction_radius, init_temp, cool_rate, algo_radius_dir_path, system_sol['exact_min_energy'])
            util.plot_step_optimization(stats_vs_t, param_result['step_per_run'], param_result['step_from_exact'], system_size, interaction_shape, interaction_radius, init_temp, cool_rate, algo_radius_dir_path)
            util.plot_prob_ground_state_temp_vs_step(stats_vs_t, param_result['step_per_run'], param_result['step_from_exact'], system_size, interaction_shape, interaction_radius, init_temp, cool_rate, algo_radius_dir_path)
            util.plot_energy_entropy_vs_temp(stats_vs_temp, system_size, interaction_shape, interaction_radius, init_temp, cool_rate, algo_radius_dir_path, system_sol['exact_min_energy'])
            
        boltzmann_temps, boltzmann_energies = util.plot_params_energy_vs_temp_heatmap(param_results, system_sol['init_temp'], system_sol['cool_rate'], system_size, interaction_shape, interaction_radius, algo_radius_dir_path, system_sol['exact_min_energy'], exact_radius_dir_path)
        util.plot_params_energy_vs_temp(param_results, system_sol['init_temp'], system_sol['cool_rate'], system_size, interaction_shape, interaction_radius, algo_radius_dir_path, system_sol['exact_min_energy'], exact_radius_dir_path, boltzmann_temps, boltzmann_energies)
        
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
            util.plot_spin_lattice(prob, ground_state_partition_hist, ground_state_energy_hist, interaction_shape, interaction_radius, system_size[0], system_size[1], algo_radius_dir_path)
            
            exact_ground_state_partition_hist = []
            with open('{}/ground_state_partitions.csv'.format(exact_dir_path), 'r') as input_file:  
                dict_reader = csv.DictReader(input_file)
                for row in dict_reader:
                    partition = {}
                    for atom, spin in partition.values():
                        partition[int(atom)] = int(spin)
                    ground_state_partition_hist.append(partition)
            util.plot_spin_lattice(prob, exact_ground_state_partition_hist, [system_sol['exact_min_energy']], interaction_shape, interaction_radius, system_size[0], system_size[1], algo_radius_dir_path)        
