import experiment
import problem
import algorithms
import timeit
import util
import numpy as np
import copy
import csv

LATTICE_X = 5
LATTICE_Y = 5
LATTICE_SPACING = 1
NUM_PERFORMANCE_TRIALS = 100
NUM_PARAM_TRIALS = 20  

def initialize_problem(interaction_fn, radius, fill=1.0):
    setup = experiment.SpinLattice(LATTICE_X, LATTICE_Y, LATTICE_SPACING, prob_full=fill)
    setup.turn_on_interactions(interaction_fn(radius))
    return problem.MaxCutProblem(setup)

def run(algorithm, interaction_fn, radius, exact_best_energy=None, plot=False, fill=1.0):
    if plot:
        # plot of radius vs. interaction strength
        util.plot_interaction(interaction_fn(radius), radius, LATTICE_X, LATTICE_Y)
    partition_history_trials = []
    energy_history_trials = []
    temp_history_trials = []
    runtime_trials = []
    step_trials = []
    best_trial = 0
    best_energy = 0
    if exact_best_energy:
        best_energy = exact_best_energy
    for i in range(NUM_PERFORMANCE_TRIALS):
        prob = initialize_problem(interaction_fn, radius, fill=fill)
        myGlobals = globals()
        myGlobals.update({'algorithm': algorithm, 'prob': prob})
        runtime = timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
        partition_history_trials.append(copy.deepcopy(prob.get_partition_history()))
        energy_history_trials.append(copy.deepcopy(prob.get_energy_history()))
        temp_history_trials.append(copy.deepcopy(algorithm.get_temp_history()))
        runtime_trials.append(runtime)
        step_trials.append(len(algorithm.get_temp_history()))
        if not exact_best_energy and prob.get_energy() < best_energy:
            best_trial = i
            best_energy = prob.get_energy()
        elif exact_best_energy and prob.get_energy() == best_energy:
            best_trial = i
        elif exact_best_energy and prob.get_energy() < best_energy:
            raise RuntimeError("Energy cannot be lower than exact minimum energy")
    if plot: # display plot for best trial only
        # sample visualization of spin dynamics
        util.plot_spin_lattice(partition_history_trials[best_trial], LATTICE_X, LATTICE_Y, radius)
        # sample plot of energy vs. time
        # util.plot_energy_in_time(energy_history_trials[best_trial], radius)
        util.plot_energy_temp_in_time(energy_history_trials[best_trial], temp_history_trials[best_trial], radius)
    return {'partition_histories': partition_history_trials, 'energy_histories': energy_history_trials, 'runtimes': runtime_trials, 'temp_histories': temp_history_trials, 'steps': step_trials}

def run_trials_for_param(algorithm, interaction_fn, radius, init_temp, cool_rate, fill=1.0):
    ave_runtime = 0
    ave_step = 0
    ave_final_energy = 0
    best_step = 0
    best_final_energy = 0
    best_prob = None
    best_temp_history = None
    ave_energy_vs_temp = {}
    for _ in range(NUM_PARAM_TRIALS):
        prob = initialize_problem(interaction_fn, radius, fill=fill)
        algorithm.set_cooling_schedule(init_temp, cool_rate)
        myGlobals = globals()
        myGlobals.update({'algorithm': algorithm, 'prob': prob})
        runtime = timeit.timeit(stmt='algorithm.solve(prob)', number=1, globals=myGlobals)
        step = len(algorithm.get_temp_history())
        final_energy = prob.get_energy()
        energy_hist = prob.get_energy_history()
        temp_hist = algorithm.get_temp_history()
        ave_runtime += runtime
        ave_step += step
        ave_final_energy += final_energy
        if final_energy < best_final_energy or (final_energy == best_final_energy and step < best_step):
            best_final_energy = final_energy
            best_step = step
            best_prob = prob
            best_temp_history = copy.copy(temp_hist)
        elif best_prob == prob:
            raise RuntimeError('Best solution must be preserved throughout trials')
        if len(temp_hist) != len(energy_hist):
            raise RuntimeError('Length of temperature and energy histories must match')
        for t in range(len(temp_hist)):
            temp = temp_hist[t]
            # TODO: instead of last energy select last 50 energies (equilibrium)
            # TODO: get STD dev and plot error bars
            # TODO: Plot mean energy for each temperature and also plot std dev
            # Finite temperature - fluctuations - get mean energy instead of final energy
            # 50 equilibrate
            # 50 equilibrium
            # Ave over last 50
            if t == len(temp_hist) - 1 or temp != temp_hist[t + 1]: # last energy at given temperature after equilibration
                ave_energy_vs_temp.setdefault(temp, []).append(energy_hist[t])
    for temp in ave_energy_vs_temp.keys():
        ave_energy_vs_temp[temp] = np.mean(ave_energy_vs_temp[temp])
    ave_final_energy /= NUM_PARAM_TRIALS
    ave_step /= NUM_PARAM_TRIALS
    ave_runtime /= NUM_PARAM_TRIALS
    return {'ave_final_energy': ave_final_energy, 'ave_runtime': ave_runtime, 'ave_step': ave_step, 'best_final_energy': best_final_energy, 'ave_energy_vs_temp': ave_energy_vs_temp, 'best_partition_history': best_prob.get_partition_history(), 'best_energy_history': best_prob.get_energy_history(), 'best_temp_history': best_temp_history}

def search_params(algorithm_name, interaction_fn, radius, ensemble=1, exact_best_energy=None, fill=1.0, plot=True):
    if algorithm_name == 'simulated annealing':
        algorithm = algorithms.SimulatedAnnealing(ensemble=ensemble)
        init_temps = np.linspace(0, 10, 6)
        cool_rates = np.linspace(0.1, 0.9, 9)
        best = {'energy': 0, 'step': float('inf'), 'runtime': float('inf'), 'params': {'init_temp': None, 'cool_rate': None}, 'partition_history': [], 'energy_history': [], 'temp_history': []}
        best_solution = None
        ave_energy_vs_temp_by_init_temp = {}
        for i in range(len(init_temps)): # consider acceptances starting from optimal value for previous radius
            ave_energy_vs_temp_by_cool_rate = {}
            for j in range(len(cool_rates)):
                param_solution = run_trials_for_param(algorithm, interaction_fn, radius, init_temps[i], cool_rates[j], fill=fill)
                ave_energy_vs_temp_by_cool_rate[cool_rates[j]] = param_solution['ave_energy_vs_temp']
                if param_solution['best_final_energy'] < best['energy'] or (param_solution['best_final_energy'] == best['energy'] and param_solution['ave_step'] < best['step']):
                    best['energy'] = param_solution['best_final_energy']
                    best_solution = param_solution
                    best['step'] = best_solution['ave_step']
                    best['runtime'] = best_solution['ave_runtime']
                    best['params']['init_temp'] = init_temps[i]
                    best['params']['cool_rate'] = cool_rates[j]
                    best['partition_history'] = best_solution['best_partition_history']
                    best['energy_history'] = best_solution['best_energy_history']
                    best['temp_history'] = best_solution['best_temp_history']
                elif best_solution == param_solution:
                    raise RuntimeError('Best solution must be preserved throughout trials')
                if exact_best_energy and best['energy'] < exact_best_energy:
                    raise RuntimeError('Energy cannot be lower than exact minimum energy')
            ave_energy_vs_temp_by_init_temp[init_temps[i]] = ave_energy_vs_temp_by_cool_rate
        if plot:
            for init_temp, ave_energy_vs_temp_by_cool_rate in ave_energy_vs_temp_by_init_temp.items():
                util.plot_params_energy_vs_temp(ave_energy_vs_temp_by_cool_rate, init_temp, best['params'], radius, LATTICE_X, LATTICE_Y, exact_best_energy)
            # sample visualization of spin dynamics for best params
            util.plot_spin_lattice([best['partition_history'][-1]], LATTICE_X, LATTICE_Y, radius)
            # sample plot of energy vs. time for best params
            util.plot_energy_temp_in_time(best['energy_history'], best['temp_history'], radius)
        return best

def get_exact_solutions(interaction_fn, interaction_fn_name=None, fill=1.0):
    w1 = csv.writer(open(interaction_fn_name + '_exact_sols_{}x{}/optimal_sols.csv'.format(LATTICE_X, LATTICE_Y), 'w'))
    w1.writerow(['radius', 'min energy', '# ground states', '# states'])
    for radius in np.arange(1, util.euclidean_dist_2D((0, 0), (LATTICE_X, LATTICE_Y), 1)):
        prob = prob = initialize_problem(interaction_fn, radius, fill=fill)
        exact_best_energy, num_best_partitions, sample_best_partition, total_num_partitions, all_partition_energies = algorithms.BruteForce().solve(prob)
        filename = interaction_fn_name + '_exact_sols_{}x{}/optimal_spin_lattice_radius_{}.html'.format(LATTICE_X, LATTICE_Y, radius)
        util.plot_spin_lattice([sample_best_partition], LATTICE_X, LATTICE_Y, radius, filename=filename)
        w1.writerow([radius, exact_best_energy, num_best_partitions, total_num_partitions])
        w2 = csv.writer(open(interaction_fn_name + '_exact_sols_{}x{}/energies_radius_{}.csv'.format(LATTICE_X, LATTICE_Y, radius), 'w'))
        w2.writerow(['energy'])
        for energy in all_partition_energies:
            w2.writerow([energy])

def algorithm_performance(algorithm_name, interaction_fn, interaction_fn_name=None, fill=1.0, ensemble=1, brute_force_solution=None, plot=True):
    algorithm_performance_by_radius = {}
    radii = []
    runtimes = []
    steps = []
    for radius in np.arange(1, util.euclidean_dist_2D((0, 0), (LATTICE_X, LATTICE_Y), 1)):
        if plot:
            # plot of radius vs. interaction strength
            util.plot_interaction(interaction_fn(radius), radius, LATTICE_X, LATTICE_Y)
        exact_best_energy = None
        if brute_force_solution:
            exact_best_energy = brute_force_solution[radius]
        best_solution = search_params(algorithm_name, interaction_fn, radius, ensemble=ensemble, exact_best_energy=exact_best_energy, fill=fill)
        algorithm_performance_by_radius[radius] = best_solution
        radii.append(radius)
        runtimes.append(best_solution['runtime'])
        steps.append(best_solution['step'])
    if plot:
        util.plot_runtimes_steps_vs_radius(radii, runtimes, steps)
        # TODO: check scaling with system size
        # e.g. random coupling - exponential with system size
        # random coupling - how it scales with dimension
        # completely random graph
        # TODO: faster for short-range case: mechanism for knowing if it converges
        # TODO: filling of the lattice - randomly turn off corners
        # TODO: different interaction functions
        # TODO: find convergence without imposing cutoff for decaying interaction functions
        # TODO: try ensemble switch

def main(plot=True):
    exact_min_energy_by_radius = None
    # brute force, step function
    get_exact_solutions(experiment.step_fn, 'step_fn')
    exact_min_energy_by_radius = {}
    num_ground_states_by_radius = {}
    r = csv.reader(open('step_fn_exact_sols_{}x{}/optimal_sols.csv'.format(LATTICE_X, LATTICE_Y), 'r'))
    next(r)
    for row in r:
        radius = float(row[0])
        exact_min_energy_by_radius[radius] = float(row[1])
        num_ground_states_by_radius[radius] = float(row[2])
    # simulated annealing, step function
    algorithm_performance('simulated annealing', experiment.step_fn, ensemble=1, brute_force_solution=exact_min_energy_by_radius)
    # get_exact_solutions(experiment.power_decay_fn, 'power_decay_fn')
    # algorithm_performance('simulated annealing', experiment.power_decay_fn, 'power_decay_fn', brute_force=True)
    # get_exact_solutions(experiment.step_fn, 'step_fn_0.8_filled', fill=0.8)
    # since system is probabilistic, brute force is not accurate
    # algorithm_performance('simulated annealing', experiment.step_fn, 'step_fn_0.8_filled', brute_force=False, fill=0.8)
    # algorithm_performance(algorithms.SemiDefinite())

if __name__ == '__main__':
    main()
