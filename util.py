import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotly import graph_objects as go, express as px
from collections import defaultdict
import csv
from celluloid import Camera
import os

class Vertex(object):

    def __init__(self, node):
        self.id = node
        self.adjacent = defaultdict(int)

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph(object):

    def __init__(self, vert_dict=None):
        if vert_dict == None:
            vert_dict = {}
        self.vert_dict = vert_dict
        self.num_vertices = len(self.vert_dict)

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices += 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def add_edge(self, frm, to, weight = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)
        self.vert_dict[frm].add_neighbor(to, weight)
        self.vert_dict[to].add_neighbor(frm, weight)

    def get_vertices(self):
        return list(self.vert_dict.keys())
    
    def get_vertex_obj(self, v):
        if v in self.vert_dict:
            return self.vert_dict[v]
        else:
            return None
    
    def get_vertex_dict(self):
        return self.vert_dict
    
    def get_edge(self, frm, to):
        if frm in self.vert_dict and to in self.vert_dict:
            return self.vert_dict[frm].get_weight(to)
        else:
            return None

    def get_neighbors(self, v):
        if v in self.vert_dict:
            return self.vert_dict[v].get_connections()
        else:
            return None

    def get_num_vertices(self):
        return self.num_vertices

def euclidean_dist_2D(loc1, loc2, spacing):
    coord1 = spacing * np.array(loc1)
    coord2 = spacing * np.array(loc2)
    return np.linalg.norm(coord1 - coord2)

def plot_interaction(interaction_fn, radius, lattice_X, lattice_Y, interaction_shape, path):
    plt.figure()
    dist = np.arange(0, euclidean_dist_2D((0, 0), (lattice_X, lattice_Y), 1), 0.01)
    interaction_strength = [interaction_fn(r) for r in dist]
    plt.plot(dist, interaction_strength)
    plt.xlabel('Distance - r')
    plt.ylabel('Interaction Strength - J')
    plt.ylim(bottom=0)
    plt.axvline(radius, linestyle='dashed', color='g')
    plt.savefig('{}/interaction_function.png'.format(path))
    plt.close()

def plot_runtimes_steps_vs_radius(algorithm_performance_by_radius, lattice_X, lattice_Y, interaction_shape, path):
    radii = []
    runtimes = []
    steps = []
    col_info = {}
    for radius, (solution, params) in algorithm_performance_by_radius.items():
        radii.append(radius)
        runtimes.append(solution['runtime'])
        steps.append(solution['step'])
        col_info[radius] = 'radius = {}\nE_f = {}\nT_0 = {}, r = {}'.format(radius, round(solution['ave_final_energy'], 1), params['init_temp'], params['cool_rate'])
    fig, ax1 = plt.subplots()
    plt.title('L = {}, {}'.format(lattice_X, interaction_shape))
    color = 'tab:red'
    ax1.set_xlabel('Radius - r')
    ax1.set_ylabel('Runtime (s)', color=color)
    ax1.plot(radii, runtimes, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bottom=0)
    ax1.set_xticks(list(col_info.keys()))
    ax1.set_xticklabels(list(col_info.values()))
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Steps', color=color)
    ax2.plot(radii, steps, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('{}/runtimes_steps_vs_radius.png'.format(path))
    plt.close()

def plot_runtimes_steps_vs_system_size(algorithm_performance_by_system, interaction_shape, path):
    runtimes_steps_vs_system_size_by_radius = defaultdict(dict)
    if interaction_shape == 'random':
        for system_size, algorithm_performance_by_radius in algorithm_performance_by_system.items():
            runtimes_steps_vs_system_size_by_radius['NA'][system_size] = algorithm_performance_by_radius
    else:
        for system_size, algorithm_performance_by_radius in algorithm_performance_by_system.items():
            for radius, solution in algorithm_performance_by_radius.items():
                runtimes_steps_vs_system_size_by_radius[radius][system_size] = solution
    for radius, runtimes_steps_vs_system_size in runtimes_steps_vs_system_size_by_radius.items():
        system_sizes = []
        runtimes = []
        steps = []
        col_info = {}
        for system_size, (solution, params) in runtimes_steps_vs_system_size.items():
            system_sizes.append(system_size)
            runtimes.append(solution['runtime'])
            steps.append(solution['step'])
            col_info[system_size] = 'L = {}\nE_f = {}\nT_0 = {}, r = {}'.format(system_size, round(solution['best_energy'], 1), params['init_temp'], params['cool_rate'])
        fig, ax1 = plt.subplots()
        plt.title('{}, radius = {}'.format(interaction_shape, radius))
        color = 'tab:red'
        ax1.set_xlabel('System Size - L')
        ax1.set_ylabel('Runtime (s)', color=color)
        ax1.plot(system_sizes, runtimes, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Steps', color=color)
        ax2.plot(system_sizes, steps, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)
        plt.xticks(list(col_info.keys()), list(col_info.values()))
        plt.tight_layout()
        plt.savefig('{}/runtimes_steps_vs_system_size_radius_{}.png'.format(path, radius))
        plt.close()

def plot_num_ground_states_vs_system_size(num_ground_states_by_system, interaction_shape, path):
    num_ground_states_vs_system_size_by_radius = defaultdict(dict)
    if interaction_shape == 'random':
        for system_size, num_ground_states_by_radius in num_ground_states_by_system.items():
            num_ground_states_vs_system_size_by_radius['NA'][system_size] = num_ground_states_by_radius
    else:
        for system_size, num_ground_states_by_radius in num_ground_states_by_system.items():
            for radius, num_ground_states in num_ground_states_by_radius.items():
                num_ground_states_vs_system_size_by_radius[radius][system_size] = num_ground_states
    for radius, num_ground_states_vs_system_size in num_ground_states_vs_system_size_by_radius.items():
        system_sizes = []
        nums_ground_states = []
        for system_size, num_ground_states in num_ground_states_vs_system_size.items():
            system_sizes.append(system_size)
            nums_ground_states.append(num_ground_states)
        plt.figure()
        plt.plot(system_sizes, nums_ground_states)
        plt.xlabel('System Size - L')
        plt.ylabel('Number of Ground States')
        plt.title('{}, radius = {}'.format(interaction_shape, radius))
        plt.savefig('{}/num_ground_states_vs_system_size_radius_{}.png'.format(path,radius))
        plt.close()

def get_spin_lattice(spins):
    x_up = []
    y_up = []
    x_down = []
    y_down = []
    for atom, spin in spins.items():
        if spin == 1:
            x_up.append(atom[0])
            y_up.append(atom[1])
        else:
            x_down.append(atom[0])
            y_down.append(atom[1])
    return np.array(x_up), np.array(y_up), np.array(x_down), np.array(y_down)

def plot_spin_lattice(spin_history, energy_history, radius, lattice_X, lattice_Y, interaction_shape, triangular, path):
    fig = plt.figure()
    plt.title('L = {}, {}, radius = {}'.format(lattice_X, interaction_shape, radius))
    camera = Camera(fig)
    # plot 1 spin configuration per temperature
    for t in range(0, len(spin_history), 1000):
        x_up, y_up, x_down, y_down = get_spin_lattice(spin_history[t])
        plt.scatter(x_up, y_up, s=20**2, c='red')
        plt.scatter(x_down, y_down, s=20**2, c='blue')
        plt.text(0.05, 0.95, 'E = {}'.format(round(energy_history[t], 1)), transform=fig.transFigure, verticalalignment='top')
        plt.gca().set_aspect('equal', adjustable='box')
        camera.snap()
    animation = camera.animate()
    animation.save('{}/spin_lattice.gif'.format(path), writer = 'imagemagick')
    plt.close()

def plot_energy_temp_vs_step(ave_energy_history, ave_temp_history, radius, system_size, interaction_shape, init_temp, cool_rate, path, exact_best_energy):
    ave_energy_history = np.array([ave_energy_history[t] for t in range(len(ave_energy_history))])
    ave_temp_history = np.array([ave_temp_history[t] for t in range(len(ave_temp_history))])
    if len(ave_energy_history) != len(ave_temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
    fig, ax1 = plt.subplots()
    title = 'L = {}, {}, radius = {}, T_0 = {}, r = {}\nE_min = {}'.format(system_size, interaction_shape, radius, init_temp, cool_rate, min(ave_energy_history))
    color = 'tab:red'
    ax1.set_xlabel('Time Steps - t')
    ax1.set_ylabel('Energy - E', color=color)
    if exact_best_energy:
        ave_energy_history -= exact_best_energy
        title += ', exact E_min = {}'.format(exact_best_energy)
        ax1.set_ylabel('Energy Difference from Ground State - \u0394E', color=color)
        ax1.set_yscale('log')
    ax1.plot(range(len(ave_energy_history)), ave_energy_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Temperature - T', color=color)
    ax2.plot(range(len(ave_temp_history)), ave_temp_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(title)
    fig.tight_layout()
    plt.savefig('{}/energy_temp_vs_step_T0_{}_r_{}.png'.format(path, init_temp, cool_rate))
    plt.close()

def plot_prob_ground_state_temp_vs_step(prob_ground_state_history, ave_temp_history, optimal_step, radius, num_particles, interaction_shape, init_temp, cool_rate, path):
    prob_ground_state_history = np.array([prob_ground_state_history[t] for t in range(len(prob_ground_state_history))])
    ave_temp_history = np.array([ave_temp_history[t] for t in range(len(ave_temp_history))])
    if len(prob_ground_state_history) != len(ave_temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
    fig, ax1 = plt.subplots()
    title = 'N = {}, {}, radius = {}, T_0 = {}, r = {}'.format(num_particles, interaction_shape, radius, init_temp, cool_rate)
    color = 'tab:red'
    ax1.set_xlabel('Time Steps - t')
    ax1.set_ylabel('Probability of Reaching Ground State - P(t)', color=color)
    ax1.plot(range(len(prob_ground_state_history)), prob_ground_state_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Temperature - T', color=color)
    ax2.plot(range(len(ave_temp_history)), ave_temp_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    if optimal_step:
        plt.axvline(optimal_step, color='g', label='t_optimal = {}, P(t_optimal) = {}'.format(optimal_step, prob_ground_state_history[optimal_step]))
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    plt.savefig('{}/prob_ground_state_temp_vs_step_T0_{}_r_{}.png'.format(path, init_temp, cool_rate))
    plt.close()

def plot_step_optimization(optimize_step, ave_temp_history, optimal_step, radius, num_particles, interaction_shape, init_temp, cool_rate, path):
    optimize_step = np.array([optimize_step[t] for t in range(1, len(optimize_step) + 1)])
    ave_temp_history = np.array([ave_temp_history[t] for t in range(1, len(ave_temp_history))])
    if len(optimize_step) != len(ave_temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
    fig, ax1 = plt.subplots()
    title = 'N = {}, {}, radius = {}, T_0 = {}, r = {}'.format(num_particles, interaction_shape, radius, init_temp, cool_rate)
    color = 'tab:red'
    ax1.set_xlabel('Time Steps - t')
    ax1.set_ylabel('Quantity to Optimize - t / log(1 - P(t))', color=color)
    ax1.plot(range(1, len(optimize_step) + 1), optimize_step, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Temperature - T', color=color)
    ax2.plot(range(1, len(ave_temp_history) + 1), ave_temp_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    if optimal_step:
        plt.axvline(optimal_step, color='g', label='t_optimal = {}'.format(optimal_step))
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    plt.savefig('{}/step_optimization_temp_vs_step_T0_{}_r_{}.png'.format(path, init_temp, cool_rate))
    plt.close()

def plot_temporary(energy_history, temp_history, init_temp, cool_rate, t, path, exact_best_energy):
    if len(energy_history) != len(temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
    if exact_best_energy:
        energy_history = np.array(energy_history) - exact_best_energy
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Time Steps - t')
    ax1.set_ylabel('Energy - E', color=color)
    ax1.plot(range(len(energy_history)), energy_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Temperature - T', color=color)
    ax2.plot(range(len(temp_history)), temp_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.savefig('{}/{}_energy_temp_in_time_T0_{}_r_{}.png'.format(path, t, init_temp, cool_rate))
    plt.close()

def boltzmann_dist(all_states_energy, temp):
    num = 0.0
    denom = 0.0
    for state_energy in all_states_energy:
        factor = np.exp(- state_energy / temp)
        num += state_energy * factor
        denom += factor
    return num / denom

def plot_params_energy_vs_temp(ave_energy_vs_temp_by_params, best_params, radius, system_size, interaction_shape, path, exact_best_energy, exact_path):
    for init_temp, ave_energy_vs_temp_by_cool_rate in ave_energy_vs_temp_by_params.items():
        plt.figure()
        plt.title('N = {}, {}, radius = {}'.format(system_size, interaction_shape, radius))
        if exact_best_energy:
            plt.axhline(exact_best_energy, linestyle='dashed', color='b', alpha=0.5, label='brute force')
        for cool_rate, ave_energy_vs_temp in ave_energy_vs_temp_by_cool_rate.items():
            inverse_temps = []
            ave_energies = []
            errors = []
            for temp, (ave_energy, error) in ave_energy_vs_temp.items():
                inverse_temps.append(1.0 / temp)
                ave_energies.append(ave_energy)
                errors.append(error)
            label = 'T_0 = {}, r = {}, T_f = {}'.format(init_temp, round(cool_rate, 1), np.format_float_scientific(1.0 / inverse_temps[-1], precision=1))
            if init_temp == best_params['init_temp'] and cool_rate == best_params['cool_rate']:
                label += ', optimal'
            p = plt.plot(inverse_temps, ave_energies, label=label)
            plt.errorbar(inverse_temps, ave_energies, yerr=errors, alpha=0.5, ecolor=p[-1].get_color())
        if exact_best_energy:
            all_energies = []
            r = csv.reader(open('{}/energies_radius_{}.csv'.format(exact_path, radius), 'r'))
            next(r)
            for row in r:
                all_energies.append(float(row[0]))
            exact_ave_energies = [boltzmann_dist(all_energies, 1.0 / inverse_temp) for inverse_temp in inverse_temps]
            plt.plot(inverse_temps, exact_ave_energies, label='Boltzmann distribution')
        plt.legend()
        plt.xscale('log')
        plt.xlabel('1 / Temperature - 1 / T')
        plt.ylabel('Energy - E')
        plt.savefig('{}/param_energy_vs_temp_T0_{}.png'.format(path, init_temp))
        plt.close()

def plot_params_energy_vs_temp_heatmap(ave_energy_vs_temp_by_params, best_params, radius, system_size, interaction_shape, path, exact_best_energy, exact_path):
    num_subplots = sum(len(n) for n in ave_energy_vs_temp_by_params.values())
    if exact_best_energy:
        num_subplots += 1
    fig, axs = plt.subplots(num_subplots, 1, sharex=True, gridspec_kw=dict(hspace=0), figsize=(20, 15), constrained_layout=True)
    fig.suptitle('N = {}, {}, radius = {}'.format(system_size, interaction_shape, radius))
    x_min = float('inf')
    x_max = 0
    h_min = exact_best_energy if exact_best_energy else float('inf')
    h_max = np.float('-inf')
    param_data = []
    for init_temp, ave_energy_vs_temp_by_cool_rate in ave_energy_vs_temp_by_params.items():
        for cool_rate, ave_energy_vs_temp in ave_energy_vs_temp_by_cool_rate.items():
            inverse_temps = []
            ave_energies = []
            errors = []
            for temp, (ave_energy, error) in ave_energy_vs_temp.items():
                inverse_temps.append(1.0 / temp)
                ave_energies.append(ave_energy)
                errors.append(error)
                x_min = min(1.0 / temp, x_min)
                x_max = max(1.0 / temp, x_max)
                h_min = min(ave_energy, h_min)
                h_max = max(ave_energy, h_max)
            param_data.append((inverse_temps, (init_temp, cool_rate), ave_energies))
    for i in range(len(param_data)):
        x, (T_0, r), h = param_data[i]
        x_edges, y_edges = np.meshgrid(x, [0, 5])
        heatmap = axs[i].pcolormesh(x_edges, y_edges, np.array(h)[np.newaxis,:], vmin = h_min, vmax = h_max, cmap="jet")
        label = 'T_0 = {}, r = {},\nT_f = {}'.format(T_0, round(r, 1), np.format_float_scientific(1.0 / x[-1], precision=1))
        if T_0 == best_params['init_temp'] and r == best_params['cool_rate']:
            label += ', optimal'
        axs[i].set_ylabel(label, rotation='horizontal', verticalalignment='center')
        axs[i].set_yticks([])
        axs[i].xaxis.set_visible(False)
    i = len(param_data)
    if exact_best_energy:
        all_energies = []
        r = csv.reader(open('{}/energies_radius_{}.csv'.format(exact_path, radius), 'r'))
        next(r)
        for row in r:
            all_energies.append(float(row[0]))
        inverse_temps = np.logspace(np.floor(np.log10(x_min)), np.ceil(np.log10(x_max)))
        exact_ave_energies = [boltzmann_dist(all_energies, 1.0 / inverse_temp) for inverse_temp in inverse_temps]
        X, Y = np.meshgrid(inverse_temps, [0, 5])
        axs[i].pcolormesh(X, Y, np.array(exact_ave_energies)[np.newaxis,:], vmin = h_min, vmax = h_max, cmap="jet", )
        axs[i].set_ylabel('Boltzmann distribution', rotation='horizontal', verticalalignment='center')
        axs[i].set_xlabel('1 / Temperature - 1 / T')
        axs[i].set_xscale('log')
        axs[i].set_yticks([])
        axs[i].set_xlim(x_min, x_max)
    # plt.subplots_adjust(hspace=None, wspace=None)
    fig.colorbar(heatmap, ax=axs)
    plt.savefig('{}/param_energy_vs_temp_heatmap.png'.format(path), bbox_inches='tight')
    plt.close()

def plot_params_prob_ground_state_vs_temp(prob_ground_state_vs_temp_by_params, best_params, radius, system_size, interaction_shape, path, exact_best_energy, exact_path):
    for init_temp, prob_ground_state_vs_temp_by_cool_rate in prob_ground_state_vs_temp_by_params.items():
        plt.figure()
        plt.title('N = {}, {}, radius = {}'.format(system_size, interaction_shape, radius))
        for cool_rate, prob_ground_state_vs_temp in prob_ground_state_vs_temp_by_cool_rate.items():
            inverse_temps = []
            probs_ground_state = []
            errors = []
            for temp, (prob_ground_state, error) in prob_ground_state_vs_temp.items():
                inverse_temps.append(1.0 / temp)
                probs_ground_state.append(prob_ground_state)
                errors.append(error)
            label = 'T_0 = {}, r = {}, T_f = {}'.format(init_temp, round(cool_rate, 1), np.format_float_scientific(1.0 / inverse_temps[-1], precision=1))
            if init_temp == best_params['init_temp'] and cool_rate == best_params['cool_rate']:
                label += ', optimal'
            p = plt.plot(inverse_temps, probs_ground_state, label=label)
            plt.errorbar(inverse_temps, probs_ground_state, yerr=errors, alpha=0.5, ecolor=p[-1].get_color())
        if exact_best_energy:
            all_energies = []
            r = csv.reader(open('{}/energies_radius_{}.csv'.format(exact_path, radius), 'r'))
            next(r)
            for row in r:
                all_energies.append(float(row[0]))
            exact_energies = [boltzmann_dist(all_energies, 1.0 / inverse_temp) for inverse_temp in inverse_temps]
            exact_probs_ground_state = [float(energy == exact_best_energy) for energy in exact_energies]
            plt.plot(inverse_temps, exact_probs_ground_state, label='Boltzmann distribution')
        plt.legend()
        plt.xscale('log')
        plt.xlabel('1 / Temperature - 1 / T')
        plt.ylabel('Probability of Reaching Ground State - P(T)')
        plt.savefig('{}/param_prob_ground_state_vs_temp_T0_{}.png'.format(path, init_temp))
        plt.close()

def plot_params_prob_ground_state_vs_temp_heatmap(prob_ground_state_vs_temp_by_params, best_params, radius, system_size, interaction_shape, path, exact_best_energy, exact_path):
    num_subplots = sum(len(n) for n in prob_ground_state_vs_temp_by_params.values())
    if exact_best_energy:
        num_subplots += 1
    fig, axs = plt.subplots(num_subplots, 1, sharex=True, gridspec_kw=dict(hspace=0), figsize=(20, 15), constrained_layout=True)
    fig.suptitle('N = {}, {}, radius = {}'.format(system_size, interaction_shape, radius))
    x_min = float('inf')
    x_max = 0
    param_data = []
    for init_temp, prob_ground_state_vs_temp_by_cool_rate in prob_ground_state_vs_temp_by_params.items():
        for cool_rate, prob_ground_state_vs_temp in prob_ground_state_vs_temp_by_cool_rate.items():
            inverse_temps = []
            probs_ground_state = []
            errors = []
            for temp, (prob_ground_state, error) in prob_ground_state_vs_temp.items():
                inverse_temps.append(1.0 / temp)
                probs_ground_state.append(prob_ground_state)
                errors.append(error)
                x_min = min(1.0 / temp, x_min)
                x_max = max(1.0 / temp, x_max)
            param_data.append((inverse_temps, (init_temp, cool_rate), probs_ground_state))
    for i in range(len(param_data)):
        x, (T_0, r), h = param_data[i]
        x_edges, y_edges = np.meshgrid(x, [0, 5])
        heatmap = axs[i].pcolormesh(x_edges, y_edges, np.array(h)[np.newaxis,:], vmin = 0.0, vmax = 1.0, cmap="jet")
        label = 'T_0 = {}, r = {},\nT_f = {}'.format(T_0, round(r, 1), np.format_float_scientific(1.0 / x[-1], precision=1))
        if T_0 == best_params['init_temp'] and r == best_params['cool_rate']:
            label += ', optimal'
        axs[i].set_ylabel(label, rotation='horizontal', verticalalignment='center')
        axs[i].set_yticks([])
        axs[i].xaxis.set_visible(False)
    i = len(param_data)
    if exact_best_energy:
        all_energies = []
        r = csv.reader(open('{}/energies_radius_{}.csv'.format(exact_path, radius), 'r'))
        next(r)
        for row in r:
            all_energies.append(float(row[0]))
        inverse_temps = np.logspace(np.floor(np.log10(x_min)), np.ceil(np.log10(x_max)))
        exact_energies = [boltzmann_dist(all_energies, 1.0 / inverse_temp) for inverse_temp in inverse_temps]
        exact_probs_ground_state = [float(energy == exact_best_energy) for energy in exact_energies]
        X, Y = np.meshgrid(inverse_temps, [0, 5])
        axs[i].pcolormesh(X, Y, np.array(exact_probs_ground_state)[np.newaxis,:], vmin = 0.0, vmax = 1.0, cmap="jet", )
        axs[i].set_ylabel('Boltzmann distribution', rotation='horizontal', verticalalignment='center')
        axs[i].set_xlabel('1 / Temperature - 1 / T')
        axs[i].set_xscale('log')
        axs[i].set_yticks([])
        axs[i].set_xlim(x_min, x_max)
    fig.colorbar(heatmap, ax=axs)
    plt.savefig('{}/param_prob_ground_state_vs_temp_heatmap.png'.format(path), bbox_inches='tight')
    plt.close()

def make_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        print('Directory {} already exists'.format(dir_name))
    
    



