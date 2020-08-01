from DTWA.TamLib import getLatticeCoord
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotly import graph_objects as go, express as px
import csv
from celluloid import Camera
import os

#### SYSTEM SETUP CLASSES & HELPERS ####
class Vertex(object):

    def __init__(self, id, coord):
        self.id = id
        self.coord = coord
        self.adjacent = defaultdict(int)

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def get_id(self):
        return self.id
    
    def get_coord(self):
        return self.coord
    
    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph(object):

    def __init__(self, vert_dict={}, edge_dict={}):
        self.vert_dict = vert_dict
        self.edge_dict = edge_dict
        self.num_vertices = len(self.vert_dict)

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, id, coord):
        self.num_vertices += 1
        new_vertex = Vertex(id, coord)
        self.vert_dict[id] = new_vertex
        return new_vertex

    def add_edge(self, frm, to, weight = 0.0):
        if frm not in self.vert_dict or to not in self.vert_dict:
            raise RuntimeError('Both vertices of the edge must be present')
        self.vert_dict[frm].add_neighbor(to, weight)
        self.vert_dict[to].add_neighbor(frm, weight)
        self.edge_dict[(frm, to)] = weight

    def get_vertices(self):
        return list(self.vert_dict.keys())
    
    def get_vertex(self, v):
        if v not in self.vert_dict:
            raise RuntimeError('Vertex must be present')
        return self.vert_dict[v]

    def get_coord(self, v):
        if v not in self.vert_dict:
            raise RuntimeError('Vertex must be present')
        return self.vert_dict[v].get_coord()
    
    def get_vertex_dict(self):
        return self.vert_dict
    
    def get_edge_dict(self):
        return self.edge_dict
    
    def get_edge(self, frm, to):
        if frm not in self.vert_dict or to not in self.vert_dict:
            raise RuntimeError('Both vertices of the edge must be present')
        return self.vert_dict[frm].get_weight(to)

    def get_neighbors(self, v):
        if v not in self.vert_dict:
            raise RuntimeError('Vertex must be present')
        return self.vert_dict[v].get_connections()

    def get_num_vertices(self):
        return self.num_vertices

def euclidean_dist_2D(loc1, loc2):
    return np.linalg.norm(np.array(loc1) - np.array(loc2))

#### CLASSICAL PLOT HELPERS ####

def get_lattice_coords(num_lattice_dims, lattice_dims, lattice_spacing, triangular=False):
    lattice_coords = getLatticeCoord(num_lattice_dims, lattice_dims, lattice_spacing)
    if triangular:
        lattice_coords[:, 1] += (lattice_coords[:, 0] % 2) / 2
        lattice_coords[:, 0] *= np.sqrt(3) / 2
    return lattice_coords

def plot_interaction(interaction_fn, radius, lattice_X, lattice_Y, lattice_spacing, interaction_shape, path):
    plt.figure()
    dist = np.arange(0, euclidean_dist_2D((0, 0), (lattice_X * lattice_spacing, lattice_Y * lattice_spacing)), 0.01)
    interaction_strength = [- interaction_fn(r) for r in dist]
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
    fig.tight_layout()
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
        fig, ax1 = plt.subplots(figsize=(12, 9))
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
        fig.tight_layout()
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

def get_spin_lattice(spins, prob):
    x_up = []
    y_up = []
    x_down = []
    y_down = []
    for atom, spin in spins.items():
        atom_x, atom_y = prob.get_coord(atom)
        partition_x, partition_y = (x_up, y_up) if spin == 1 else (x_down, y_down)
        partition_x.append(atom_x)
        partition_y.append(atom_y)
    return np.array(x_up), np.array(y_up), np.array(x_down), np.array(y_down)

def plot_spin_lattice(prob, spin_history, energy_history, radius, lattice_X, lattice_Y, interaction_shape, triangular, path):
    fig = plt.figure()
    plt.title('L = {}, {}, radius = {}'.format(lattice_X, interaction_shape, radius))
    camera = Camera(fig)
    # plot 1 spin configuration per temperature
    for t in range(0, len(spin_history), 1000):
        x_up, y_up, x_down, y_down = get_spin_lattice(spin_history[t], prob)
        plt.scatter(x_up, y_up, s=20**2, c='red')
        plt.scatter(x_down, y_down, s=20**2, c='blue')
        plt.text(0.05, 0.95, 'E = {}'.format(round(energy_history[t], 1)), transform=fig.transFigure, verticalalignment='top')
        plt.gca().set_aspect('equal', adjustable='box')
        camera.snap()
    animation = camera.animate()
    animation.save('{}/spin_lattice.gif'.format(path), writer = 'imagemagick')
    plt.close()

def plot_energy_temp_vs_step(ave_energy_history, ave_temp_history, radius, system_size, interaction_shape, init_temp, cool_rate, path, exact_best_energy):
    ave_energy_history = np.array([ave_energy_history[t][0] for t in range(len(ave_energy_history))])
    # energy_error_history = np.array([ave_energy_history[t][1] for t in range(len(ave_energy_history))])
    ave_temp_history = np.array([ave_temp_history[t] for t in range(len(ave_temp_history))])
    if len(ave_energy_history) != len(ave_temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
    fig, ax1 = plt.subplots()
    title = 'L = {}, {}, radius = {}, T_0 = {}, r = {}\nE_min = {}'.format(system_size, interaction_shape, radius, init_temp, cool_rate, min(ave_energy_history))
    color = 'tab:blue'
    ax1.set_xlabel('Time Steps - t')
    ax1.set_ylabel('Temperature - T', color=color)
    ax1.plot(range(len(ave_temp_history)), ave_temp_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    if exact_best_energy:
        ave_energy_history -= exact_best_energy
        title += ', exact E_min = {}'.format(exact_best_energy)
        ax2.set_ylabel('Energy Difference from Ground State - \u0394E', color=color)
        ax2.set_yscale('log')
    ax2.plot(range(len(ave_energy_history)), ave_energy_history, color=color)
    ax2.set_ylabel('Energy - E', color=color)
    # ax2.errorbar(range(len(ave_energy_history)), ave_energy_history, yerr=energy_error_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(title)
    fig.tight_layout()
    plt.savefig('{}/energy_temp_vs_step_T0_{}_r_{}.png'.format(path, init_temp, cool_rate))
    plt.close()

def plot_prob_ground_state_temp_vs_step(prob_ground_state_hist, ave_temp_hist, optimal_t, optimal_step, radius, num_particles, interaction_shape, init_temp, cool_rate, path):
    prob_ground_state_history = np.array([prob_ground_state_hist[t][0] for t in range(len(prob_ground_state_hist))])
    prob_ground_state_errors = np.array([prob_ground_state_hist[t][1] for t in range(len(prob_ground_state_hist))])
    ave_temp_history = np.array([ave_temp_hist[t] for t in range(len(ave_temp_hist))])
    if len(prob_ground_state_history) != len(ave_temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
    fig, ax1 = plt.subplots()
    title = 'N = {}, {}, radius = {}, T_0 = {}, r = {}'.format(num_particles, interaction_shape, radius, init_temp, cool_rate)
    color = 'tab:red'
    ax1.set_xlabel('Time Steps - t')
    ax1.set_ylabel('Probability of Reaching Ground State - P(t)', color=color)
    ax1.errorbar(range(len(prob_ground_state_history)), prob_ground_state_history, yerr=prob_ground_state_errors, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Temperature - T', color=color)
    ax2.plot(range(len(ave_temp_history)), ave_temp_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    if optimal_t:
        plt.axvline(optimal_t, color='g', label='T = {}, M * T = {}, P(T) = {}'.format(optimal_t, optimal_step, prob_ground_state_history[optimal_t]))
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    plt.savefig('{}/prob_ground_state_temp_vs_step_T0_{}_r_{}.png'.format(path, init_temp, cool_rate))
    plt.close()

def plot_entropy_temp_vs_step(entropy_hist, ave_temp_hist, optimal_t, step, radius, num_particles, interaction_shape, init_temp, cool_rate, path):
    entropy_history = np.array([entropy_hist[t] for t in range(len(entropy_hist))])
    ave_temp_history = np.array([ave_temp_hist[t] for t in range(len(ave_temp_hist))])
    if len(entropy_history) != len(ave_temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
    fig, ax1 = plt.subplots()
    title = 'N = {}, {}, radius = {}, T_0 = {}, r = {}'.format(num_particles, interaction_shape, radius, init_temp, cool_rate)
    color = 'tab:red'
    ax1.set_xlabel('Time Steps - t')
    ax1.set_ylabel('Entropy - S(t)', color=color)
    ax1.plot(range(len(entropy_history)), entropy_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Temperature - T', color=color)
    ax2.plot(range(len(ave_temp_history)), ave_temp_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    if optimal_t:
        plt.axvline(optimal_t, color='g', label='T = {}, S(T) = {}'.format(optimal_t, entropy_history[optimal_t]))
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    plt.savefig('{}/entropy_temp_vs_step_T0_{}_r_{}.png'.format(path, init_temp, cool_rate))
    plt.close()

def plot_step_optimization(optimize_step, ave_temp_history, optimal_t, optimal_step, radius, num_particles, interaction_shape, init_temp, cool_rate, path):
    optimize_step = np.array([optimize_step[t] for t in range(1, len(optimize_step) + 1)])
    ave_temp_history = np.array([ave_temp_history[t] for t in range(1, len(ave_temp_history))])
    if len(optimize_step) != len(ave_temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
    fig, ax1 = plt.subplots()
    title = 'N = {}, {}, radius = {}, T_0 = {}, r = {}'.format(num_particles, interaction_shape, radius, init_temp, cool_rate)
    color = 'tab:red'
    ax1.set_xlabel('Time Steps - t')
    ax1.set_ylabel('Quantity to Optimize - t / |log(1 - P(t)|)', color=color)
    ax1.plot(range(1, len(optimize_step) + 1), optimize_step, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Temperature - T', color=color)
    ax2.plot(range(1, len(ave_temp_history) + 1), ave_temp_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    if optimal_t:
        plt.axvline(optimal_t, color='g', label='T = {}, M * T = {}'.format(optimal_t, optimal_step))
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

def plot_params_energy_vs_temp(ave_energy_vs_temp_by_params, best_params, radius, system_size, interaction_shape, path, exact_best_energy=None, boltzmann_temps=None, boltzmann_energies=None, exact_path=None):
    for init_temp, ave_energy_vs_temp_by_cool_rate in ave_energy_vs_temp_by_params.items():
        plt.figure()
        plt.title('N = {}, {}, radius = {}'.format(system_size, interaction_shape, radius))
        x_min = float('inf')
        x_max = 0
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
                x_min = min(1.0 / temp, x_min)
                x_max = max(1.0 / temp, x_max)
            label = 'T_0 = {}, r = {}, T_f = {}'.format(init_temp, round(cool_rate, 4), np.format_float_scientific(1.0 / inverse_temps[-1], precision=1))
            if init_temp == best_params['init_temp'] and cool_rate == best_params['cool_rate']:
                label += ', optimal'
            p = plt.plot(inverse_temps, ave_energies, label=label)
            plt.errorbar(inverse_temps, ave_energies, yerr=errors, alpha=0.1, ecolor=p[-1].get_color())
        if exact_best_energy:
            exact_ave_energies = boltzmann_energies[(x_min <= boltzmann_temps) & (boltzmann_temps <= x_max)]
            inverse_temps = boltzmann_temps[(x_min <= boltzmann_temps) & (boltzmann_temps <= x_max)]
            plt.plot(inverse_temps, exact_ave_energies, label='Boltzmann distribution', linestyle='dashed')
        plt.legend()
        plt.xscale('log')
        plt.xlabel('1 / Temperature - 1 / T')
        plt.ylabel('Energy - E')
        plt.savefig('{}/param_energy_vs_temp_T0_{}.png'.format(path, init_temp))
        plt.close()

def plot_params_energy_vs_temp_heatmap(ave_energy_vs_temp_by_params, best_params, radius, system_size, interaction_shape, path, exact_best_energy=None, exact_path=None):
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
        exact_ave_energies = np.array([boltzmann_dist(all_energies, 1.0 / inverse_temp) for inverse_temp in inverse_temps])
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
    if exact_best_energy:
        return inverse_temps, exact_ave_energies

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

#### QUANTUM PLOT HELPERS ####

# returns binary representation of length N for each state
def get_states_str(dims):
    N = np.prod(dims)
    num_states = 2 ** N
    states = []
    for state_id in range(num_states):
        state = '{0:b}'.format(state_id)
        state = '0' * (N - len(state)) + state
        # matrix representation of spin lattice
        states.append(state)
    return states

# returns sum of angles corresponding to Ising Hamiltonian
def get_beta_sum(angles): 
    return np.sum(get_betas(angles))

# returns angles corresponding to Ising Hamiltonian
def get_betas(angles):
    return angles[::2]

def get_beta_indices(alpha):
    return 2 * np.array(range(alpha))


def plot_state_probs(state_probs_t, energy_t, dims, interaction_shape, radius, alpha, path):
    states = np.array(get_states_str(dims))
    fig = plt.figure(figsize=(9, 12))
    plt.title('L = {}, {}, radius = {}, alpha = {}'.format(dims[0], interaction_shape, radius, alpha))
    plt.xlabel('State')
    plt.ylabel('Probability')
    camera = Camera(fig)
    for t in range(0, len(state_probs_t)):
        state_probs = state_probs_t[t]
        max_prob = 0
        ground_states = []
        for state_id in range(len(state_probs)):
            state_prob = state_probs[state_id]
            if state_prob - max_prob > 1e-10:
                ground_states = [state_id]
                max_prob = state_prob
            elif abs(state_prob - max_prob) < 1e-10:
                ground_states.append(state_id)
        plt.bar(range(len(state_probs)), state_probs)
        plt.xticks(ground_states, labels=states[ground_states], rotation='vertical')
        plt.ylim(bottom=0, top=1)
        plt.text(0.05, 0.95, '<H> = {}'.format(round(energy_t[t], 1)), transform=fig.transFigure, verticalalignment='top')
        # fig.tight_layout()
        camera.snap()
    animation = camera.animate()
    animation.save('{}/state_probs_alpha_{}.gif'.format(path, alpha), writer = 'imagemagick')
    plt.close()
    # sample spin lattice in ground state
    print(np.reshape(np.array([int(spin) for spin in states[ground_states[0]]]), dims))

def plot_ground_state_fidelities_vs_time(state_probs_t_alpha, ground_states_id, angles_alpha, dims, interaction_shape, radius, path):
    fig = plt.figure()
    plt.title('L = {}, {}, radius = {}'.format(dims[0], interaction_shape, radius))
    plt.xlabel('Time (t)')
    plt.ylabel('Probability of Ground State (P)')
    for alpha, (state_probs_t, t) in state_probs_t_alpha.items():
        ground_state_prob_t = [np.sum(state_probs[ground_states_id]) for state_probs in state_probs_t]
        angles = angles_alpha[alpha]
        beta_sum = get_beta_sum(angles)
        plt.plot(t, ground_state_prob_t, label='alpha = {}, beta_sum = {}'.format(alpha, beta_sum))
    plt.ylim(bottom=0, top=1)
    plt.legend()
    fig.tight_layout()
    plt.savefig('{}/ground_state_fidelities_vs_time.png'.format(path), bbox_inches='tight')
    plt.close()

def plot_final_ground_state_fidelities_vs_alpha(state_probs_t_alpha, ground_states_id, dims, interaction_shape, radius, path):
    alphas = []
    final_ground_state_probs = []
    for alpha, (state_probs_t, t) in state_probs_t_alpha.items():
        alphas.append(alpha)
        final_ground_state_probs.append(np.sum(state_probs_t[-1][ground_states_id]))
    fig = plt.figure()
    plt.title('L = {}, {}, radius = {}'.format(dims[0], interaction_shape, radius))
    plt.xlabel('Circuit Depth (alpha)')
    plt.ylabel('Probability of Ground State (P)')
    plt.plot(alphas, final_ground_state_probs)
    plt.ylim(bottom=0, top=1)
    fig.tight_layout()
    plt.savefig('{}/ground_state_fidelities_vs_alpha.png'.format(path), bbox_inches='tight')
    plt.close()

def plot_final_ground_state_fidelities_vs_beta_sum(state_probs_t_alpha, ground_states_id, angles_alpha, dims, interaction_shape, radius, path):
    beta_sums = []
    final_ground_state_probs = []
    for alpha, (state_probs_t, t) in state_probs_t_alpha.items():
        final_ground_state_probs.append(np.sum(state_probs_t[-1][ground_states_id]))
        angles = angles_alpha[alpha]
        beta_sum = get_beta_sum(angles)
        beta_sums.append(beta_sum)
    fig = plt.figure()
    plt.title('L = {}, {}, radius = {}'.format(dims[0], interaction_shape, radius))
    plt.xlabel('Integrated Interaction Strength * Time (beta_sum)')
    plt.ylabel('Probability of Ground State (P)')
    plt.plot(beta_sums, final_ground_state_probs)
    plt.ylim(bottom=0, top=1)
    fig.tight_layout()
    plt.savefig('{}/ground_state_fidelities_vs_beta_sum.png'.format(path), bbox_inches='tight')
    plt.close()

def plot_VQE_runtimes_beta_sums_vs_alpha(VQE_runtimes_alpha, angles_alpha, dims, interaction_shape, radius, path):
    alphas = []
    VQE_runtimes = []
    beta_sums = []
    for alpha, VQE_runtime in VQE_runtimes_alpha.items():
        alphas.append(alpha)
        VQE_runtimes.append(VQE_runtime)
        angles = angles_alpha[alpha]
        beta_sum = get_beta_sum(angles)
        beta_sums.append(beta_sum)
    fig, ax1 = plt.subplots()
    plt.title('L = {}, {}, radius = {}'.format(dims[0], interaction_shape, radius))
    ax1.set_xlabel('Circuit Depth (alpha)')
    color = 'tab:red'
    ax1.set_ylabel('VQE Optimization Runtime (s)', color=color)
    ax1.plot(alphas, VQE_runtimes, color=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Integrated Interaction Strength * Time (beta_sum)', color=color)
    ax2.plot(alphas, beta_sums, color=color)
    fig.tight_layout()
    plt.savefig('{}/VQE_runtimes_beta_sums_vs_alpha.png'.format(path), bbox_inches='tight')
    plt.close()

def plot_energy_vs_time(energy_t, t, dims, interaction_shape, radius, alpha, path):
    states = np.array(get_states_str(dims))
    fig = plt.figure()
    plt.title('L = {}, {}, radius = {}, alpha = {}'.format(dims[0], interaction_shape, radius, alpha))
    plt.xlabel('Time (t)')
    plt.ylabel('Expectation Value of Hamiltonian (<H>)')
    plt.plot(t, energy_t)
    plt.savefig('{}/expH_in_time_alpha_{}.png'.format(path, alpha), bbox_inches='tight')
    plt.close()

def plot_energy_hamiltonians_vs_time(exp_t, H_t, B_t, t, dims, interaction_shape, radius, alpha, path):
    states = np.array(get_states_str(dims))
    fig, ax1 = plt.subplots()
    plt.title('L = {}, {}, radius = {}, alpha = {}'.format(dims[0], interaction_shape, radius, alpha))
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Expectation Value of Ising Hamiltonian (<H>)')
    ax1.plot(t, exp_t)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Ratio of Total Hamiltonian')
    ax2.step(t, H_t, alpha=0.5, label='Ising Hamiltonian (H)')
    ax2.step(t, B_t, alpha=0.5, label='Reference Hamiltonian (B)')
    ax2.legend()
    fig.tight_layout()
    plt.savefig('{}/<H>_H_B_in_time_alpha_{}.png'.format(path, alpha), bbox_inches='tight')
    plt.close()

def plot_eigval_crossing(eig_i_t, t, dims, interaction_shape, radius, path):
    eigs_t = defaultdict(list)
    for eig_i in eig_i_t:
        eig_i.sort()
        for i in range(len(eig_i)):
            eigs_t[i].append(eig_i[i])
    fig = plt.figure()
    plt.title('L = {}, {}, radius = {}'.format(dims[0], interaction_shape, radius))
    plt.xlabel('Time (t)')
    plt.ylabel('Eigenvalues Total Hamiltonian (H + B)')
    for i, eig_t in eigs_t.items():
        plt.plot(t, eig_t, label=i)
    plt.legend()
    plt.savefig('{}/eigval_crossing_in_time.png'.format(path), bbox_inches='tight')
    plt.close()
