import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotly import graph_objects as go, express as px
from collections import defaultdict
import csv
from celluloid import Camera

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

def plot_interaction(interaction_fn, radius, lattice_X, lattice_Y, interaction_shape):
    plt.figure()
    dist = np.arange(0, euclidean_dist_2D((0, 0), (lattice_X, lattice_Y), 1), 0.01)
    interaction_strength = [interaction_fn(r) for r in dist]
    plt.plot(dist, interaction_strength)
    plt.xlabel('Distance - r')
    plt.ylabel('Interaction Strength - J')
    plt.axvline(radius, linestyle='dashed', color='g')
    plt.savefig('{}_{}x{}/interaction_function_radius_{}.png'.format(interaction_shape, lattice_X, lattice_Y, radius))
    plt.close()

def plot_runtimes_steps_vs_radius(algorithm_performance_by_radius, lattice_X, lattice_Y, interaction_shape):
    radii = []
    runtimes = []
    steps = []
    col_info = {}
    for radius, (solution, params) in algorithm_performance_by_radius.items():
        radii.append(radius)
        runtimes.append(solution['ave_runtime'])
        steps.append(solution['ave_step'])
        col_info[radius] = 'radius = {}\nE_f = {}\nT_0 = {}, r = {}'.format(radius, round(solution['ave_final_energy'], 1), params['init_temp'], params['cool_rate'])
    fig, ax1 = plt.subplots()
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
    fig.suptitle('L = {}, {}'.format(lattice_X, interaction_shape))
    plt.tight_layout()
    plt.savefig('{}_{}x{}/runtimes_steps_vs_radius.png'.format(interaction_shape, lattice_X, lattice_Y))
    plt.close()

def plot_runtimes_steps_vs_system_size(algorithm_performance_by_system, interaction_shape):
    runtimes_steps_vs_system_size_by_radius = defaultdict(dict)
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
            runtimes.append(solution['ave_runtime'])
            steps.append(solution['ave_step'])
            col_info[system_size] = 'L = {}\nE_f = {}\nT_0 = {}, r = {}'.format(system_size, round(solution['ave_final_energy'], 1), params['init_temp'], params['cool_rate'])
        fig, ax1 = plt.subplots()
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
        fig.suptitle('{}, radius = {}'.format(interaction_shape, radius))
        plt.tight_layout()
        plt.savefig('runtimes_steps_vs_system_size_{}_radius_{}.png'.format(interaction_shape, radius))
        plt.close()

def plot_num_ground_states_vs_system_size(num_ground_states_by_system, interaction_shape):
    num_ground_states_vs_system_size_by_radius = defaultdict(dict)
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
        plt.savefig('num_ground_states_vs_system_size_{}_radius_{}.png'.format(interaction_shape,radius))
        plt.close()

def get_spin_lattice(spins, lattice_X, lattice_Y):
    lattice = np.zeros((lattice_X, lattice_Y))
    for atom, spin in spins.items():
        lattice[atom[0]][atom[1]] = spin
    return lattice

def get_spin_lattice_xyz(spins, lattice_X, lattice_Y):
    x_data, y_data = np.meshgrid(range(lattice_X), range(lattice_Y))
    z_data = []
    for x, y in zip(x_data.flatten(), y_data.flatten()):
        if (x, y) in spins.keys():
            z_data.append(spins[(x, y)])
        else:
            z_data.append(0)
    return x_data.flatten(), y_data.flatten(), z_data

def get_tri_spin_lattice(spins):
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

def animation_layout(spin_history):
    layout=go.Layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                    method="animate",
                    args=[None, 
                        dict(frame=dict(duration=0.1, redraw=True), 
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode='immediate'
                        )
                    ]
                ),
                dict(label="Stop",
                    method="animate",
                    args=[None, 
                        dict(frame=dict(duration=0, redraw=False), 
                            transition=dict(duration=0),
                            mode='immediate'
                        )
                    ]
                )
            ]
        )],
        sliders=[dict(
            steps=[dict(method='animate',
                args=[[str(t)], 
                    dict(mode='immediate', frame=dict(duration=0.1, redraw=True), transition=dict(duration=0))
                ], label=str(t)
            ) for t in range(len(spin_history))], 
            transition=dict(duration=100),
            currentvalue=dict(font=dict(size=12), visible=True, xanchor= 'center'),
            len=1.0
        )],
        # scene=dict(zaxis = dict(nticks=2, range=[-1, 1]))
        scene=dict(aspectmode='data')
    )
    return layout


def plot_spin_lattice(spin_history, radius, lattice_X, lattice_Y, interaction_shape, filename=None, triangular=False, fancy=False):
    if not triangular:
        # storage issue
        # TODO: plot 2 energies per temperature
        fig = plt.figure()
        plt.title('L = {}, {}, radius = {}'.format(lattice_X, interaction_shape, radius))
        im = plt.imshow(get_spin_lattice(spin_history[0], lattice_X, lattice_Y), cmap='bwr', animated=True)
        def update_fig(t):
            data = get_spin_lattice(spin_history[t], lattice_X, lattice_Y)
            im.set_array(data)
            return im
        ani = animation.FuncAnimation(fig, update_fig, frames=range(len(spin_history)))
        if not filename:
            filename = '{}_{}x{}/spin_lattice_radius_{}.gif'.format(interaction_shape, lattice_X, lattice_Y, radius)
        else:
            filename += '.gif'
        ani.save(filename, writer='imagemagick', fps=30)
    else:
        fig = plt.figure()
        plt.title('L = {}, {}, radius = {}'.format(lattice_X, interaction_shape, radius))
        x_up, y_up, x_down, y_down = get_tri_spin_lattice(spin_history[0])
        plt.plot(x_up, y_up, 'ro')
        plt.plot(x_down, y_down, 'bo')
        plt.gca().set_aspect('equal', adjustable='box')
        if not filename:
            filename = '{}_{}x{}/spin_lattice_radius_{}.png'.format(interaction_shape, lattice_X, lattice_Y, radius)
        else:
            filename += '.png'
        plt.savefig(filename)
        plt.close()

    # spin_lattice_history = [get_spin_lattice_xyz(spins, lattice_X, lattice_Y) for spins in spin_history]
    # x, y, z = spin_lattice_history[0]
    # fig = go.Figure(
    #     data=[go.Heatmap(x=x, y=y, z=z, colorscale='Bluered', showscale=False)],
    #     frames=[go.Frame(
    #         data=[go.Heatmap(x=x, y=y, z=z, colorscale='Bluered', showscale=False)], name=str(t)
    #     ) for t, (x, y, z) in enumerate(spin_lattice_history)],
    #     layout=animation_layout(spin_history)
    # )
    # if filename:
    #         fig.write_html(filename)
    # else:
    #     fig.write_html('spin_lattice_radius_{}.html'.format(radius))

    if fancy:
        spin_vectors_history = [get_spin_vectors(spins) for spins in spin_history]
        u_x, u_y, u_z, u_u, u_v, u_w, d_x, d_y, d_z, d_u, d_v, d_w = spin_vectors_history[0]
        fig = go.Figure(
            data=[
                go.Cone(x=u_x, y=u_y, z=u_z, u=u_u, v=u_v, w=u_w,
                    anchor='tail', sizeref=1.5, colorscale=[[0, 'red'], [1, 'red']], showscale = False
                ),
                go.Cone(x=d_x, y=d_y, z=d_z, u=d_u, v=d_v, w=d_w,
                    anchor='tail', sizeref=1.5, colorscale=[[0, 'blue'], [1, 'blue']], showscale = False
                )
            ], 
            layout=animation_layout(spin_history),
            frames=[go.Frame(
                data=[
                    go.Cone(x=u_x, y=u_y, z=u_z, u=u_u, v=u_v, w=u_w, 
                        anchor='tail', sizemode='absolute', colorscale=[[0, 'red'], [1, 'red']], showscale = False
                    ),
                    go.Cone(x=d_x, y=d_y, z=d_z, u=d_u, v=d_v, w=d_w, 
                        anchor='tail', sizemode='absolute', colorscale=[[0, 'blue'], [1, 'blue']], showscale = False
                    )
                ],
                name=str(t)
            ) for t, (u_x, u_y, u_z, u_u, u_v, u_w, d_x, d_y, d_z, d_u, d_v, d_w) in enumerate(spin_vectors_history)]
        )
        if filename:
            fig.write_html(filename)
        else:
            fig.write_html('spin_lattice_radius_{}.html'.format(radius))

def get_spin_vectors(spins):
    u_x = []
    u_y = []
    u_z = []
    u_u = []
    u_v = []
    u_w = []
    d_x = []
    d_y = []
    d_z = []
    d_u = []
    d_v = []
    d_w = []
    for atom, spin in spins.items():
        if spin > 0:
            u_x.append(atom[0])
            u_y.append(atom[1])
            u_z.append(0)
            u_u.append(0)
            u_v.append(0)
            u_w.append(spin)
        else:
            d_x.append(atom[0])
            d_y.append(atom[1])
            d_z.append(0)
            d_u.append(0)
            d_v.append(0)
            d_w.append(spin)
    return u_x, u_y, u_z, u_u, u_v, u_w, d_x, d_y, d_z, d_u, d_v, d_w

def get_atoms(spins):
    x = []
    y = []
    z = np.zeros(len(spins))
    for atom in spins.keys():
        x.append(atom[0])
        y.append(atom[1])
    return np.array(x), np.array(y), z

def plot_energy_in_time(energy_history, radius, lattice_X, lattice_Y, interaction_shape):
    plt.figure()
    plt.plot(range(len(energy_history)), energy_history)
    plt.xlabel('Time Steps - t')
    plt.ylabel('Energy - E')
    plt.savefig('{}_{}x{}/energy_in_time_radius_{}.png'.format(interaction_shape, lattice_X, lattice_Y, radius))
    plt.close()

def plot_energy_temp_in_time(energy_history, temp_history, radius, lattice_X, lattice_Y, interaction_shape):
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
    plt.savefig('{}_{}x{}/energy_temp_in_time_radius_{}.png'.format(interaction_shape, lattice_X, lattice_Y, radius))
    plt.close()

def plot_params_energy_temp_vs_step(ave_energy_history, ave_temp_history, radius, lattice_X, lattice_Y, interaction_shape, init_temp, cool_rate):
    if len(ave_energy_history) != len(ave_temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
    fig, ax1 = plt.subplots()
    plt.title('L = {}, {}, radius = {}, T_0 = {}, r = {}'.format(lattice_X, interaction_shape, radius, init_temp, cool_rate))
    color = 'tab:red'
    ax1.set_xlabel('Time Steps - t')
    ax1.set_ylabel('Energy - E', color=color)
    ax1.plot(range(len(ave_energy_history)), list(ave_energy_history.values()), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Temperature - T', color=color)
    ax2.plot(range(len(ave_temp_history)), list(ave_temp_history.values()), color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.savefig('{}_{}x{}/energy_temp_in_time_radius_{}_T0_{}_r_{}.png'.format(interaction_shape, lattice_X, lattice_Y, radius, init_temp, cool_rate))
    plt.close()

def plot_temporary(energy_history, temp_history, radius, lattice_X, lattice_Y, interaction_shape, init_temp, cool_rate, t):
    if len(energy_history) != len(temp_history):
        raise RuntimeError('Length of energy and temperature histories must match')
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
    plt.savefig('{}_{}x{}/{}_energy_temp_in_time_radius_{}_T0_{}_r_{}.png'.format(interaction_shape, lattice_X, lattice_Y, t, radius, init_temp, cool_rate))
    plt.close()

# def plot_params_steps(data, y_axis, x_axis, data_title, y_axis_title, x_axis_title, radius, best_params):
#     title = 'parameter selection (radius = {}, best initial temperature = {}, best cooling rate = {}'.format(radius, best_params['init_temp'], best_params['cool_rate'])
#     fig = px.imshow(data, title=title, labels=dict(x=x_axis_title, y=y_axis_title, color=data_title), x=x_axis, y=y_axis, color_continuous_scale='RdBu_r')
#     fig.write_html('param_heatmap_radius_{}.html'.format(radius))

def boltzmann_dist(all_states_energy, temp):
    num = 0.0
    denom = 0.0
    for state_energy in all_states_energy:
        factor = np.exp(- state_energy / temp)
        num += state_energy * factor
        denom += factor
    return num / denom

def plot_params_energy_vs_temp(ave_energy_vs_temp_by_params, best_params, radius, lattice_X, lattice_Y, interaction_shape, exact_best_energy=None):
    for init_temp, ave_energy_vs_temp_by_cool_rate in ave_energy_vs_temp_by_params.items():
        plt.figure()
        plt.title('L = {}, {}, radius = {}'.format(lattice_X, interaction_shape, radius))
        # x_right_lim = 0
        for cool_rate, ave_energy_vs_temp in ave_energy_vs_temp_by_cool_rate.items():
            inverse_temps = []
            ave_energies = []
            errors = []
            for temp, (ave_energy, error) in ave_energy_vs_temp.items():
                inverse_temps.append(1.0 / temp)
                ave_energies.append(ave_energy)
                errors.append(error)
            # temp_cutoff_index = - min(5, len(inverse_temps))
            # x_right_lim = max(inverse_temps[temp_cutoff_index], x_right_lim)
            label = 'T_0 = {}, r = {}, T_f = {}'.format(init_temp, round(cool_rate, 1), np.format_float_scientific(1.0 / inverse_temps[-1], precision=1))
            if init_temp == best_params['init_temp'] and cool_rate == best_params['cool_rate']:
                label += ', optimal'
            p = plt.plot(inverse_temps, ave_energies, label=label)
            plt.errorbar(inverse_temps, ave_energies, yerr=errors, alpha=0.5, ecolor=p[-1].get_color())
        if exact_best_energy:
            plt.axhline(exact_best_energy, label='brute force')
            all_energies = []
            r = csv.reader(open('{}_exact_sols_{}x{}/energies_radius_{}.csv'.format(interaction_shape, lattice_X, lattice_Y, radius), 'r'))
            next(r)
            for row in r:
                all_energies.append(float(row[0]))
            exact_ave_energies = [boltzmann_dist(all_energies, 1.0 / inverse_temp) for inverse_temp in inverse_temps]
            plt.plot(inverse_temps, exact_ave_energies, label='Boltzmann distribution')
        plt.legend()
        plt.xscale('log')
        # plt.xlim(right=x_right_lim)
        plt.xlabel('1 / Temperature - 1 / T')
        plt.ylabel('Energy - E')
        plt.savefig('{}_{}x{}/param_energy_vs_temp_radius_{}_T0_{}.png'.format(interaction_shape, lattice_X, lattice_Y, radius, init_temp))
        plt.close()

def plot_params_energy_vs_temp_heatmap(ave_energy_vs_temp_by_params, best_params, radius, lattice_X, lattice_Y, interaction_shape, exact_best_energy=None):
    num_subplots = sum(len(n) for n in ave_energy_vs_temp_by_params.values())
    if exact_best_energy:
        num_subplots += 1
    fig, axs = plt.subplots(num_subplots, 1, sharex=True, gridspec_kw=dict(hspace=0))
    fig.suptitle('L = {}, {}, radius = {}'.format(lattice_X, interaction_shape, radius))
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
        x_edges, y_edges = np.meshgrid(x, [0, 1])
        heatmap = axs[i].pcolormesh(x_edges, y_edges, np.array(h)[np.newaxis,:], vmin = h_min, vmax = h_max, cmap="jet")
        label = 'T_0 = {}, r = {}, T_f = {}'.format(T_0, round(r, 1), np.format_float_scientific(1.0 / x[-1], precision=1))
        if T_0 == best_params['init_temp'] and r == best_params['cool_rate']:
            label += ', optimal'
        axs[i].set_ylabel(label, rotation='horizontal')
        axs[i].set_xlabel('1 / Temperature - 1 / T')
        axs[i].set_xscale('log')
        axs[i].set_yticks([])
        axs[i].set_xlim(x_min, x_max)
    i = len(param_data)
    if exact_best_energy:
        all_energies = []
        r = csv.reader(open('{}_exact_sols_{}x{}/energies_radius_{}.csv'.format(interaction_shape, lattice_X, lattice_Y, radius), 'r'))
        next(r)
        for row in r:
            all_energies.append(float(row[0]))
        inverse_temps = np.logspace(np.floor(np.log10(x_min)), np.ceil(np.log10(x_max)))
        exact_ave_energies = [boltzmann_dist(all_energies, 1.0 / inverse_temp) for inverse_temp in inverse_temps]
        X, Y = np.meshgrid(inverse_temps, [0, 1])
        heatmap = axs[i].pcolormesh(X, Y, np.array(exact_ave_energies)[np.newaxis,:], vmin = h_min, vmax = h_max, cmap="jet", )
        axs[i].set_ylabel('Boltzmann distribution', rotation='horizontal')
        axs[i].set_xlabel('1 / Temperature - 1 / T')
        axs[i].set_xscale('log')
        axs[i].set_yticks([])
        axs[i].set_xlim(x_min, x_max)
    fig.colorbar(heatmap, ax=axs)
    # fig.tight_layout()
    plt.savefig('{}_{}x{}/param_energy_vs_temp_radius_{}_heatmap.png'.format(interaction_shape, lattice_X, lattice_Y, radius))
    plt.close()
    
    



