import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from plotly import graph_objects as go, express as px

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

def plot_interaction(interaction_fn, radius, dim_x, dim_y):
    plt.figure()
    dist = np.arange(0, euclidean_dist_2D((0, 0), (dim_x, dim_y), 1), 0.01)
    interaction_strength = [interaction_fn(r) for r in dist]
    plt.plot(dist, interaction_strength)
    plt.xlabel('Distance - r')
    plt.ylabel('Interaction Strength - J')
    ymin, ymax = plt.ylim()
    plt.vlines(radius, ymin, ymax, linestyles='dashed', colors='g')
    plt.savefig('interaction_function_radius_{}.png'.format(radius))

def plot_runtime(radii, runtimes):
    plt.figure()
    plt.plot(radii, runtimes)
    plt.xlabel('Radius')
    plt.ylabel('Runtime')
    plt.savefig('runtime.png')

def plot_spin_lattice(spin_history, lattice_X, lattice_Y, radius):
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
                ) for t in range(len(spin_vectors_history))], 
                transition=dict(duration=100),
                currentvalue=dict(font=dict(size=12), visible=True, xanchor= 'center'),
                len=1.0
            )],
            scene = dict(zaxis = dict(nticks=2, range=[-1, 1]))
        ),
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
    fig.write_html('spin_lattice_radius_{}.html'.format(radius))

def get_spin_vectors(spins):
    # x = []
    # y = []
    # z = np.zeros(len(spins))
    # u = np.zeros(len(spins))
    # v = np.zeros(len(spins))
    # w = []
    # for atom, spin in spins.items():
    #     x.append(atom[0])
    #     y.append(atom[1])
    #     w.append(spin)
    # return np.array(x), np.array(y), z, u, v, np.array(w)
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

def plot_energy_in_time(objective_history, radius):
    plt.figure()
    plt.plot(range(len(objective_history)), objective_history)
    plt.xlabel('Time Steps - t')
    plt.ylabel('Energy - E')
    plt.savefig('energy_in_time_radius_{}.png'.format(radius))

def plot_params_performance(data, y_axis, x_axis, data_title, y_axis_title, x_axis_title, radius, best_params):
    title = 'parameter selection (radius = {}, best acceptance = {}, best cooling = {}'.format(radius, best_params['acceptance'], best_params['cooling'])
    fig = px.imshow(data, title=title, labels=dict(x=x_axis_title, y=y_axis_title, color=data_title), x=x_axis, y=y_axis, color_continuous_scale='RdBu_r')
    fig.write_html('parameter_selection_radius_{}.html'.format(radius))


