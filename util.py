import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

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
        return self.vert_dict.keys()
    
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

def euclidean_dist_2D(loc1, loc2, spacing):
    coord1 = spacing * np.array(loc1)
    coord2 = spacing * np.array(loc2)
    return np.linalg.norm(coord1 - coord2)

def plot_interaction(interaction_fn, radius):
    plt.figure()
    dist = np.arange(0, 100, 0.01)
    interaction_strength = [lambda r : interaction_fn(r) for r in dist]
    plt.plot(dist, interaction_strength)
    plt.xlabel('Distance - r')
    plt.ylabel('Interaction Strength - J')
    ymin, ymax = plt.ylim()
    plt.vlines(radius, ymin, ymax, linestyles='dashed', colors='g')
    plt.show()

def plot_runtime(radius, runtime):
    plt.figure()
    plt.plot(radius, runtime)
    plt.xlabel('Radius')
    plt.ylabel('Runtime')
    plt.show()


