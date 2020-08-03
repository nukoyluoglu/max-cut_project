import csv
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

if __name__ == '__main__':
    title = 'free_random'
    P_opt = 1.0 - 1.0 / np.exp(1)
    random_select_steps_system_size = {}
    for system_size in range(4, 26):
        exact_sols_dir = '{}/exact_sols_{}'.format(title, system_size)
        r = csv.reader(open('{}/optimal_sols.csv'.format(exact_sols_dir), 'r'))
        next(r)
        for row in r:
            num_ground_states = float(row[1])
            num_total_states = float(row[2])
        p = num_ground_states / num_total_states
        P_t = {}
        def P(t, p):
            return 1 - np.power(1 - p, t)
        def total_iter(t):
            P_t = P(t, p)
            if P_t >= P_opt:
                return t
            else:
                return np.divide(t, np.abs(np.log(1 - P_t)))
        t_vals = np.arange(1, 300000, 1)
        total_iters = []
        for t in t_vals:
            total_iter_t = total_iter(t)
            if total_iter_t == float('nan'):
                break
            total_iters.append(total_iter_t)
        random_select_steps_system_size[system_size] = min(total_iters)
    system_sizes = []
    random_select_steps = []
    for system_size, random_select_step in random_select_steps_system_size.items():
        system_sizes.append(system_size)
        random_select_steps.append(random_select_step)
    plt.figure()
    plt.plot(system_sizes, random_select_steps)
    plt.xlabel('System Size')
    plt.ylabel('Steps')
    plt.ylim(bottom=0)
    plt.savefig('{}/summary/random_select_runtimes_steps_vs_system_size_radius_NA.png'.format(title))
    plt.close()
    
        
