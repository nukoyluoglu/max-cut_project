import classical_algorithms
import experiment
import maxcut
import util
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from quspin.tools.measurements import obs_vs_time
# from DTWA.DTWA_Lib import genUniformConfigs, IsingEvolve
from scipy.optimize import minimize, Bounds
import numpy as np
import time
import os

# returns expected value of operator
def expectation(psi, op):
    return (np.dot(np.conjugate(psi), op @ psi) / np.dot(np.conjugate(psi), psi))

# returns probability distribution over states
def state_probs(psi):
    # normalize
    psi /= np.linalg.norm(psi)
    return (np.conjugate(psi) * psi)

def VQE_optimization_fn(psi_0, H, B, alpha, DTWA, penalize):
    def fn(angles):
        # if DTWA:
        #     final_exp_H = QAOA_evolution_DTWA(DTWA, angles, np.linspace(1., 1., 1))
        # else:
        final_exp_H = QAOA_evolution(psi_0, H, B, angles, np.linspace(1., 1., 1))
        if penalize:
            return final_exp_H + 0.1 * util.get_beta_sum(angles)
        return final_exp_H
    return fn

def VQE(psi_0, H, B, n_params, DTWA, param_bounds=None, penalize=False):
    if n_params == 1:
        init_params = 1.
    else:
        init_params = np.ones(n_params)
    optimization_fn = VQE_optimization_fn(psi_0, H, B, alpha, DTWA, penalize)
    classical_optimization = minimize(optimization_fn, init_params, bounds=param_bounds)
    if classical_optimization.get('success'):
        return classical_optimization.get('x')
    raise RuntimeError('Optimization failed')

def QAOA_evolution(psi_0, H, B, angles, t_it, store_states=False):
    psi = psi_0
    energy = obs_vs_time(np.reshape(psi_0, (-1, 1)), [0], dict(energy=H))['energy'][0]
    if store_states: 
        psi_t = np.array([psi])
        energy_t = np.array([energy])
        H_t = np.zeros(1)
        B_t = np.zeros(1)
        t = np.zeros(1)
    for it in range(len(angles)):
        t_op = angles[it] * t_it
        op = H if it % 2 == 0 else B
        psi_it = op.evolve(psi, 0, t_op)
        obs_it = obs_vs_time(psi_it, t_op, dict(energy=H))
        psi_it = np.transpose(psi_it)
        energy_it = obs_it['energy']
        psi = psi_it[-1]
        energy = energy_it[-1]
        if store_states:
            psi_t = np.concatenate((psi_t, psi_it))
            energy_t = np.concatenate((energy_t, energy_it))
            H_t = np.concatenate((H_t, [angles[it] if it % 2 == 0 else 0 for _ in t_it]))
            B_t = np.concatenate((B_t, [angles[it] if it % 2 == 1 else 0 for _ in t_it]))
            t = np.concatenate((t, it + t_it))
    return (psi_t, energy_t, H_t, B_t, t) if store_states else energy

##### THIS ONLY DOES OPERATIONS ON Z SPINS - CANNOT SIMULATE REFERENCE HAMILTONIAN
# def QAOA_evolution_DTWA(DTWA, angles, t_it):
#     init_configs = genUniformConfigs(DTWA['N'],DTWA['nt'], axis='x')
#     psi_0 = np.mean(init_configs, axis=1) 
#     configs, psi = init_configs, psi_0
#     psi_0_z = psi_0[:, 2]
#     for it in range(len(angles)):
#         t_evol = angles[it] * t_it
#         Jfunc = experiment.step_fn(1) if it % 2 == 0 else ???
#         configs, psi_it = IsingEvolve(configs, t_evol, -1., alpha=6, Jfunc=Jfunc, coord=DTWA['coord'])
#         psi_it_z = psi_it[:, :, 2]
#         # double since spins are +/- 1/2
#         energy_it = [2 * () for psi_z in psi_it_z]


#         psi_it = op.evolve(psi, 0, t_op)
#         obs_it = obs_vs_time(psi_it, t_op, dict(energy=H))
#         psi_it = np.transpose(psi_it)
#         energy_it = obs_it['energy']
#         psi = psi_it[-1]
#         energy = energy_it[-1]
#         if store_states:
#             psi_t = np.concatenate((psi_t, psi_it))
#             energy_t = np.concatenate((energy_t, energy_it))
#             H_t = np.concatenate((H_t, [angles[it] if it % 2 == 0 else 0 for _ in t_it]))
#             B_t = np.concatenate((B_t, [angles[it] if it % 2 == 1 else 0 for _ in t_it]))
#             t = np.concatenate((t, it + t_it))
#     return (psi_t, energy_t, H_t, B_t, t) if store_states else energy

def QAOA(H, B, alpha, dims, title, DTWA=None):
    # initial state: all up in x basis (ground state of reference Hamiltonian)
    psi_0 = (1 / (2 ** (N / 2))) * np.ones(2 ** N,) # all up in x basis
    
    # optimal angles to reach ground state using VQE
    angle_bounds = [(0, np.inf) if i % 2 == 0 else (- np.inf, np.inf) for i in range(2 * alpha)]
    start = time.time()
    angles = VQE(psi_0, H, B, 2 * alpha, DTWA, param_bounds=angle_bounds, penalize=True) # series of reference 
    end = time.time()
    
    print('ANGLES (BETA - GAMMA - ...)')
    print(angles)

    # state vs. time and energy vs. time
    psi_t, energy_t, H_t, B_t, t = QAOA_evolution(psi_0, H, B, angles, np.linspace(.2, 1., 5), store_states=True)

    # state probabilities vs. time
    state_probs_t = np.array([state_probs(psi) for psi in psi_t]) 
    # util.plot_state_probs(state_probs_t, energy_t, dims, 'step_fn', 1, alpha, title)

    util.plot_energy_hamiltonians_vs_time(energy_t, H_t, B_t, t, dims, 'step_fn', 1, alpha, title)
    return t, angles, state_probs_t, end - start

if __name__ == '__main__':

    for L in range(2, 5):
        # setup system
        dims = (L, L)
        prob = maxcut.initialize_problem(experiment.step_fn, 1, *dims)
        N = prob.get_num_vertices()
        basis = spin_basis_general(N)

        title = 'qaoa_{}x{}'.format(L, L)
        util.make_dir(title)

        # Hamiltonian terms for Ising interactions and reference field
        J_zz = prob.get_edges()
        h_x = [[- 1, i] for i in range(N)]

        # Hamiltonian for Ising interactions
        static = [["zz", J_zz]]
        dynamic = []
        H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

        # reference Hamiltonian
        B = hamiltonian([["x", h_x]], [], dtype=np.float64, basis=basis, check_herm=False, check_symm=False)

        # exact ground states
        states = util.get_states_str(dims)
        ground_state_energy, num_ground_states, ground_states = classical_algorithms.BruteForce().solve(prob, allGroundStates=True)
        ground_states_id = []
        for ground_state in ground_states:
            ground_state_str = ''
            for v in range(N):
                ground_state_str += str(int((ground_state[v] + 1) / 2))
            ground_states_id.append(states.index(ground_state_str))

        # QAOA algorithm runs
        state_probs_t_alpha = {}
        angles_alpha = {}
        VQE_runtimes_alpha = {}
        alpha_range = range(1, 11) if L < 4 else range(1, 5)
        for alpha in alpha_range:
            t, angles, state_probs_t, VQE_runtime = QAOA(H, B, alpha, dims, title)
            state_probs_t_alpha[alpha] = (state_probs_t, t)
            angles_alpha[alpha] = angles
            VQE_runtimes_alpha[alpha] = VQE_runtime
        util.plot_ground_state_fidelities_vs_time(state_probs_t_alpha, ground_states_id, angles_alpha, dims, 'step_fn', 1, title)
        util.plot_final_ground_state_fidelities_vs_alpha(state_probs_t_alpha, ground_states_id, dims, 'step_fn', 1, title)
        util.plot_final_ground_state_fidelities_vs_beta_sum(state_probs_t_alpha, ground_states_id, angles_alpha, dims, 'step_fn', 1, title)
        util.plot_VQE_runtimes_beta_sums_vs_alpha(VQE_runtimes_alpha, angles_alpha, dims, 'step_fn', 1, title)