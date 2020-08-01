import maxcut
import experiment
import util
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
from quspin.tools.measurements import obs_vs_time
import numpy as np
import scipy
from scipy.optimize import minimize

ALPHA = 2 # number of time slices
DT = 0.1

# returns expected value of operator
def expectation(psi, op):
    return (np.dot(np.conjugate(psi), op @ psi) / np.dot(np.conjugate(psi), psi)).real

# returns Unitary operators
def unitary(op, angle):
    return scipy.linalg.expm(-1.j * np.array(op) * angle)

# returns probability distribution over states
def state_probs(psi):
    # normalize
    psi /= np.linalg.norm(psi)
    return (np.conjugate(psi) * psi).real

# time evolution under the QAOA procedure
def QAOA_evolution(psi_0, H, B, angles, store_states=False):
    psi = psi_0
    # t = 0
    if store_states: 
        psi_t = [psi]
        H_t = [0.]
        B_t = [0.]
        # t_vals = [t]
    op_t = [None]
    for it in range(len(angles)):
        op = H if it % 2 == 0 else B
        psi = unitary(op, angles[it]) @ psi
        # normalize
        psi /= np.linalg.norm(psi)
        # t += angles[it]
        op_t.append(op)
        if store_states: 
            psi_t.append(psi)
            if it % 2 == 0:
                H_t.append(angles[it])
                B_t.append(0.)
            else:
                H_t.append(0.)
                B_t.append(angles[it])
            # t_vals.append(t)
    return (psi_t, H_t, B_t, op_t, range(len(angles) + 1)) if store_states else psi

# time evolution under adiabatic process
def adiabatic_evolution(psi_0, H, B, t_max, store_states=False):
    t_vals = np.arange(0, t_max + DT, DT)
    psi = psi_0
    if store_states: 
        psi_t = [psi]
        H_t = [0.]
        B_t = [1.]
    op_t = [B]
    for t in t_vals[1:]:
        op = (1 - t / t_max) * B + (t / t_max) * H
        op_t.append(op)
        psi = unitary(op, DT) @ psi
        # normalize
        psi /= np.linalg.norm(psi)
        if store_states:
            psi_t.append(psi)
            H_t.append(t / t_max)
            B_t.append(1 - t / t_max)
    return (psi_t, H_t, B_t, op_t, t_vals) if store_states else op_t

def VQE_optimization_fn(evolution_fn, psi_0, H, B):
    def fn(param):
        psi_f = evolution_fn(psi_0, H, B, param)
        return expectation(psi_f, H)
    return fn

# def VQE_adiabatic_optimization_fn(evolution_fn, psi_0, H, B):
#     # maximize minimum gap between ground state and 1st excited state eigenvalues
#     def fn(param):
#         op_t = evolution_fn(psi_0, H, B, param)
#         eig_t = [np.linalg.eigvals(op) for op in op_t]
#         eig_0_t = [(np.unique(np.round(eig, 8))[0]).real for eig in eig_t]
#         eig_1_t = [(np.unique(np.round(eig, 8))[1].real) for eig in eig_t]
#         eig_gaps_t = np.array(eig_1_t) - np.array(eig_0_t)
#         return - min(eig_gaps_t).real
#     return fn

# def VQE(evolution_fn, psi_0, H, B, n_params, param_bounds=None):
def VQE(psi_0, J_zz, h_x, n_params, param_bounds=None):
    # optimization_fn = VQE_optimization_fn(evolution_fn, psi_0, H, B)
    optimization_fn = VQE_optimization_fn_quspin(psi_0, J_zz, h_x, H)
    if n_params == 1:
        init_params = 1.
    else:
        init_params = np.ones(n_params)
    classical_optimization = minimize(optimization_fn, init_params, constraints=param_bounds)
    if classical_optimization.get('success'):
        return classical_optimization.get('x')
    raise RuntimeError('Optimization failed')

    # init_angles = np.ones(2 * ALPHA) # series of beta, gamma for ALPHA iterations
    # optimization_fn = lambda angles: expectation(QAOA_evolution(psi_0, H, B, angles), H)
    # classical_optimization = minimize(optimization_fn, init_angles, method='Nelder-Mead')
    # if classical_optimization.get('success'):
    #     return classical_optimization.get('x')
    # raise RuntimeError('Optimization failed')

# def QAOA(H, B, dims):
def QAOA(J_zz, h_x, dims):
    # initial state: all up in x basis (ground state of reference Hamiltonian)
    psi_0 = (1 / (2 ** (N / 2))) * np.ones(2 ** N,) # all up in x basis
    
    # optimal angles to reach ground state using VQE
    # optimization_fn = VQE_optimization_fn(QAOA_evolution, psi_0, H, B)
    # angles = VQE(QAOA_evolution, psi_0, H, B, 2 * ALPHA) # series of reference angle (beta), maxcut angle (gamma) for ALPHA iterations
    angles = VQE(psi_0, J_zz, h_x, 2 * ALPHA) # series of reference 
    
    print('ANGLES (BETA - GAMMA - ...)')
    print(angles)

    # time evolution under the QAOA procedure
    # psi_t, H_t, B_t, op_t, t = QAOA_evolution(psi_0, H, B, angles, store_states=True)
    op, H_t, B_t = QAOA_Hamiltonian_quspin(J_zz, h_x, angles, mag=True)
    tvals = np.linspace(0, 2 * ALPHA, 101)
    psi_t = op.evolve(psi_0, 0, tvals)
    obs_t = obs_vs_time(psi_t, tvals, dict(energy=H))


    # final state
    # print('FINAL STATE')
    # psi_f = psi_t[-1]
    # print(psi_f)

    # energy vs. time
    # energy_t = np.array([expectation(psi, H) for psi in psi_t])
    energy_t = energy_t = obs_t['energy']
    # psi_t = np.transpose(obs_t['psi_t'])
    state_probs_t = np.array([state_probs(psi) for psi in np.transpose(psi_t)])
    # print('ENERGY VS. TIME')
    # print(energies)
    # state_probs_t = np.array([state_probs(psi) for psi in psi_t])
    # util.plot_state_probs(state_probs_t, energy_t, dims, 'step_fn', 1, 'quantum_qaoa')
    # util.plot_exp_hamiltonian_vs_time(energy_t, tvals, dims, 'step_fn', 1, 'quantum_qaoa')
    util.plot_energy_hamiltonians_vs_time(energy_t, H_t, B_t, np.linspace(0, 2 * ALPHA, 101), dims, 'step_fn', 1, 'quantum_qaoa')

    
def adiabatic(H, B, dims):
    # initial state: all up in x basis (ground state of reference Hamiltonian)
    psi_0 = (1 / (2 ** (N / 2))) * np.ones(2 ** N,) # all up in x basis
    
    # optimal params to reach ground state using VQE
    # optimization_fn = VQE_adiabatic_optimization_fn(adiabatic_evolution, psi_0, H, B)
    # params = VQE(optimization_fn, psi_0, H, B, 1, param_bounds=[dict(type='ineq', fun=lambda x: x)]) # parameterization of evolution speed
    
    # time evolution under the QAOA procedure
    t_max = 20
    psi_t, H_t, B_t, op_t, t = adiabatic_evolution(psi_0, H, B, t_max, store_states=True)

    # energy vs. time
    energy_t = np.array([expectation(psi, H) for psi in psi_t])
    
    state_probs_t = np.array([state_probs(psi) for psi in psi_t])
    util.plot_state_probs(state_probs_t, energy_t, dims, 'step_fn', 1, 'quantum_adiabatic')
    # util.plot_exp_hamiltonian_vs_time(energy_t, t_vals, dims, 'step_fn', 1, 'quantum_adiabatic')
    util.plot_energy_hamiltonians_vs_time(energy_t, H_t, B_t, t, dims, 'step_fn', 1, 'quantum_adiabatic')

    eig_t = [np.linalg.eigvals(op) for op in op_t]
    # eig_t_uniq = [np.unique(np.round(eig, 8)) for eig in eig_t]
    # print('FINAL STATE EIGENVALUES AND EIGENVECTORS')
    # print(np.linalg.eig(op_t[-1]))
    # eig_0_t = [eig[0] for eig in eig_t_uniq]
    # eig_1_t = [eig[1] for eig in eig_t_uniq]

    util.plot_eigval_crossing(eig_t, t, dims, 'step_fn', 1, 'quantum_adiabatic')

def QAOA_Hamiltonian_quspin(J_zz, h_x, angles, mag=False):
    def H_strength(t, *angles):
        angles = [angle for angle in angles]
        if int(t) % 2 == 0 and int(t) < len(angles):
            returnval = angles[int(t)]
            return returnval
        return 0
    def B_strength(t, *angles):
        angles = [angle for angle in angles]
        if int(t) % 2 == 1 and int(t) < len(angles):
            return angles[int(t)]
        return 0
    static = []
    dynamic = [["zz", J_zz, H_strength, list(angles)], ["x", h_x, B_strength, list(angles)]]
    op = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    # print(op.toarray())
    H_t = [H_strength(t, *angles) for t in np.linspace(0, 2 * ALPHA, 101)]
    B_t = [B_strength(t, *angles) for t in np.linspace(0, 2 * ALPHA, 101)]
    return op if not mag else (op, H_t, B_t)

def VQE_optimization_fn_quspin(psi_0, J_zz, h_x, H):
    def fn(angles):
        op = QAOA_Hamiltonian_quspin(J_zz, h_x, angles)
        tvals = np.linspace(0, 2 * ALPHA, 101)
        psi_t = op.evolve(psi_0, 0, tvals)
        obs_t = obs_vs_time(psi_t, tvals, dict(energy=H))
        return obs_t['energy'][-1]
    return fn

if __name__ == '__main__':
    # setup system
    dims = (3, 3)
    prob = maxcut.initialize_problem(experiment.step_fn, 1, *dims)
    N = prob.get_num_vertices()
    basis = spin_basis_general(N)

    # Hamiltonian for Ising interactions
    J_zz = prob.get_edges()
    h_x = [[- 1, i] for i in range(N)]

    static = [["zz", J_zz]]
    dynamic = []
    H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    # print('ISING HAMILTONIAN MATRIX')
    # print(H.toarray())

    eigvals, eigvecs = np.linalg.eig(H.toarray())
    # print('EIGENVALUES & EIGENVECTORS OF ISING HAMILTONIAN MATRIX')
    # for i in range(len(eigvals)):
    #     print(eigvals[i], eigvecs[:,i])

    # reference Hamiltonian
    B = hamiltonian([["x", h_x]], [], dtype=np.float64, basis=basis, check_herm=False, check_symm=False)
    print('=== QAOA ===')
    # QAOA(H.toarray(), B.toarray(), dims)
    QAOA(J_zz, h_x, dims)
    # print('=== Adiabatic ===')
    # adiabatic(H.toarray(), B.toarray(), dims)