import experiment
import maxcut
import maxcut_free
import util
from DTWA.TamLib import Heisenberg_A2A
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d, spin_basis_general
from quspin.tools.measurements import obs_vs_time
from quspin.tools.evolution import evolve
import numpy as np
import scipy
from itertools import combinations
import time
import matplotlib.pyplot as plt

LATTICE_SPACING = 1

# returns expected value of operator
def expectation(psi, op):
    return (np.dot(np.conjugate(psi), op @ psi) / np.dot(np.conjugate(psi), psi)).real

# returns probability distribution over states
def state_probs(psi):
    # normalize
    psi /= np.linalg.norm(psi)
    return (np.conjugate(psi) * psi).real
    
def quspin(prob, dims, plot=True):
    # possible 2d symmetries
    # s = np.arange(N_2d) # sites [ 0,1,2,....]
    # x = s%Lx # x positions for sites
    # y = s//Lx # y positions for sites
    # T_x = (x+1)%Lx + Lx*y # translation along x-direction
    # T_y = x +Lx*((y+1)%Ly) # translation along y-direction
    # P_x = x + Lx*(Ly-y-1) # reflection about x-axis
    # P_y = (Lx-x-1) + Lx*y # reflection about y-axis
    # Z = -(s+1) # spin inversion
    # basis_2d = spin_basis_general(N_2d)

    # setup system
    N = prob.get_num_vertices()
    basis = spin_basis_general(N)

    # Hamiltonian for Ising interactions
    Jzz = prob.get_edges()
    static = [["zz", Jzz]]
    dynamic = []
    Ising_Hamiltonian = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    if dims == (3,3):
        print('HAMILTONIAN MATRIX')
        print(Ising_Hamiltonian.toarray())

    # initial state: all up in x basis
    psi_0 = (1 / (2 ** (N / 2))) * np.ones(2 ** N,) # all up in x basis
    # initial state: random eigenvector
    # Why doesn't imaginary time evolution converge to ground state when I start at an eigenstate? Does it have zero overlap with ground state wavefunction?
    # E, V = IsingHamiltonian.eigh()
    # random_col = np.random.choice(np.size(V, axis=1))
    # psi_0 = V[:, random_col]
    
    # imaginary time evolution
    tvals = np.linspace(0, 5, 50)
    # psi_t = Ising_Hamiltonian.evolve(psi_0, 0, tvals, imag_time=True, iterate=True)
    start = time.time()
    psi_t = Ising_Hamiltonian.evolve(psi_0, 0, tvals, imag_time=True)
    end = time.time()
    evolve_time = end - start
    # define z operator on each spin
    # How to measure z spin of all atoms simultaneously? Can I have a non-aggregate solution that gives me information about all spins?
    # spin_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    # identity = np.array([[1.0, 0.0], [0.0, 1.0]])
    # spin_z_multi = spin_z
    # spin_z_0 = spin_z
    # for _ in range(N - 1):
    #     spin_z_multi = np.kron(spin_z_multi, spin_z)
    #     spin_z_0 = np.kron(spin_z_0, identity)
    Sz = hamiltonian([[N * "z", [[1] + list(range(N)) for _ in range(1)]]], [], dtype=np.float64, basis=basis, check_herm=False, check_symm=False)
    Sz_sum = hamiltonian([["z",[[1,i] for i in range(N)]]],[],dtype=np.float64,basis = basis,check_herm = False, check_symm = False)
    # Sz_0 = hamiltonian([["z", [[1, 0] for _ in range(1)]]], [], dtype=np.float64, basis=basis, check_herm=False, check_symm=False)
    # measure observables
    # obs_t = obs_vs_time(psi_t, tvals, dict(energy=Ising_Hamiltonian, spin_z=Sz, spin_z_sum=Sz_sum), return_state=True)
    start = time.time()
    obs_t = obs_vs_time(psi_t, tvals, dict(energy=Ising_Hamiltonian))
    end = time.time()
    observe_time = end - start
    # obs_t = obs_vs_time(psi_t, tvals, dict(energy=Ising_Hamiltonian, spin_z=Sz, spin_z_0=Sz_0), return_state=True)
    energy_t = obs_t['energy']
    # psi_t = np.transpose(obs_t['psi_t'])
    state_probs_t = np.array([state_probs(psi) for psi in np.transpose(psi_t)])
    
    # final state
    # print('FINAL STATE')
    # psi_f = obs_t['psi_t'][:, -1]
    # print(psi_f)

    # expected value of Hamiltonian (energy) in time
    # print('ENERGY VS. TIME')
    # print(obs_t['energy'])

    if plot:
        util.plot_state_probs(state_probs_t, energy_t, dims, 'step_fn', 1, 'quantum_quspin')
        util.plot_energy_vs_time(energy_t, tvals, dims, 'step_fn', 1, 'quantum_quspin')

    # spin-z vs. time
    # print('SPIN-Z OBSERVABLE IN TIME')
    # print(obs_t['spin_z'])
    # print('SUM(SPIN-Z) OBSERVABLE IN TIME')
    # print(obs_t['spin_z_sum'])
    # print(np.dot(obs_t['psi_t'][:, -1], np.dot(spin_z_multi, obs_t['psi_t'][:, -1])))
    # print(np.dot(obs_t['psi_t'][:, -1], np.dot(Sz.toarray(),obs_t['psi_t'][:, -1])))
    # print('SPIN-Z OBSERVABLE OF ATOM 0 IN TIME')
    # print(obs_t['spin_z_0'])
    # print(np.dot(obs_t['psi_t'][:, -1], np.dot(spin_z_0, obs_t['psi_t'][:, -1])))
    return evolve_time, observe_time

def tamlib(prob, dims, plot=True):
    # setup system
    N = prob.get_num_vertices()
    lattice_coords = util.get_lattice_coords(2, dims, 1)

    # Hamiltonian for Ising interactions
    Jfunc = [experiment.none(), experiment.none(), experiment.step_fn(1)]
    Ising_Hamiltonian, opList, S_tot_ops = Heisenberg_A2A(dims, 0., -1., 0, 0, alpha=6, Jfunc=Jfunc, coord=lattice_coords, PauliBool=True)
    if dims == (3,3):
        print('HAMILTONIAN MATRIX')
        print(Ising_Hamiltonian)

    # initial state: all up in x basis
    psi_0 = (1 / (2 ** (N / 2))) * np.ones(2 ** N,) # all up in x basis
    
    # imaginary time evolution
    tvec = np.linspace(0, 5, 50)
    start = time.time()
    psi_t = np.array([scipy.linalg.expm(-1.j * np.array(Ising_Hamiltonian) * t) @ psi_0 for t in tvec * -1.j])
    end = time.time()
    # normalize at each time step
    psi_t = np.array([psi / np.linalg.norm(psi) for psi in psi_t])
    evolve_time = end - start

    # final state
    # print('FINAL STATE')
    # psi_f = psi_t[-1]
    # print(psi_f)

    # energy vs. time
    start = time.time()
    energy_t = np.array([expectation(psi, Ising_Hamiltonian) for psi in psi_t])
    end = time.time()
    observe_time = end - start
    # print('ENERGY VS. TIME')
    # print(energies)
    
    state_probs_t = np.array([state_probs(psi) for psi in psi_t])
    
    if plot:
        util.plot_state_probs(state_probs_t, energy_t, dims, 'step_fn', 1, 'quantum_tamlib')
        util.plot_energy_vs_time(energy_t, tvec, dims, 'step_fn', 1, 'quantum_tamlib')

    # spin-z vs. time
    # sum_S_z = np.array([np.round(np.dot(np.conjugate(psi), S_tot_ops[2] @ psi), 15) for psi in psi_t])
    # print('SUM(SPIN-Z) OBSERVABLE IN TIME')
    # print(sum_S_z)
    return evolve_time, observe_time

if __name__ == '__main__':
    quspin_evolve_time = []
    quspin_observe_time = []
    tamlib_evolve_time = []
    tamlib_observe_time = []
    for d in range(2, 4):
        dims = (d, d)
        prob = maxcut.initialize_problem(experiment.step_fn, 1, *dims)
        quspin_evolve_time_d = 0.
        quspin_observe_time_d = 0.
        tamlib_evolve_time_d = 0.
        tamlib_observe_time_d = 0.
        for i in range(10):
            plot = True if i == 0 and d == 3 else False
            # print('=== Exact (QuSpin) ===')
            quspin_evolve_time_i, quspin_observe_time_i = quspin(prob, dims, plot)
            # print('evolve time: {}'.format(quspin_evolve_time))
            # print('observe time: {}'.format(quspin_observe_time))
            # print('=== Exact (TamLib) ===')
            quspin_evolve_time_d += quspin_evolve_time_i
            quspin_observe_time_d += quspin_observe_time_i
            tamlib_evolve_time_i, tamlib_observe_time_i = tamlib(prob, dims, plot)
            # print('evolve time: {}'.format(tamlib_evolve_time))
            # print('observe time: {}'.format(tamlib_observe_time))
            tamlib_evolve_time_d += tamlib_evolve_time_i
            tamlib_observe_time_d += tamlib_observe_time_i
        quspin_evolve_time.append(quspin_evolve_time_d)
        quspin_observe_time.append(quspin_observe_time_d)
        tamlib_evolve_time.append(tamlib_evolve_time_d)
        tamlib_observe_time.append(tamlib_observe_time_d)
    fig = plt.figure()
    plt.title('QuSpin vs. TamLib, Imaginary Time Evolution, L x L square lattice, step_fn, radius=1, 10 runs')
    plt.xlabel('System Size (L)')
    plt.ylabel('Runtime (s)')
    plt.plot(range(2, 4), quspin_evolve_time, label='quspin, evolve')
    print('quspin_evolve_time')
    print(quspin_evolve_time)
    plt.plot(range(2, 4), quspin_observe_time, label='quspin, expect')
    print('quspin_observe_time')
    print(quspin_observe_time)
    plt.plot(range(2, 4), tamlib_evolve_time, label='tamlib, evolve')
    print('tamlib_evolve_time')
    print(tamlib_evolve_time)
    plt.plot(range(2, 4), tamlib_observe_time, label='tamlib, expect')
    plt.legend()
    print('tamlib_observe_time')
    print(tamlib_observe_time)
    plt.savefig('{}/runtimes.png'.format('quantum_tamlib'), bbox_inches='tight')
    plt.close()
    