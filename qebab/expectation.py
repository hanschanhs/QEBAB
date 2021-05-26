import numpy as np

from pytket.backends import Backend
from pytket.pauli import Pauli, QubitPauliString
from pytket.predicates import CompilationUnit
from pytket.circuit import Circuit, Qubit, Bit
from pytket.passes import OptimisePhaseGadgets, RemoveRedundancies, SequencePass
from pytket.utils import expectation_from_counts, expectation_from_shots, append_pauli_measurement
from pytket.utils.operators import QubitPauliOperator

from openfermion import QubitOperator, FermionOperator
from scipy.optimize import minimize



##### OVERLAPS #####
def get_zero_state_probability(circ: Circuit, backend: Backend):
    if backend.supports_state:
        statevector = backend.get_state(circ)
    else:
        raise NotImplementedError

    return abs(statevector[0])**2


def gen_overlap(a_circ: Circuit, b_circ: Circuit, backend: Backend):
    """Overlap measurement
    Args:
        a_circ ([type]): [description]
        b_circ ([type]): [description]
    Returns:
        [type]: [description]
    """
    # Takes in circuit with angles still parametrized gates
    circ = a_circ.copy()
    b_circ = b_circ.dagger()
    circ.append(b_circ)
    
    backend.compile_circuit(circ)

    prob_X = get_zero_state_probability(circ=circ, backend=backend)
    
    return prob_X


##### ENERGY #####
def gen_vqd_objective(ansatz: Circuit,
                        symbols: dict,
                        hamiltonian: QubitOperator,
                        backend: Backend,
                        eigen_ansatze: list,
                        beta: float):
    """Set up objective function for the VQE optimisation
    """
    if backend.supports_expectation:
        print(" Running expectation...")
        def objective_function(params):
            # Do Hamiltonian expectation measurement
            ansatz_at_params = ansatz.copy()
            symbol_map = dict(zip(symbols, params))
            ansatz_at_params.symbol_substitution(symbol_map)
            energy = backend.get_operator_expectation_value(ansatz_at_params,
                                                            QubitPauliOperator.from_OpenFermion(hamiltonian))
            energy = energy.real
            
            # Do overlap expectation measurement
            if len(eigen_ansatze)!=0:
                overlap_list = []
                for eigen in eigen_ansatze:
                    overlap = gen_overlap(ansatz_at_params, eigen, backend)
                    overlap_list.append(beta * overlap)
                assert(len(overlap_list)==len(eigen_ansatze))
                overlap_sum = sum(overlap_list)
                energy = overlap_sum + energy

            return energy
    else:
        raise NotImplementedError


    return objective_function  


##### SPIN #####
def get_spin_expectation(ansatz: Circuit, symbols: dict, optimised_params, spin_op, backend: Backend):

    ansatz_at_params = ansatz.copy()
    symbol_map = dict(zip(symbols, optimised_params))
    ansatz_at_params.symbol_substitution(symbol_map)
    spin_sq = backend.get_operator_expectation_value(ansatz_at_params, spin_op)

    return spin_sq



def gen_spin_operator(n_tot):
    """n_tot: number of molecular orbitals
    """
    spin_op = FermionOperator()

    for p in range(n_tot):
        for q in range(n_tot):
            # Term 1: S+S-
            p_alpha = 2*p
            p_beta = 2*p+1
            q_alpha = 2*q
            q_beta = 2*q+1

            splus_sminus = (
            FermionOperator(((p_alpha, 1), (p_beta, 0), (q_beta, 1), (q_alpha, 0)))
            )
            spin_op += splus_sminus    

            
            # Term 2: Sz_p * Sz_q 
            sz_p_sz_q = 1/4 * (
            FermionOperator(((p_alpha, 1), (p_alpha, 0), (q_alpha, 1), (q_alpha, 0)))
            - FermionOperator(((p_alpha, 1), (p_alpha, 0), (q_beta, 1), (q_beta, 0)))
            - FermionOperator(((p_beta, 1), (p_beta, 0), (q_alpha, 1), (q_alpha, 0)))
            + FermionOperator(((p_beta, 1), (p_beta, 0), (q_beta, 1), (q_beta, 0)))
            )
            spin_op += sz_p_sz_q

        

        # Term 3: -Sz_p
        sz_p = - 1/2 * (FermionOperator(((p_alpha, 1), (p_alpha, 0))) - FermionOperator(((p_beta, 1), (p_beta, 0))))
        spin_op += sz_p

    return spin_op






############# OPTIMISER ################
def energy_optimise(objective_function, initial_params, opt_method: str, opt_iter: int):
    """This does the optimisation
    Args:
        an objective function
        the rotation values
        optimisation method
    """ 
    print(" Start variational minimisation")
    var_result = []
    initial_energy = objective_function(initial_params)
    var_result.append(initial_energy)

    def callback_function(xk):
        """global iterator function
        """
        energy = objective_function(xk)
        var_result.append(energy)
        print("  Var. E. = {}".format(energy))

    opt_result = minimize(objective_function, 
                          initial_params,
                          method=opt_method,
                          callback=callback_function,
                          options = {'maxiter': opt_iter,
                                     'disp':False}) #Limit the maximum run time
    
    opt_energy, opt_params = opt_result.fun, opt_result.x

    
    return opt_energy, opt_params, var_result






