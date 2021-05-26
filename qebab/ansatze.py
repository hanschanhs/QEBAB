from qebab.operators import *
from qebab.references import *
from qebab.expectation import gen_overlap

from math import sqrt

from pytket.circuit import Circuit, OpType, fresh_symbol
from pytket.utils import QubitPauliOperator, gen_term_sequence_circuit
from pytket.partition import PauliPartitionStrat, GraphColourMethod
from pytket.transform import Transform, PauliSynthStrat, CXConfigType

import numpy as np

class Ansatz:
    """
    Takes initialised pool of unitary operators and returns parameterised ansatz
    """
    def __init__(self, pool: OperatorPool):
        self.pool = pool
        self.smart_circ = None

        self.qubit_pauliop = {}
        self.symbols = {}
        self.f_op = []
        self.op_indices = []
        self.ref_circ = None
        self.grad_circ = None
        self.converged = False

    def generate_Circuit(self):
        print("VIRTUAL")
        exit()


    def generate_SparseMat(self):
        print("VIRTUAL")
        exit()



class k_UCC_Ansatz(Ansatz):
    def generate_Circuit(self, ref: str, k: int, backend):
        """
        """
        print("Form ansatz preparation circuit:")
        n_qubits = self.pool.n_spin_orb
        n_electrons = self.pool.n_electrons
        n_params = self.pool.n_ops
        print(" No. of qubits: ", n_qubits)

        # Reference circuits
        try:
            gen_ref_circ = ref_circ_library[ref]
        except:
            raise ValueError('{} not a valid reference state.'.format(ref))
        
        ref_circ = gen_ref_circ(self.pool.n_spin_orb, self.pool.n_electrons)

        # Ansatz - k-depth
        print(" k :", k)
        self.symbols = {}
        for rep in range(1, k+1):
            qubit_pauliop = {}
            # Iterate through operators in the pool
            for i in range(n_params):
                # Generate fresh symbol for a new operator
                theta = fresh_symbol('t{}'.format(i))
                self.symbols[theta] = None
                # Isolate the operator
                op = self.pool.qubit_paulistrs[i]
                for qpstr, coeff in op.items():
                    if coeff.imag > 0:
                        qubit_pauliop[qpstr] = theta
                    else:
                        qubit_pauliop[qpstr] = -1.0 * theta

            Pauli_U = QubitPauliOperator(qubit_pauliop)

            if rep==1:
                sym_circ = gen_term_sequence_circuit(Pauli_U,
                                                     ref_circ,
                                                     partition_strat=PauliPartitionStrat.CommutingSets,
                                                     colour_method=GraphColourMethod.Lazy)

            else:
                k_circ = Circuit(n_qubits)
                k_circ = gen_term_sequence_circuit(Pauli_U,
                                                   k_circ,
                                                   partition_strat=PauliPartitionStrat.CommutingSets,
                                                   colour_method=GraphColourMethod.Lazy)

                sym_circ.append(k_circ)

        assert(len(self.symbols) == n_params*k)
        assert(len(sym_circ.free_symbols()) == n_params*k)

        print(" Without circuit optimisation:")
        naive_circ = sym_circ.copy()
        Transform.DecomposeBoxes().apply(naive_circ)
        print("  Depth: {} gates".format(naive_circ.depth()))
        print("  CX Depth: {}".format(naive_circ.depth_by_type(OpType.CX)))
        print("  CX Count: {}".format(naive_circ.n_gates_of_type(OpType.CX)))

        print(" With circuit optimisation:")
        self.smart_circ = sym_circ.copy()
        Transform.UCCSynthesis(PauliSynthStrat.Sets, CXConfigType.Tree).apply(self.smart_circ)
        print("  Depth: {} gates".format(self.smart_circ.depth()))
        print("  CX Depth: {}".format(self.smart_circ.depth_by_type(OpType.CX)))
        print("  CX Count: {}".format(self.smart_circ.n_gates_of_type(OpType.CX)))
        
        # Compile to relevant backend and store
        print("Compiling ansatz for {}:".format(backend))
        backend.compile_circuit(self.smart_circ)
        print(" Depth: {} gates".format(self.smart_circ.depth()))
        print(" CX Depth: {}".format(self.smart_circ.depth_by_type(OpType.CX)))
        print(" CX Count: {}".format(self.smart_circ.n_gates_of_type(OpType.CX)))
        print("")
        
        return self.smart_circ, self.symbols



class ADAPT_VQD_Ansatz(Ansatz):
    def generate_Circuit(self,
                        ref: str,
                        params: list, # currently desired state
                        eigen_ansatze: list, # list of circuits
                        beta: float,
                        ham_sparse,
                        backend,
                        threshold):
                
        if len(params)==0: # no parameters in currently desired state -> new eigenstate!
            # reset for new eigenstate
            self.smart_circ = None 
            self.symbols = {}
            self.f_op = []
            # reset reference circuit
            try:
                gen_ref_circ = ref_circ_library[ref]
            except:
                raise ValueError('{} not a valid reference state.'.format(ref))
        
            self.ref_circ = gen_ref_circ(self.pool.n_spin_orb, self.pool.n_electrons)
            
            qubit_pauliop = {}
            Pauli_U = QubitPauliOperator(qubit_pauliop)
            self.grad_circ = gen_term_sequence_circuit(Pauli_U,
                                                       self.ref_circ,
                                                       partition_strat=PauliPartitionStrat.CommutingSets,
                                                       colour_method=GraphColourMethod.Lazy)
            self.smart_circ = self.grad_circ.copy()

        else: # repopulate eigenstate currently stored in constructor
            assert(len(self.symbols)==len(params))
            self.grad_circ = self.smart_circ.copy()
            self.symbols = dict(zip(self.symbols, params))
            self.grad_circ.symbol_substitution(self.symbols)


        # Calculating gradients for operators in pool
        grad_toggle = False

        curr_norm = 0
        next_deriv = 0
        print(" Gradients:")
        for op_index in range(self.pool.n_ops):
            opstring = ""
            for t in self.pool.fermi_ops[op_index].terms:
                opstring += str(t)
                break
            
            # Gradient
            gi = self.pool.compute_gradient_i(op_index,
                                         ham_sparse,
                                         self.grad_circ,
                                         backend)
            # Overlap
            overlap_list = []
            for eigen_circ in eigen_ansatze:
                # 2 Re beta * <ansatz|A(k)|eigen><eigen|ansatz>
                overlap = sqrt(gen_overlap(self.grad_circ, eigen_circ, backend))
                ov_g = abs(self.pool.compute_ov_grad_i(op_index, self.grad_circ, eigen_circ, backend))
                #print("  Overlap = %12.8f     Over. Grad = %12.8f" %(overlap, ov_g))
                #overlap_list.append(abs(ov_g))
                overlap_list.append(abs(np.real(2 * beta * ov_g * overlap)))
            assert(len(overlap_list)==len(eigen_ansatze))
            overlap_sum = sum(overlap_list)

            if grad_toggle==True:
                print(" %4i %40s %12.8f %12.8f" %(op_index, opstring, abs(gi), overlap_sum) )
            else:    
                if abs(gi) + overlap_sum > threshold:
                    print(" %4i %40s %12.8f %12.8f" %(op_index, opstring, abs(gi), overlap_sum) )

            # Add it up
            gi = abs(gi) + overlap_sum

            curr_norm += gi*gi
            if abs(gi) > next_deriv:
                next_deriv = abs(gi)
                next_index = op_index

        curr_norm = np.sqrt(curr_norm)
        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" %curr_norm)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)


        # Convergence or growth
        if curr_norm < threshold:
            self.converged = True

        else:
            print(" Operator selected = ", next_index)
            qubit_pauliop = {}
            op_circ = Circuit(self.pool.n_spin_orb)

            # Generate fresh symbol
            theta = fresh_symbol('t')
            self.symbols[theta] = None

            # Append fermion operator
            self.f_op.append(self.pool.fermi_ops[next_index])
            self.op_indices.append(next_index)

            # Isolate operator
            op = self.pool.qubit_paulistrs[next_index]
            for qpstr, coeff in op.items():
                if coeff.imag > 0:
                    qubit_pauliop[qpstr] = theta
                else:
                    qubit_pauliop[qpstr] = -1.0 * theta

            Pauli_U = QubitPauliOperator(qubit_pauliop)
            op_circ = gen_term_sequence_circuit(Pauli_U,
                                                op_circ,
                                                partition_strat=PauliPartitionStrat.CommutingSets,
                                                colour_method=GraphColourMethod.Lazy)
            self.smart_circ.append(op_circ)

            print(" Without circuit optimisation:")
            naive_circ = self.smart_circ.copy()
            Transform.DecomposeBoxes().apply(naive_circ)
            print("  Depth: {} gates".format(naive_circ.depth()))
            print("  CX Depth: {}".format(naive_circ.depth_by_type(OpType.CX)))
            print("  CX Count: {}".format(naive_circ.n_gates_of_type(OpType.CX)))

            print(" With circuit optimisation:")
            #self.smart_circ = self.grad_circ.copy()
            Transform.UCCSynthesis(PauliSynthStrat.Sets, CXConfigType.Tree).apply(self.smart_circ)
            print("  Depth: {} gates".format(self.smart_circ.depth()))
            print("  CX Depth: {}".format(self.smart_circ.depth_by_type(OpType.CX)))
            print("  CX Count: {}".format(self.smart_circ.n_gates_of_type(OpType.CX)))

            # Compile to relevant backend and store
            print("Compiling ansatz for {}:".format(backend))
            backend.compile_circuit(self.smart_circ)
            print(" Depth: {} gates".format(self.smart_circ.depth()))
            print(" CX Depth: {}".format(self.smart_circ.depth_by_type(OpType.CX)))
            print(" CX Count: {}".format(self.smart_circ.n_gates_of_type(OpType.CX)))
            print("")
            
            
        return self.converged, self.smart_circ, self.symbols



# if __name__ == "__main__":
#     pool = sUpCCGSD_Pool()
#     pool.init(n_orb=6,n_occ=2,n_vir=4)
    
#     ansatz = k_UCC_Ansatz(pool)
#     sym_circ, sym_dic = ansatz.generate_Circuit(ref='s0',k=2)

    
#     # print("QubitPauliOperator Sequence: ")
#     # for command in sym_circ:
#     #     if command.op.type == OpType.CircBox:
#     #         print(" New CircBox:")
#     #         for pauli_exp in command.op.get_circuit():
#     #             print("  {} {} {}".format(pauli_exp,
#     #                                     pauli_exp.op.get_paulis(),
#     #                                     pauli_exp.op.get_phase()))
#     #     else:
#     #         print(" Native gate: {}".format(command))