import numpy as np
import itertools

from openfermion import *
from openfermion import transforms
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit
from pytket import Circuit
from pytket.backends import Backend


class OperatorPool:
    """Abstract class which generates pool of operators from electron occupancy
    """
    def __init__(self):
        self.n_orb = 0
        self.n_occ = 0
        self.n_vir = 0
        
        self.n_spin_orb = 0
        self.n_electrons = 0


    def init(self,n_orb,n_occ=None,n_vir=None):
        self.n_orb = n_orb
        self.n_spin_orb = 2 * self.n_orb

        if n_occ!=None:
            self.n_occ = n_occ
            self.n_vir = n_vir
            assert(self.n_occ + self.n_vir == self.n_orb)
            self.n_electrons = 2 * self.n_occ
        
        self.n_ops = 0
        self.generate_FermionOperators()
        self.generate_QubitOperators()
        self.generate_QubitPauliString()


    def generate_FermionOperators(self):
        print("Virtual")
        exit()


    def generate_QubitOperators(self):
        """Applies Jordan-Wigner, 
        transform FermionOperators to QubitOperators
        """
        self.qubit_ops = []
        self.n_paulis = 0
        print("Form QubitOperators from operators in pool:")
        for op in self.fermi_ops:
            q_op = transforms.jordan_wigner(op)
            q_op.compress()
            self.n_paulis += len(q_op.terms)
            self.qubit_ops.append(q_op)
        assert(len(self.qubit_ops) == self.n_ops)
        print(" No. of Paulis: ", self.n_paulis)
        print("")


    def generate_QubitPauliString(self):
        """Maps QubitOperator(OpenFermion) to QubitPauliString(pytket)
        """
        self.qubit_paulistrs = []
        self.n_uni_paulis = 0
        q_reg = [Qubit(i) for i in range(self.n_spin_orb)]
        print("Form QubitPauliStrings from operators in pool:")
        for op in self.qubit_ops:
            qbit_pauliop = {}
            for pstring, coeff in op.terms.items():
                idx = []
                pgates = []
                for qbit_id, P in pstring:
                    idx.append(q_reg[qbit_id])
                    if P == 'X':
                        pgates.append(Pauli.X)
                    elif P == 'Y':
                        pgates.append(Pauli.Y)
                    elif P == 'Z':
                        pgates.append(Pauli.Z)
                
                qbit_pstring = QubitPauliString(idx, pgates)
                qbit_pauliop[qbit_pstring] = coeff
            assert(len(qbit_pauliop)==len(op.terms))
            self.qubit_paulistrs.append(qbit_pauliop)
        assert(len(self.qubit_paulistrs)==self.n_ops)
        
        # Check for unique Pauli Strings
        flag = False
        if flag == True:
            counter = 0
            cache = []
            for i,j in itertools.combinations(range(self.n_ops),2):
                op_dici = self.qubit_paulistrs[i]
                op_dicj = self.qubit_paulistrs[j]

                repeated_str = [op_dici[key] for key in op_dici.keys() & op_dicj.keys()]
                if len(repeated_str) != 0:
                    # if i and j in cache:
                    #     pass
                    # else:
                    #     cache.extend([i,j])
                    counter += len(repeated_str)
                    print("  Flag: Operators {} and {} share {} Paulis.".format(i,j,len(repeated_str)))
                

            self.n_uni_paulis = self.n_paulis - counter
            print(" No. of unique Paulis: ", self.n_uni_paulis)
            print("")


    def generate_SparseMatrix(self):
        self.spmat_ops = []
        print("Form Sparse Matrices for operators in pool ")
        for op in self.fermi_ops:
            self.spmat_ops.append(transforms.get_sparse_operator(op, n_qubits = self.n_spin_orb))
        assert(len(self.spmat_ops) == self.n_ops)


    def compute_gradient_i(self,
                           op_index: int,
                           ham_sparse,
                           circ: Circuit,
                           backend: Backend):

        """Equation
        dE/dk = 2 Re <ansatz|HA(k)|ansatz> 
        """
        if backend.supports_state:
            state = backend.get_state(circ)
            bra = ham_sparse.dot(state)
            op = self.qubit_ops[op_index]
            op = qubit_operator_sparse(op, n_qubits=self.n_spin_orb)
            ket = op.dot(state)

            gradient = 2 * np.vdot(bra,ket)
            gradient_scalar = np.real(gradient)
            
        else:
            NotImplementedError

        return gradient_scalar

    
    def compute_ov_grad_i(self,
                          op_index: int,
                          circ,
                          eigen_circ,
                          backend):
        """Equation
        dO/dk = A * Re <ansatz|A(k)|eigen> 
        """
        if backend.supports_state:
            bra = backend.get_state(circ)
            ket = backend.get_state(eigen_circ)
        
            op = self.qubit_ops[op_index]
            op = qubit_operator_sparse(op, n_qubits=self.n_spin_orb)
            ket = op.dot(ket)

            gradient = np.vdot(bra,ket)
            gradient_scalar = np.real(gradient)
            
        else:
            NotImplementedError

        return gradient_scalar




class sUpCCGSD_Pool(OperatorPool):
    def generate_FermionOperators(self):
        """
        """
        print("Form singlet UpCCGSD operators pool: ")
        self.fermi_ops = []

        # Construct general singles
        print(' Construct singles')
        for p in range(self.n_orb):
            for q in range(self.n_orb):
                pa = 2*p
                qa = 2*q

                pb = 2*p+1
                qb = 2*q+1

                if p > q:
                    #if True:
                    termA = FermionOperator(((qa, 1), (pa, 0)))
                    termA -= hermitian_conjugated(termA)
                    termB = FermionOperator(((qb, 1), (pb, 0)))
                    termB -= hermitian_conjugated(termB)
                    op = termA + termB
                    self.fermi_ops.append(op)

        #Construct general doubles
        print(' Construct doubles')
        spatial_orb = list(range(self.n_orb))
        n_double_amps = len(list(itertools.combinations(spatial_orb, 2)))
        t_qq_pp = [1] * n_double_amps

        double_excitations = []
        for i, (p, q) in enumerate(itertools.combinations(spatial_orb, 2)):
            pa = 2*p
            qa = 2*q

            pb = 2*p+1
            qb = 2*q+1

            double_excitations.append([[qa, pa, qb, pb], t_qq_pp[i]])

        empty_sing_amps = []
        for item in double_excitations:
            op = uccsd_generator(empty_sing_amps, [item])
            op = normal_ordered(op)
            self.fermi_ops.append(op)

        self.n_ops = len(self.fermi_ops)
        print(" Total No. of operators: ", self.n_ops)
        print("")
        
        return





class sUCCSD_Pool(OperatorPool):
    def generate_FermionOperators(self):
        """
        """
        print("Form singlet UCCSD operators pool: ")
        # PREDICT OPERATORS
        n_total_amplitudes = uccsd_singlet_paramsize(n_qubits=self.n_spin_orb,
                                                     n_electrons=self.n_electrons)
        n_single_amplitudes = self.n_occ * self.n_vir
        # Single amplitudes
        t1 = n_single_amplitudes
        # Double amplitudes associated with one spatial occupied-virtual pair
        t2_1 = n_single_amplitudes
        # Double amplitudes associated with two spatial occupied-virtual pairs
        t2_2 = n_total_amplitudes - 2*n_single_amplitudes
        print(" No. singles: ", t1)
        print(" No. doubles (1 MO pair): ", t2_1)
        print(" No. doubles (2 MO pairs): ", t2_2)

        # GENERATE OPERATORS
        self.fermi_ops = []
        spin_index_functions = [up_index, down_index]
        coeff = 1.0

        # Generate all spin-conserving single and double excitations derived from one spatial occupied-virtual pair
        for i, (p, q) in enumerate(itertools.product(range(self.n_vir), range(self.n_occ))):
            # list of operators sharing the same amplitude (indexed i)
            singles_i = FermionOperator()
            doubles1_i = FermionOperator()

            # Get indices of spatial orbitals
            virtual_spatial = self.n_occ + p
            occupied_spatial = q

            for spin in range(2):
                # Get the functions which map a spatial orbital index to a spin orbital index
                this_index = spin_index_functions[spin]
                other_index = spin_index_functions[1 - spin]

                # Get indices of spin orbitals
                virtual_this = this_index(virtual_spatial)
                virtual_other = other_index(virtual_spatial)
                occupied_this = this_index(occupied_spatial)
                occupied_other = other_index(occupied_spatial)

                # SINGLES
                singles_i += FermionOperator(((virtual_this, 1),(occupied_this, 0)),
                                             coeff)
                singles_i += FermionOperator(((occupied_this, 1),(virtual_this, 0)),
                                             -coeff)

                # DOUBLES
                doubles1_i += FermionOperator((
                                            (virtual_this, 1),
                                            (occupied_this, 0),
                                            (virtual_other, 1),
                                            (occupied_other, 0)),
                                            coeff)
                doubles1_i += FermionOperator((
                                            (occupied_other, 1),
                                            (virtual_other, 0),
                                            (occupied_this, 1),
                                            (virtual_this, 0)),
                                            -coeff)

            self.fermi_ops.append(singles_i)
            self.fermi_ops.append(doubles1_i)

        # Generate all spin-conserving double excitations derived from two spatial occupied-virtual pairs
        for i, ((p, q), (r, s)) in enumerate(itertools.combinations(
                                             itertools.product(range(self.n_vir), range(self.n_occ)),2)):
            # list of operators sharing the same amplitude (indexed i)
            doubles2_i = FermionOperator()

            # Get indices of spatial orbitals
            virtual_spatial_1 = self.n_occ + p
            occupied_spatial_1 = q
            virtual_spatial_2 = self.n_occ + r
            occupied_spatial_2 = s

            # Generate double excitations
            for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
                # Get the functions which map a spatial orbital index to a spin orbital index
                index_a = spin_index_functions[spin_a]
                index_b = spin_index_functions[spin_b]

                # Get indices of spin orbitals
                virtual_1_a = index_a(virtual_spatial_1)
                occupied_1_a = index_a(occupied_spatial_1)
                virtual_2_b = index_b(virtual_spatial_2)
                occupied_2_b = index_b(occupied_spatial_2)
                
                doubles2_i += FermionOperator((
                    (virtual_1_a, 1),
                    (occupied_1_a, 0),
                    (virtual_2_b, 1),
                    (occupied_2_b, 0)),
                    coeff)
                doubles2_i += FermionOperator((
                    (occupied_2_b, 1),
                    (virtual_2_b, 0),
                    (occupied_1_a, 1),
                    (virtual_1_a, 0)),
                    -coeff)
            
            self.fermi_ops.append(doubles2_i)

        # CROSSCHECK OPERATORS
        self.n_ops = len(self.fermi_ops)
        assert(n_total_amplitudes == self.n_ops)
        print(" Total No. of operators: ", self.n_ops)
        print("")

        return

        


class sUCCGSD_Pool(OperatorPool):
    def generate_FermionOperators(self):
        """
        WARNING: THIS IS INTENDED TO ONLY WORK FOR H2
        These are the types of general singlet operators:
        - For each unique pair of spatial MOs:
            - single electron hops from one MO to the other
            - double electron hops from one MO to the other
            - cross electron hops e.g. t_{03}^{12}
        - For each 2 pairs of spatial orbitals:
            - 
        """
        print("WARNING: Form H2 singlet UCCGSD operators pool: ")

        # GENERATE OPERATORS
        self.fermi_ops = []
        spin_index_functions = [up_index, down_index]
        coeff = 1.0

        # Generate all spin-conserving single and double excitations derived from one spatial occupied-virtual pair
        for i, (p, q) in enumerate(itertools.product(range(self.n_vir), range(self.n_occ))):
            # list of operators sharing the same amplitude (indexed i)
            singles_i = FermionOperator()
            singles_t_i = FermionOperator()
            doubles1_i = FermionOperator()
            
            # Get indices of spatial orbitals
            virtual_spatial = self.n_occ + p
            occupied_spatial = q

            for spin in range(2):
                # Get the functions which map a spatial orbital index to a spin orbital index
                this_index = spin_index_functions[spin]
                other_index = spin_index_functions[1 - spin]

                # Get indices of spin orbitals
                virtual_this = this_index(virtual_spatial)
                virtual_other = other_index(virtual_spatial)
                occupied_this = this_index(occupied_spatial)
                occupied_other = other_index(occupied_spatial)

                # SINGLES (sing)
                singles_i += FermionOperator(((virtual_this, 1),(occupied_this, 0)),
                                             coeff)
                singles_i += FermionOperator(((occupied_this, 1),(virtual_this, 0)),
                                             -coeff)

                # SINGLES (trip)
                singles_t_i += FermionOperator(((virtual_this, 1),(occupied_other, 0)),
                                                coeff)
                singles_t_i -= hermitian_conjugated(singles_t_i)

                # DOUBLES
                doubles1_i += FermionOperator((
                                            (virtual_this, 1),
                                            (occupied_this, 0),
                                            (virtual_other, 1),
                                            (occupied_other, 0)),
                                            coeff)
                doubles1_i += FermionOperator((
                                            (occupied_other, 1),
                                            (virtual_other, 0),
                                            (occupied_this, 1),
                                            (virtual_this, 0)),
                                            -coeff)

                # DOUBLECROSS
                # doublecross_i += FermionOperator((
                #                                 (occupied_other, 1),
                #                                 (occupied_this, 0),
                #                                 (virtual_this, 1),
                #                                 (virtual_other, 0)),
                #                                 coeff)
                # doublecross_i += FermionOperator((
                #                                 (virtual_other, 1),
                #                                 (virtual_this, 0),
                #                                 (occupied_this, 1),
                #                                 (occupied_other, 0)),
                #                                 -coeff)

            doublecross_i = FermionOperator((
                                            (1, 1),
                                            (0, 0),
                                            (2, 1),
                                            (3, 0)),
                                            coeff)
            doublecross_i -= hermitian_conjugated(doublecross_i)
                
            self.fermi_ops.append(singles_i)
            self.fermi_ops.append(singles_t_i)
            # print(singles_i)
            # print("")
            self.fermi_ops.append(doubles1_i)
            # print(doubles1_i)
            # print("")
            # self.fermi_ops.append(doublecross_i)
            #print(doublecross_i)

        self.n_ops = len(self.fermi_ops)
        print(" Total No. of operators: ", self.n_ops)
        print("")
        
        return


        





if __name__ == "__main__":
    pool = sUpCCGSD_Pool()
    pool.init(n_orb=6,n_occ=2,n_vir=4)


    for i in pool.fermi_ops:
        print(i)
        print("")


    print(" QubitPauliString: ")
    for i in pool.qubit_paulistrs:
        print(len(i))
