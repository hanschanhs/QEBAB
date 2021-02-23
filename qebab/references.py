from pytket.circuit import Circuit
from pytket.extensions.qiskit import tk_to_qiskit


def s0_circ(n_orbitals: int, n_electrons: int):
    """
    Args:
        n_orbitals (int) = number of qubits/spin orbitals from MolecularData 
        n_electrons (int) = number of electrons from MolecularData
    Returns:
        circ (pytket.circuit.Circuit object) = in Hartree-Fock ground state
    """   
    circ = Circuit(n_orbitals)
    
    for i in range(n_electrons):
        circ.X(i) #rotates from |0> to |1>
    
    return circ


def t1_circ(n_orbitals: int, n_electrons: int):
    """
    Args:
        n_orbitals (int) = number of qubits/spin orbitals from MolecularData 
        n_electrons (int) = number of electrons from MolecularData
    Returns:
        circ (pytket.circuit.Circuit object) = in T1 configuration from HOMO->LUMO
    """
    circ = Circuit(n_orbitals)
    
    for i in range(n_electrons-1): # for all but one electrons
        circ.X(i) # rotates from |0> to |1>
    circ.X(n_electrons) # rotate from |0> to |1> for the triplet spin orbital

    return circ


ref_circ_library = {
    's0': s0_circ,
    't1': t1_circ
}


if __name__ == "__main__":
    n_orbitals = 12
    n_electrons = 4
    ref = 's0'

    gen_ref_circ = ref_circ_library[ref]

    print(tk_to_qiskit(gen_ref_circ(n_orbitals, n_electrons)))
