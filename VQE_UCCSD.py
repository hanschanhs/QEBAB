import numpy as np
import timeit
import json

from qebab import *

from pytket.backends.ibm import AerBackend, AerStateBackend, IBMQBackend
from pytket.backends.projectq import ProjectQBackend

from openfermionpyscf import run_pyscf
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner

########### SETTINGS ##########
basis = 'sto-3g'    # Orbital basis set
multiplicity = 1    # Spin of reference determinant
n_points = 5   # Number of geometries 
bond_length_interval = 4.0 / n_points

run_scf = 1
run_fci = 1
delete_input = True
delete_output = True

pool = sUpCCGSD_Pool()    # Operator Pool
constructor = k_UpCCGSD_Ansatz     # Ansatz constructor
reference = "s0"    # Qubit reference state
backend = AerStateBackend() # Backend for simulation
n_shots = 8000
syndrome = False # syndrome qubit
opt_method = "L-BFGS-B" #"Nelder-Mead"
opt_maxiter = 50


########### MAIN QUANTUM ROUTINE ##########
print("\n")
print(" --------------------------------------------------------------------------")
print("                             UCCSD-VQE")
print(" --------------------------------------------------------------------------")

molecules = []
SymbolicAnsatz = None

for point in range(1, n_points + 1):
    bond_length = bond_length_interval * float(point) + 0.2
    geometry = [['H', (0, 0, 0)],
                ['H', (0, bond_length, 0)], 
                ['H', (bond_length, bond_length, 0)],
                ['H', (bond_length, 0, 0)]]
    
    MolecularMeta = {}
    molecule = MolecularData(geometry,
                            basis,
                            multiplicity,
                            description=str(round(bond_length, 2)))
    molecule = run_pyscf(molecule,
                         run_scf=run_scf,
                         run_fci=run_fci)

    # Generate state preparation circuit (reusable for any geometry!)
    if not isinstance(SymbolicAnsatz, Circuit):
        pool.init(n_orb=molecule.n_orbitals,
                  n_occ=molecule.get_n_alpha_electrons(),
                  n_vir=molecule.n_orbitals - molecule.get_n_alpha_electrons())
        
        ansatz = constructor(pool)

        if syndrome==True and backend.supports_shots==False:
            raise NotImplementedError("Syndrome qubit not supported by backend.")
        else:
            SymbolicAnsatz, symbols = ansatz.generate_Circuit(ref=reference, k=1)
                
        # Compile to relevant backend and store
        print("Compiling ansatz for {}:".format(backend))
        SymbolicAnsatz, final_map = compile_circ(SymbolicAnsatz, backend)
        print(" Depth: {} gates".format(SymbolicAnsatz.depth()))
        print(" CX Depth: {}".format(SymbolicAnsatz.depth_by_type(OpType.CX)))
        print(" CX Count: {}".format(SymbolicAnsatz.n_gates_of_type(OpType.CX)))
        print("")

    # Basic information
    jim = molecule.name
    print("===== Molecule {} =====".format(jim))
    hf = molecule.hf_energy
    fci = run_FCI(molecule, 4)
    print(" HF: {} /Eh".format(hf))
    print(" FCI: {} /Eh".format(fci))
    MolecularMeta['Name'] = jim
    MolecularMeta['Geo'] = bond_length
    MolecularMeta['HF energy'] = hf
    MolecularMeta['FCI eigenstates'] = fci

    # Hamiltonian
    ham_qubit = jordan_wigner(molecule.get_molecular_hamiltonian())
    ham_qubit.compress() # Now in QubitOperator form
    ham = QubitPauliOperator.from_OpenFermion(ham_qubit)
    #MolecularMeta['QubitHamiltonian'] = ham_qubit

    # Generate energy objective function
    objective = gen_energy_objective(ansatz=SymbolicAnsatz,
                                     final_map=final_map,
                                     symbols=symbols,
                                     hamiltonian=ham,
                                     backend=backend,
                                     n_shots=n_shots)

    # Generate initial guess params
    if len(molecules) == 0:
        initial_params = np.zeros(len(symbols)) # zeros - poor guess!
    else:
        last_Meta = molecules[-1]
        initial_params = np.array(last_Meta['Optimised parameters']) # reuse previous optimisation

    # Energy optimisation
    start_t = timeit.default_timer()
    opt_energy, opt_params, var_result = energy_optimise(objective, initial_params, opt_method, opt_maxiter)
    stop_t = timeit.default_timer()

    MolecularMeta['VQE energy'] = opt_energy
    MolecularMeta['Optimised parameters'] = opt_params.tolist()
    MolecularMeta['Variational optimisation'] = var_result
    print(' Variational minimisation steps: ',len(var_result))
    print(' *Finish: {} /Eh'.format(opt_energy))
    print(' Accuracy v. FCI: {} /Eh'.format(opt_energy - molecule.fci_energy))
    t = stop_t - start_t
    MolecularMeta['Time'] = t
    print(' Variational Time: {}/s'.format(t))
    print("")

    molecules.append(MolecularMeta)

    gc.collect()



with open(jim.replace('_'+str(bond_length),'') + '.json', 'w') as json_file:
    json.dump(molecules, json_file)