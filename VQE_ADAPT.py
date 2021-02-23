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
n_points = 1   # Number of geometries 
bond_length_interval = 4.0 / n_points

run_scf = 1
run_fci = 1
delete_input = True
delete_output = True

pool = sUpCCGSD_Pool()   # Operator Pool
constructor = ADAPT_Ansatz
reference = "s0"    # Qubit reference state
backend = AerStateBackend() # Backend for simulation
adapt_maxiter = 25
threshold = 0.01

n_shots = 8000
syndrome = False # syndrome qubit
opt_method = "L-BFGS-B" #"Nelder-Mead"
opt_maxiter = 80


########### MAIN QUANTUM ROUTINE ##########
print("\n")
print(" --------------------------------------------------------------------------")
print("                             ADAPT-VQE")
print(" --------------------------------------------------------------------------")

molecules = []

for point in [1.5]:#range(1, n_points + 1):
    bond_length = point#bond_length_interval * float(point) + 0.2
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
                         run_ccsd=1,
                         run_fci=run_fci)

    # Generate state preparation circuit (reusable for any geometry!)
    if pool.n_electrons == 0: # pool not initialised globally
        pool.init(n_orb=molecule.n_orbitals,
                  n_occ=molecule.get_n_alpha_electrons(),
                  n_vir=molecule.n_orbitals - molecule.get_n_alpha_electrons())
    Ansatz = constructor(pool)

    # Basic information
    jim = molecule.name
    print("===== Molecule {} =====".format(jim))
    hf = molecule.hf_energy
    fci = molecule.fci_energy
    print(" HF: {} /Eh".format(hf))
    print(" FCI: {} /Eh".format(fci))
    MolecularMeta['Name'] = jim
    MolecularMeta['Geo'] = bond_length
    MolecularMeta['HF energy'] = hf
    MolecularMeta['FCI eigenstates'] = run_FCI(molecule,4)
    run_CCSD(molecule,2,reference)
    

    # Hamiltonian
    ham_qubit = jordan_wigner(molecule.get_molecular_hamiltonian())
    ham_qubit.compress() # Now in QubitOperator form
    ham_QPO = QubitPauliOperator.from_OpenFermion(ham_qubit)
    ham_sparse = qubit_operator_sparse(ham_qubit, n_qubits=molecule.n_qubits)
    print("")
    
    # Build ansatz for this geometry
    print('Start to grow ADAPT ansatz')
    MolecularMeta['ADAPT energy'] = []
    opt_params = []
    for n_iter in range(0, adapt_maxiter):
        print('Iteration: ', n_iter)
        converged, SymbolicAnsatz, symbols, final_map = Ansatz.generate_Circuit(ref=reference,
                                                                     params=opt_params,
                                                                     ham_sparse=ham_sparse,
                                                                     backend=backend,
                                                                     threshold=threshold,
                                                                     n_unpaired_electrons=n_unpaired_electrons,
                                                                     n_beta_electrons=n_beta_electrons
                                                                     )
        
        if converged:
            print("")
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(Ansatz.f_op))
            print(" *Finished: %20.12f" % opt_energy)
            print(" -----------Final ansatz----------- ")
            coeffs = list(Ansatz.symbols.values())
            print(" %4s %40s %12s" %("#","Term","Coeff"))
            for si in range(len(Ansatz.symbols)):
                s = Ansatz.f_op[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4i %40s %12.8f" %(si, opstring, coeffs[si]) )
            print("")
            print("")
            break

        # Generate energy objective function
        objective = gen_energy_objective(ansatz=SymbolicAnsatz,
                                        final_map=final_map,
                                        symbols=symbols,
                                        hamiltonian=ham_QPO,
                                        backend=backend,
                                        n_shots=n_shots)

    # # Generate initial guess params
    # if len(molecules) == 0:
        initial_params = [np.random.uniform(0, 0.001) for i in range(len(symbols))]
    # else:
    #     last_Meta = molecules[-1]
    #     initial_params = np.array(last_Meta['Optimised parameters']) # reuse previous optimisation

        # Energy optimisation
        start_t = timeit.default_timer()
        opt_energy, opt_params, var_result = energy_optimise(objective, initial_params, opt_method, opt_maxiter)
        stop_t = timeit.default_timer()

        MolecularMeta['ADAPT energy'].append(opt_energy)
        print(' Accuracy v. FCI: {} /Eh'.format(opt_energy - molecule.fci_energy))
        t = stop_t - start_t
        print(' Variational Steps: ', len(var_result))
        print(' Variational Time: {}/s'.format(t))

        print("")


    MolecularMeta['Optimised parameters'] = opt_params.tolist()
    MolecularMeta['Gate depth, CX depth'] = (SymbolicAnsatz.depth(), SymbolicAnsatz.depth_by_type(OpType.CX))

    molecules.append(MolecularMeta)



with open(jim.replace('_'+str(bond_length),'') + '.json', 'w') as json_file:
    json.dump(molecules, json_file)