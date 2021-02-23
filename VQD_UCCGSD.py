import numpy as np
import timeit
import json

from qebab import *

from pytket.backends.ibm import AerBackend, AerStateBackend, IBMQBackend
from pytket.backends.projectq import ProjectQBackend

from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from openfermion.transforms import jordan_wigner

########### SETTINGS ##########
basis = 'sto-3g'                    # *Orbital basis set
multiplicity = 1                    # *Spin of reference determinant
n_points = 3                        # *Number of geometries 
bond_length_interval = 4.0 / n_points

run_scf = 1
delete_input = True
delete_output = True

pool = sUpCCGSD_Pool()              # Operator Pool
constructor = k_UCC_Ansatz          # Ansatz constructor
reference = "s0"                    # Qubit reference state
backend = ProjectQBackend()         # Backend for simulation
opt_method = "L-BFGS-B" #"Nelder-Mead"
opt_maxiter = 150

vqd_maxiter = 2
beta = 3.0
k = 2

########### MAIN QUANTUM ROUTINE ##########
print("\n")
print(" --------------------------------------------------------------------------")
print("                             VQD")
print(" --------------------------------------------------------------------------")

molecules = []
symbolic_ansatz = None

for point in [0.91]:#range(1, n_points + 1):
    bond_length = point#bond_length_interval * float(point) + 0.2
    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., bond_length))]
    
    MolecularMeta = {}
    molecule = MolecularData(geometry,
                            basis,
                            multiplicity,
                            description=str(round(bond_length, 2)))
    molecule = run_pyscf(molecule,
                         run_scf=run_scf)

    # Basic information
    jim = molecule.name
    print("===== Molecule {} =====".format(jim))
    MolecularMeta['Name'] = jim
    MolecularMeta['Geo'] = bond_length
    
    # Hamiltonian
    ham_qubit = jordan_wigner(molecule.get_molecular_hamiltonian())
    ham_qubit.compress() # Now in QubitOperator form

    print(' HF:', molecule.hf_energy)
    MolecularMeta['FCI eigenstates'] = run_FCI(molecule, vqd_maxiter+8)
    
    # Generate operator pool and state preparation circuit
    # (reusable for any geometry and eigenstate!)
    if not isinstance(symbolic_ansatz, Circuit):
        pool.init(n_orb=molecule.n_orbitals,
                  n_occ=molecule.get_n_alpha_electrons(),
                  n_vir=molecule.n_orbitals - molecule.get_n_alpha_electrons())
        
        ansatz = constructor(pool)
        symbolic_ansatz, symbols = ansatz.generate_Circuit(ref=reference,
                                                           k=k,
                                                           backend=backend)
        
    
    opt_eigen_params = []
    opt_eigenenergies = []
    opt_eigenansatze = []
    eigenvar = []
    eigentimes = []
    print("")

    for v_iter in range(0, vqd_maxiter):
        print(' Eigenstate: ', v_iter)
        # Generate energy objective function
        objective = gen_vqd_objective(ansatz=symbolic_ansatz,
                                      symbols=symbols,
                                      hamiltonian=ham_qubit,
                                      backend=backend,
                                      eigen_ansatze=opt_eigenansatze,
                                      beta=beta)
        # Generate initial guess params
        if len(molecules) == 0:
            initial_params = [np.random.uniform(0,0.01) for i in range(len(symbols))] 
        else:
            last_Meta = molecules[-1]
            last_params = last_Meta['Optimised parameters']
            initial_params = last_params[len(opt_eigenenergies)]

        # Energy optimisation
        start_t = timeit.default_timer()
        opt_energy, opt_params, var_result = energy_optimise(objective, initial_params, opt_method, opt_maxiter)
        stop_t = timeit.default_timer()
        t = stop_t - start_t

        print('  Energy: {} /Eh'.format(opt_energy))
        #print('  Accuracy v. FCI: {} /Eh'.format(opt_energy - molecule.fci_energy))
        opt_eigenenergies.append(opt_energy)

        print('  Variational minimisation steps: ',len(var_result))
        print('  Variational Time: {}/s'.format(t))
        
        eigenvar.append(var_result)
        eigentimes.append(t)

        param_dict = dict(zip(symbols, opt_params))
        opt_eigen_params.append(str(param_dict))

        eigen_ansatz = symbolic_ansatz.copy()
        eigen_ansatz.symbol_substitution(param_dict)
        opt_eigenansatze.append(eigen_ansatz)
        
        print("")

    MolecularMeta['Eigenstates'] = opt_eigenenergies
    MolecularMeta['Optimised parameters'] = opt_eigen_params
    MolecularMeta['Variational optimisation'] = eigenvar
    MolecularMeta['Time'] = eigentimes
    print(" Total Time {} /s ".format(sum(eigentimes)))
    print("")

    molecules.append(MolecularMeta)

with open(jim.replace('_'+str(bond_length),'') + '.json', 'w') as json_file:
    json.dump(molecules, json_file)
