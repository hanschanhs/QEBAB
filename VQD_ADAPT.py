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
basis = 'sto-3g'                   # *Orbital basis set
multiplicity = 1                   # *Spin of reference determinant
n_points = 5                       # *Number of geometries 
bond_length_interval = 4.0 / n_points

run_scf = 1
delete_input = True
delete_output = True

pool = sUpCCGSD_Pool()             # *Operator pool
constructor = ADAPT_VQD_Ansatz     # Ansatz constructor
reference = "t1"                   # *Qubit reference states
backend = AerStateBackend()        # Backend for simulation
opt_method = "L-BFGS-B"
opt_maxiter = 400

vqd_maxiter = 2                    # *No. of eigenstates
beta = 3.0

adapt_maxiter = 20
threshold = 0.01                   # *Threshold

########### MAIN QUANTUM ROUTINE ##########
print("\n")
print(" --------------------------------------------------------------------------")
print("                             VQD")
print(" --------------------------------------------------------------------------")

molecules = []
SymbolicAnsatz = None

for point in range(1, n_points + 1):
     bond_length = bond_length_interval * float(point) + 0.2
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
     ham_sparse = qubit_operator_sparse(ham_qubit, n_qubits=molecule.n_qubits)
     
     print(' HF:', molecule.hf_energy)
     MolecularMeta['FCI eigenstates'] = run_FCI(molecule, vqd_maxiter+8)
     
     # Generate operator pool
     if not isinstance(SymbolicAnsatz, Circuit):
          pool.init(n_orb=molecule.n_orbitals,
                    n_occ=molecule.get_n_alpha_electrons(),
                    n_vir=molecule.n_orbitals - molecule.get_n_alpha_electrons())

     opt_eigen_ops = [] # list of sublists (index of operators used in each eigenstate ansatz)
     opt_eigen_params = [] # list of dictionaries (symbols amplitudes for each eigenstate ansatz)

     opt_eigenenergies = [] # list of sublists (the drop of energy with ansatz growth)
     opt_eigenansatze = [] # list of optimised circuits
     eigentimes = []
     eigencirc = []
     print("")

     for v_iter in range(0, vqd_maxiter):
          print(' Start to grow ansatz for Eigenstate: ', v_iter)
          ansatz = constructor(pool)
          
          opt_params = []
          opt_eigenenergies.append([])
          eigentimes.append([])
          eigencirc.append([])

          for a_iter in range(0, adapt_maxiter):
               print('  Iteration: ', a_iter)
               converged, symbolic_ansatz, symbols = ansatz.generate_Circuit(ref=reference,
                                                                                       params=opt_params,
                                                                                       eigen_ansatze=opt_eigenansatze,
                                                                                       beta=beta,
                                                                                       ham_sparse=ham_sparse,
                                                                                       backend=backend,
                                                                                       threshold=threshold)

               if converged:
                    print("")
                    print(" Ansatz Growth Converged!")
                    print(" Number of operators in ansatz: ", len(ansatz.f_op))
                    print(" *Finished: %20.12f" % opt_energy)
                    print(" -----------Final ansatz----------- ")
                    coeffs = list(ansatz.symbols.values())
                    print(" %4s %40s %12s" %("#","Term","Coeff"))
                    for si in range(len(ansatz.symbols)):
                         s = ansatz.f_op[si]
                         opstring = ""
                         for t in s.terms:
                              opstring += str(t)
                              break
                         print(" %4i %40s %12.8f" %(si, opstring, coeffs[si]) )
                    print("")
                    print("")
                    break

               objective = gen_vqd_objective(ansatz=symbolic_ansatz,
                                             symbols=symbols,
                                             hamiltonian=ham_qubit,
                                             backend=backend,
                                             eigen_ansatze=opt_eigenansatze,
                                             beta=beta)

               initial_params = [np.random.uniform(0, 0.001) for i in range(len(symbols))]

               # Energy optimisation 
               start_t = timeit.default_timer()
               opt_energy, opt_params, var_result = energy_optimise(objective, initial_params, opt_method, opt_maxiter)
               stop_t = timeit.default_timer()
               t = stop_t - start_t

               print(' Energy: {} /Eh'.format(opt_energy))
               #print(' Accuracy v. FCI: {} /Eh'.format(opt_energy - molecule.fci_energy))
               opt_eigenenergies[-1].append(opt_energy)
               
               print(' Variational minimisation steps: ', len(var_result))
               print(' Variational Time: {}/s'.format(t))

               eigentimes[-1].append(t)
               circ_data = (symbolic_ansatz.depth(), symbolic_ansatz.depth_by_type(OpType.CX))
               eigencirc[-1].append(circ_data)
               print("")


          opt_eigen_ops.append(ansatz.op_indices)

          param_dict = dict(zip(symbols, opt_params))
          opt_eigen_params.append(str(param_dict))

          symbolic_ansatz.symbol_substitution(param_dict)
          opt_eigenansatze.append(symbolic_ansatz)

     MolecularMeta['Gate depth, CX depth'] = eigencirc
     MolecularMeta['VQD (Adaptive) Energies'] = opt_eigenenergies
     MolecularMeta['Parameters'] = opt_eigenparams
     MolecularMeta['Operators'] = opt_eigen_ops
     MolecularMeta['Time'] = eigentimes

     molecules.append(MolecularMeta)


if refs[0]=='s0':
     with open('sRef_sUp_' + jim.replace('_singlet_'+str(bond_length),'') + '.json', 'w') as json_file:
         json.dump(molecules, json_file)

elif refs[0]=='t1':
     with open('tRef_sUp_' + jim.replace('_singlet_'+str(bond_length),'') + '.json', 'w') as json_file:
         json.dump(molecules, json_file)

elif refs[0]=='open shell':
     with open('osRef_sUp_' + jim.replace('_singlet_'+str(bond_length),'') + '.json', 'w') as json_file:
         json.dump(molecules, json_file)

elif refs[0]=='closed shell':
     with open('csRef_sUp_' + jim.replace('_singlet_'+str(bond_length),'') + '.json', 'w') as json_file:
         json.dump(molecules, json_file)
