from pyscf import gto, scf, fci, ci, symm
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import numpy as np
import pyscf
import json


def run_FCI(molecule: MolecularData, roots: int):
     pyscf_mol = molecule._pyscf_data['mol']
     pyscf_scf = molecule._pyscf_data['scf']
     
     pyscf_mol.symmetry = True
     pyscf_mol.build()

     # label orbital symmetries
     orbsym = scf.hf_symm.get_orbsym(pyscf_mol, pyscf_scf.mo_coeff)
     pyscf_mol.symmetry = False
     
     # generate FCI solver
     fci_solver = fci.addons.fix_spin_(fci.FCI(pyscf_mol, pyscf_scf.mo_coeff), shift=0.0, ss=1.0)
     fci_solver.nroots = roots
     
     # execute FCI solver
     f, d = fci_solver.kernel()
     
     fci_e = []
     print(' FCI:')
     # iterate over eigenstates 
     for i,x in enumerate(d):
          energy = f[i]
          mult = fci.spin_op.spin_square0(x, molecule.n_orbitals, molecule.n_electrons)[1]
          eigen_symm = fci.addons.guess_wfnsym(x, molecule.n_orbitals, molecule.n_electrons, orbsym)
          eigen_symm = symm.irrep_id2name('Coov',eigen_symm)

          fci_e.append( (energy, mult) )
          print('state {}, E = {},  2S+1 = {}, IrRep = {} '.format(i, round(energy,5), round(mult), eigen_symm))

     return fci_e


def run_CCSD(molecule: MolecularData, roots: int, ref: str):
     pyscf_mol = molecule._pyscf_data['mol']
     pyscf_scf = molecule._pyscf_data['scf']
     pyscf_ccsd = molecule._pyscf_data['ccsd']

     ccsd_e = []

     # Ground state
     ccsd_gs = molecule.ccsd_energy
     print("GS:", ccsd_gs)
     ccsd_e.append(ccsd_gs)

     # Excited States
     pyscf_ccsd.eeccsd(nroots=roots)
     if "s" in ref:
          # S->S excitation
          eS = pyscf_ccsd.eomee_ccsd_singlet(nroots=roots)[0]
          print("S:", [ccsd_gs+i for i in eS])
          for e in eS:
               ccsd_e.append(ccsd_gs+e)
          
     elif ref == 't1':
          # S->T excitation
          eT = pyscf_ccsd.eomee_ccsd_triplet(nroots=roots)[0]
          print("T:", [ccsd_gs+i for i in eT])
          for e in range(len(eT)):
               if e != 0:
                    ccsd_e.append(ccsd_gs + eT[e])

     else:
          raise ValueError('{} not a valid reference state.'.format(ref))

     return ccsd_e






if __name__ == "__main__":

     basis = 'sto-3g'    # Orbital basis set
     multiplicity = 1    # Spin of reference determinant
     n_points = 60   # Number of geometries 
     bond_length_interval = 4.0 / n_points

     run_scf = 1
     run_ccsd = 1

     molecules = []

     for point in range(1, n_points + 1):
          bond_length = bond_length_interval * float(point) + 0.2
          geometry = [('H', (0., 0., 0.)), ('Li', (0., 0., bond_length))]
          
          MolecularMeta = {}
          molecule = MolecularData(geometry,
                                   basis,
                                   multiplicity,
                                   description=str(round(bond_length, 2)))

          molecule = run_pyscf(molecule,
                              run_scf=run_scf)
                              
          MolecularMeta['Geo'] = bond_length
          MolecularMeta['FCI eigenstates'] = run_FCI(molecule, roots=8)

          molecules.append(MolecularMeta)
     
     with open('FCI_H1-Li1_sto-3g.json', 'w') as json_file:
          json.dump(molecules, json_file)
          