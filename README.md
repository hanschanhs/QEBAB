# Quantum Eigensolver Building on Achievments of Both quantum computing and chemistry (QEBAB)
Tools for building quantum chemistry Variational Quantum Eigensolver (VQE) calculations. 

## Setup
1. (Recommended) Create a new conda environment
```console
$ conda create -n qebab python
$ conda activate qebab
```
2. Clone repo
```console
$ mkdir QEBAB
$ cd QEBAB
$ git clone https://github.com/hanschanhs/QEBAB
```
3. Install required packages 

## Availabe tools
_Operator pool types_
- Unitary Coupled Cluster Singles and Doubles (UCCSD)
- Unitary Coupled Cluster Generalised Singles and Doubles (UCCGSD)
- Unitary Coupled Cluster Generalised Singles and paired Doubles (UpCCGSD)

_Ansatz build classes_
- _k_-UCC type: string up operators from a pool, and repeat _k_ times, each time with a separate parameterisation. For conventional (disentangled) 1-Trotter UCCSD/UCCGSD, set _k_=1. 
- ADAPT type: adaptive ansatz growth 

_Reference circuits_
- singlet (Hartree-Fock) reference
- triplet (first excited) reference

Build examples use Variational Quantum Deflation (VQD).
For ground state VQE calculations, set the number of eigenstates recovered to 1.  
