import sys
sys.path.insert(1,'./..')
import psi4 as psi4
from CC_Calculator import *

timeout = float(sys.argv[1])/60
print("time in minutes is:", timeout)
numpy_memory = 2
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

opt_dict = {
  "basis": "sto-3g",
  "reference": "RHF",
  "print_MOs" : "True",
  "mp2_type": "conv",
  "scf_type": "pk",
  "roots_per_irrep": [40],
  "e_convergence": 1e-14,
  "r_convergence": 1e-14
}
psi4.set_options(opt_dict)
psi4.properties('ccsd', properties=['dipole','analyze'])
#psi4.properties('cc2', properties=['dipole','analyze'])


#Start parameters
#w0 frequency of the oscillation
#A = 0.005#the amplitude of the electric field
#t0 = 0.0000 #the start time
#dt = 0.0001 #time step
#precs = 15 #precision of the t1, t2, l1, l2 amplitudes

mol = CC_Calculator(psi4, w0=0.968635,A=0.005,t0=0.0,dt=0.0001,precs=15)
#Time-dependent CC2 calculation
#mol.TDCC(timeout, 'CC2')
#Time-dependent CCSD calculation
mol.TDCC(timeout, 'CCSD')
