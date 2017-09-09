import sys
sys.path.insert(1,'./..')
import psi4 as psi4
from CCSD_Calculator import *

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
  "basis": 'sto-3g',
  "reference": "RHF",
  "print_MOs" : "True",
  "mp2_type": "conv",
  "scf_type": "pk",
  "roots_per_irrep": [40],
  'e_convergence': 1e-14,
  'r_convergence': 1e-14
}
psi4.set_options(opt_dict)
psi4.properties('ccsd', properties=['dipole','analyze'])

mol= CCSD_Calculator(psi4)
#Time-dependent CC2 calculation
mol.TDCC2(timeout)
#Time-dependent CCSD calculation
#mol.TDCCSD(timeout)
