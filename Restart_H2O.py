import sys
import os
sys.path.insert(0,'./..')
import psi4 as psi4
from CC_Calculator import *
import csv
import pandas as pd

#psi4.core.set_memory(int(62e9), False) #blueridge

timeout = float(sys.argv[1])/60
print("allocated time in minutes is:", timeout)

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
  "mp2_type": "conv",
  "roots_per_irrep": [40],
  "scf_type": "pk",
  "e_convergence": 1e-14,
  "r_convergence": 1e-14
}
psi4.set_options(opt_dict)
psi4.core.set_output_file('output.dat', False)

mol = CC_Calculator(psi4)
mol.TDCC_restart(timeout);
