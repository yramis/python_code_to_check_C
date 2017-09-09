import sys
import os
sys.path.insert(1,'./..')
#sys.path.append(os.environ['HOME']+'/Desktop/workspace/psi411/psi4/objdir/stage/usr/local/lib')
#sys.path.append('/usr/local/psi4/bin')
#sys.path.append('/usr/local/psi4/lib')
#sys.path.append(os.environ['HOME']+'/miniconda2/lib/python2.7/site-packages')
#sys.path.append('/home/rglenn/blueridge/buildpsi/lib')
import cmath
import psi4 as psi4
import csv
#from opt_einsum import contract
from CCD_Helper import *
#from CCSD_Calculator import *
#if os.environ['SYSNAME']=='blueridge':
psi4.set_memory("5 GB")
#psi4.core.set_memory(int(62e9), False) #blueridge
#psi4.core.set_memory(int(3.5e9), False) 

timeout = float(sys.argv[1])/60
print("time in minutes is:", timeout)


#psi4.core.set_memory(int(100.e6), False) #my laptop
#psi4.core.clean()
numpy_memory = 2
#psi4.core.clean()
mol = psi4.geometry("""
O
H 1 0.9
H 1 0.9 2 104.5
symmetry c1
""")
#mol = psi4.geometry(molstring)

#psi4.set_options({'basis': '3-21g',
#                  'scf_type': 'pk',
#                  'mp2_type': 'conv',
#                  'freeze_core': 'false',
#                      "ANALYZE" : "True",
#                  'e_convergence': 1e-14,
#                  'd_convergence': 1e-14})



opt_dict = {
  "basis": 'sto-3g',
  "reference": "RHF",
  "print_MOs" : "True",
  "mp2_type": "conv",
  "roots_per_irrep": [40],
  "scf_type": "pk",
  'e_convergence': 1e-14,
  'r_convergence': 1e-14
}
#'6-31g'
#'sto-3g'
psi4.set_options(opt_dict)
psi4.energy('BCCD')
#psi4.properties('BCCD', properties=['rotation'])
#psi4.set_options("ANALYZE")
#psi4.("ANALYZE", 0);
#psi4.options.add_bool("ANALYZE", 0);
#psi4.properties('eom-cc2', properties=['oscillator_strength'])
psi4.core.set_output_file('output.dat', False)

pseudo = -0.068888224492060 #H2O sto-3g
pseudo = -0.140858583055215 #'3-21g
pseudo = -0.148311233718836 #'6-31g

mol= CCD_Calculator(psi4)

#Caculate the MP2 Energy
#mol.test_MP2()
#Converged T1, T2, L1, L2 amplitudes


mol.TDCCD(pseudo, timeout)
#mol.TDCCSD(pseudo, timeout)
