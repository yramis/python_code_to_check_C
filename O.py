import sys
import os
sys.path.insert(0,'./..')
sys.path.append(os.environ['HOME']+'/Desktop/workspace/psi411/psi4/objdir/stage/usr/local/lib')
sys.path.append(os.environ['HOME']+'/miniconda2/lib/python2.7/site-packages')
sys.path.append('/home/rglenn/blueridge/buildpsi/lib')
import cmath
import psi4 as psi4

#from opt_einsum import contract
from CCSD_Calculator import *
#if os.environ['SYSNAME']=='blueridge':
#psi4.core.set_memory(int(62e9), False)
  #  psi4.core.set_memory(int(3.5e9), False)
    #psi4.core.set_memory(int(0.5e9), False)
#psi4.core.clean()
numpy_memory = 2
#psi4.core.clean()
mol = psi4.geometry("""
O
symmetry c1
""")
#mol = psi4.geometry(molstring)

#psi4.set_options({'basis': '3-21g',
#                  'scf_type': 'pk',
#                  'mp2_type': 'conv',
#                  'freeze_core': 'false',
#                  'e_convergence': 1e-14,
#                  'd_convergence': 1e-14})



opt_dict = {
  "basis": '6-31g',
  "reference": "UHF",
  "mp2_type": "conv",
  "roots_per_irrep": [40],
  "scf_type": "pk",
  'e_convergence': 1e-14,
  'r_convergence': 1e-14
}
#'6-31g'
#'sto-3g'
psi4.set_options(opt_dict)
psi4.property('ccsd', properties=['dipole'])
#psi4.property('eom-cc2', properties=['oscillator_strength'])
#psi4.core.set_output_file('output.dat', False)

#pseudo =#sto-3g
#pseudo = #'3-21g
pseudo =-0.067803910686176 #'6-31g

mol= CCSD_Calculator(psi4)

#Caculate the MP2 Energy
#mol.test_MP2()
#Converged T1, T2, L1, L2 amplitudes
mol.T1(pseudo)
