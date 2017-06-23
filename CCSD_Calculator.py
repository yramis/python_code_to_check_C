# -*- coding: utf-8 -*-
# #################################################################
#
#
#                            Created by: Rachel Glenn
#                                 Date: 12/14/2016
#       This is the driver for CCSD_Helper and Runge_Kutta to calculate the time dependent dipole moment
#       First it calculates the converged t1, t2,
#       Second the lam1, lam2
#       It uses t1, t2, lam1, lam2 to calculate the real-time single-electron density matrix
#
#####################################################################
import sys
import os
sys.path.insert(0,'./..')
#sys.path.append('/Users/silver/Desktop/workspace/python_RTDHF/ps411/myscf_pluginUHF')
#sys.path.append('$HOME/Desktop/workspace/psi411/psi4/objdir/stage/psi4-build/bin/psi4')
#sys.path.append(os.environ['HOME']+'/Desktop/workspace/psi411/psi4/objdir/stage/usr/local/lib')
#sys.path.append('/home/rglenn/blueridge/buildpsi/lib')
import cmath
import psi4 as psi4
import numpy as np
from CCSD_Helper import *
sys.path.append('/home/rglenn/newriver/buildpython/pandas')
import pandas as pd
#import matplotlib.pyplot as plt
import time
start = time.time()
########################################################
#                                            Setup                                                                      #
########################################################
class CCSD_Calculator(object):
    
    def __init__(self,psi,ndocc=None):
        self.mol = CCSD_Helper(psi)
        mol = self.mol
        self.ndocc = mol.ndocc  
            
    def test_MP2(self):
        mol = self.mol
        scf, MP2, T2 = mol.MP2_E('Test')
        return MP2
        #print "This is the energy", MP2
        
    def TDCCSD(self, pseudo, timeout):#T1 equation
        mol = self.mol
        nmo = mol.nmo
        ndocc = mol.ndocc
        F =  mol.F_MO()
        #TEI = np.longdouble(mol.TEI_MO())  
        v = 2*(nmo-ndocc)
        o = 2*ndocc
        psienergy = psi4.energy('CCSD')
        
############################################## 
#
#
#           t1 and t2 Amplitudes:
#
#
##################################################       
        #initialize t1 and t2
        scf, MP2, t2 = mol.MP2_E('Test')
        t1 = np.zeros( shape=(o, v), dtype=np.longdouble) 
        print("Escf=", scf)
        print("Emp2=", MP2-scf)
        print("Etot=", MP2)
        
        
        maxsize = 7 # number of t1 and t2 to store
        maxiter = 40 #max iterations incase it crashes
        E_min = 1e-15 # minimum energy to match
        
        #DIIS solver
        CCSD_E, t1, t2 = mol.DIIS_solver(t1, t2, F, maxsize, maxiter, E_min)
        
        #Print out the T1 and T2 amplitudes and CCSD energy
        #print "E_ccsd_psi4=", mol.ccsd_e
        print("E_ccsd_me=", CCSD_E + scf)
        print("difference between psi4 and me=", psienergy.real - (CCSD_E + scf))
        mol.print_T_amp(t1, t2)
        
        psi4.driver.p4util.compare_values(psi4.energy('CCSD'), CCSD_E+scf, 10, 'CCSD Energy')
############################################## 
#
#
#           lam1 and lam2 Amplitudes:
#
#
##################################################    
        maxiter = 30
        E_min = 1e-15 # minimum energy to match
        #Setup the initial lambda values
        lam1 = t1
        lam2 = t2
        #Solve for the converged L1 and L2
        #pseudo_E, lam1, lam2 = mol.NO_DIIS_solve_lamr(t1, t2, lam1, lam2, F, maxsize, maxiter, E_min)
        pseudo_E, lam1, lam2 = mol.DIIS_solver_Lam(t1, t2, lam1, lam2, F, maxsize, maxiter, E_min)

        ###Print out the L1 and L2 amplitudes and Pseudo energy
        print("E_pseudo_me=", pseudo_E) 
        print("E_pseudo_psi4=", pseudo)
        print("difference between psi4 and me=", pseudo - (pseudo_E))
        mol.print_L_amp(lam1, lam2)

##############################################
#
#
#           Time-dependent dipole matrix
#
#
##################################################
        import sys
        print dir()
            #for name in dir():
            #if not name.startswith('_'):
            #del globals()[name]
        del maxiter, maxsize, psienergy,  MP2, E_min, CCSD_E, pseudo, pseudo_E, scf, timeout, v
        print dir()
        

        mol.Test_T1_rhs(t1, t2, lam1, lam2, F)







