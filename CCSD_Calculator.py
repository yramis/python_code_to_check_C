# -*- coding: utf-8 -*-
# #################################################################
#
#
#                            Created by: Rachel Glenn
#                                 Date: 12/14/2016
#       This is the driver for CC2 or CCSD_Helper and Runge_Kutta to calculate
#       the time dependent dipole moment.
#
#       First it calculates the converged t1, t2,
#       Second the lam1, lam2
#       It uses the converged t1, t2, lam1, lam2 to calculate the real-time
#       single-electron density matrix
#
#####################################################################
import sys
sys.path.insert(0,'./..')
import psi4 as psi4
import numpy as np
from CCSD_Helper import *
from  CC2_Helper import  CC2_Helper
sys.path.append('/home/rglenn/newriver/buildpython/pandas')

########################################################
#                 Setup
#
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

##############################################
#
#
#              CCSD--Calculations--
#
#
##################################################

    def TDCCSD(self, timeout):#T1 equation
        mol = self.mol
        nmo = mol.nmo
        ndocc = mol.ndocc
        F =  mol.F_MO()
        v = 2*(nmo-ndocc)
        o = 2*ndocc
        psienergy = psi4.energy('CCSD')
        
############################################## 
#
#
#           t1 and t2 Amplitudes (CCSD):
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
        
        CC2_E, t1, t2 = mol.DIIS_solver(t1, t2, F, maxsize, maxiter, E_min)
        print("E_ccsd_plugin=", CC2_E + scf)
        print("difference between psi4 and plugin=", psienergy.real - (CC2_E + scf))
        mol.print_T_amp(t1, t2)
        
        psi4.driver.p4util.compare_values(psi4.energy('CCSD'), CC2_E+scf, 10, 'CCSD Energy')
        
############################################## 
#
#
#           lam1 and lam2 Amplitudes (CCSD):
#
#
##################################################

        maxiter = 30
        E_min = 1e-15 # minimum energy to match
        lam1 = t1
        lam2 = t2
        pseudo_E, lam1, lam2 = mol.DIIS_solver_Lam(t1, t2, lam1, lam2, F, maxsize, maxiter, E_min)
        
        ###Print out the L1 and L2 amplitudes and Pseudo energy
        print("E_pseudo_plugin=", pseudo_E)
        mol.print_L_amp(lam1, lam2)

##############################################
#
#
#           Time-dependent dipole matrix(CCSD):
#
#
##############################################
        #Start parameters
        w0 = 0.968635 #frequency of the oscillation
        A = 0.005#the amplitude of the electric field
        t0 = 0.0000 #the start time
        tf = 0.1 #the stop time, the actual stop time is governed by the timelength of the job
                     #Unless it completes enough steps to get to tf first. 
        dt = 0.0001 #time step
        precs = 15 #precision of the t1, t2, l1, l2 amplitudes
        mol.Runge_Kutta_solver(F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs)

##############################################
#
#
#              CC2--Calculations--
#
#
##############################################
    def TDCC2(self, timeout):#T1 equation
        mol = self.mol
        nmo = mol.nmo
        ndocc = mol.ndocc
        F =  mol.F_MO()
        v = 2*(nmo-ndocc)
        o = 2*ndocc
        psienergy = psi4.energy('CC2')

############################################## 
#
#
#           t1 and t2 Amplitudes (CC2):
#
#
###############################################
        #initialize t1 and t2
        scf, MP2, t2 = mol.MP2_E('Test')
        t1 = np.zeros( shape=(o, v), dtype=np.longdouble) 
        print("Escf=", scf)
        print("Emp2=", MP2-scf)
        print("Etot=", MP2)
        
        maxsize = 7 # number of t1 and t2 to store
        maxiter = 40 #max iterations incase it crashes
        E_min = 1e-15 # minimum energy to match

        mol_CC2 = CC2_Helper(psi4)
        CC2_E, t1, t2 = mol_CC2.DIIS_solver_CC2(t1, t2, F, maxsize, maxiter, E_min)
        print("E_cc2_plugin=", CC2_E + scf)
        print("difference between psi4 and plugin=", psienergy.real - (CC2_E + scf))
        mol.print_T_amp(t1, t2)
        psi4.driver.p4util.compare_values(psi4.energy('CC2'), CC2_E+scf, 10, 'CCSD Energy')
############################################## 
#
#
#           lam1 and lam2 Amplitudes (CC2):
#
#
##############################################
        maxiter = 30
        E_min = 1e-15 # minimum energy to match
        lam1 = t1
        lam2 = t2
        pseudo_E, lam1, lam2 = mol_CC2.DIIS_solver_Lam_CC2(t1, t2, lam1, lam2, F, maxsize, maxiter, E_min)
        print("E_pseudo_plugin=", pseudo_E)
        mol.print_L_amp(lam1, lam2)

##############################################
#
#
#           Time-dependent dipole matrix(CC2):
#
#
##############################################

        #Start parameters
        w0 = 0.968635 #frequency of the oscillation
        A = 0.005#the amplitude of the electric field
        t0 = 0.0000 #the start time
        tf = 10.1 #the stop time, the actual stop time is governed by the timelength of the job
                 #Unless it completes enough steps to get to tf first.
        dt = 0.0001 #time step
        precs = 15 #precision of the t1, t2, l1, l2 amplitudes
        mol_CC2.Runge_Kutta_solver_CC2(F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs)























