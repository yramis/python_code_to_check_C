################################################################
#
#
#                            Created by: Rachel Glenn
#                                 Date: 12/14/2016
#       This code calculates the converaged CCD energy, pseudo energy, the t2 and lam2
#       It also calculates the single particle density matrix using the converged t2 and lam2 
#
#
#####################################################################
import sys
import os
from copy import deepcopy
import numpy as np
import cmath
import pandas as pd
from pandas import *
import psi4 as psi4
sys.path.append(os.environ['HOME']+'/miniconda2/lib/python2.7/site-packages')
from opt_einsum import contract
import csv

import time
start = time.time()

class CCD_Helper(object):
    
    def __init__(self,psi,ndocc=None):

        self.counter = 0
        self.mol = psi4.core.get_active_molecule()
        mol = self.mol
        self.wfn = psi4.scf_helper('SCF',  return_wfn = True)
        self.scf_e = psi4.energy('scf')
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        self.nmo = self.wfn.nmo()
        self.ccsd_e = psi4.energy('cc2')
        self.S = np.asarray(self.mints.ao_overlap())
        
        #define ndocc
        # Orthoganlizer
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        self.A = np.asarray(A)
        self.ndocc =int(sum(mol.Z(A) for A in range(mol.natom())) / 2)
        self.wfn.Ca_subset("AO", "ACTIVE").print_out() 
        self.C = self.wfn.Ca()
        #self.C = self.wfn.Ca_subset("AO", "ALL")
        V = np.asarray(self.mints.ao_potential())
        T = np.asarray(self.mints.ao_kinetic())
        self.H = T + V
        self.occ = slice(2*self.ndocc)
        self.vir = slice(2*self.ndocc, 2*self.nmo)
        print(self.vir)
        #MO energies
        self.eps = np.asarray(self.wfn.epsilon_a()).repeat(2, axis=0)
        #self.TEI_MO = np.asarray(self.mints.mo_spin_eri(self.C, self.C))
        #self.TEI = self.TEI_MO()
        self.TEI = np.asarray(self.mints.ao_eri())


    def test_MP2(self):
        mol = self.mol
        scf, MP2, T2 = mol.MP2_E('Test')
        return MP2
    
    def F_MO(self, H=None, C=None):
        if H is None: H = self.H
        if C is None: C = self.C
        TEI = self.TEI_MO(C)
        occ = self.occ
        nmo =self.nmo
        # Update H, transform to MO basis and tile for alpha/beta spin
        H = contract('vi,uv,uj->ij', C, H, C)
        H = H.repeat(2, axis=1).repeat(2, axis=0)
        H = H*np.tile(np.identity(2),(nmo,nmo))
        F= H + contract('pmqm->pq', TEI[:, occ, :, occ])
        return F
    
    def TEI_MO(self, C=None):
        if C is None: C = self.C
        return np.asarray(self.mints.mo_spin_eri(C, C))


    def MP2_E(self, alpha, H=None, C=None):  
        #alpha is a text variable to select the output
        if H is None: H = self.H
        if C is None: C = self.C 
        eps = self.MO_E(H,C)
        o = self.occ
        v = self.vir
        self.TEI = self.TEI_MO(C)
        TEI = self.TEI
        Dem = eps[o].reshape(-1, 1, 1, 1) + eps[o].reshape(-1, 1, 1) - eps[v].reshape(-1, 1) - eps[v]
        Dem = 1/Dem
        T2 = contract('ijab,ijab->ijab', TEI[o, o, v, v],Dem)
        MP2 = contract('ijab,ijab->', T2, TEI[o, o, v, v])
        T2 = TEI[o, o ,v, v]*Dem
        MP2 = np.sum(TEI[o, o, v, v]*T2)
        #print MP2

        MP2_E = self.scf_e + 1/4.0*MP2
        
        if alpha is 'Test':
            psi4.p4util.compare_values(psi4.energy('mp2'), MP2_E, 10, 'MP2_Energy')
            pass
        return self.scf_e, MP2_E, T2

    def GenS12(self): 
        # Update S, transform to MO basis and tile for alpha/beta spin
        S = self.S
        nmo = self.nmo
        S = S.repeat(2, axis=1).repeat(2, axis=0)
        S = S*np.tile(np.identity(2),(nmo,nmo))
        evals, evecs = np.linalg.eigh(S)
        nmo = self.nmo
        
        Ls = np.zeros(shape=(2*nmo,2*nmo))
        Lsplus = np.zeros(shape=(2*nmo,2*nmo))    
          
        for i in range (2*nmo):
            Ls[i][i]= 1/np.sqrt(evals[i])
            Lsplus[i][i]= np.sqrt(evals[i])
            
        S12 = contract('il,lk,jk->ij', evecs, Ls, evecs)
        S12plus = contract('il,lk,jk->ij', evecs, Lsplus, evecs)        
        return S12, S12plus
        
        
        
    def MO_E(self, H=None, C=None):  
        if H is None: H = self.H
        if C is None: C = self.C 
        F = self.F_MO(H,C)
        evals, evecs = np.linalg.eigh(F)
        return evals
    
    
    def Fae(self, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = F[v, v].copy()
        term2 =-0.5*contract('mnef,mnaf->ae', TEI[o, o, v, v], t2)
        total = term1 + term2
        return total
    
    #Build Foo
    def Fmi(self, t2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI 
        term1 = F[o, o].copy()
        term2 = 0.5*contract('mnef,inef->mi', TEI[o, o, v, v], t2)
        total = term1 + term2 
        return total
    
    #Build Fov    
    def Fme(self, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = F[o, v].copy()
        return term1

    def Wmnij(self, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, o, o, o].copy()
        term2 = 0.25*contract('mnef,ijef->mnij', TEI[o, o, v, v], t2)
        total = term1 + term2 
        return total   

    def Wabef(self, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[v, v, v, v].copy()
        term2 = 0.25*contract('mnef,mnab->abef', TEI[o, o, v, v], t2)
        total = term1 + term2
        return total

    def Wmbej(self, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, v, v, o].copy()
        term2 = -0.5*contract('mnef,jnfb->mbej', TEI[o, o, v, v], t2)
        total = term1 + term2
        return total

    def T2eq_rhs(self, t2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI         
        fae = self.Fae(t2, F) 
        fmi = self.Fmi(t2, F)  
        fme = self.Fme(t2, F) 
        wmnij = self.Wmnij(t2, F)
        wabef = self.Wabef(t2, F)
        wmbej = self.Wmbej(t2, F)
        #All terms in the T2 Equation
        term1 = TEI[o, o, v, v].copy()
        term2a = contract('be,ijae->ijab', fae, t2) 
        term2 = term2a - term2a.swapaxes(2, 3) #swap ab
        term3a = -contract('mj,imab->ijab', fmi, t2) 
        term3 = term3a - term3a.swapaxes(0, 1) #swap ij
        term44 = 0.5*contract('mnij,mnab->ijab', wmnij, t2)
        term55 = 0.5*contract('abef,ijef->ijab', wabef, t2)   
        term6tmp = contract('mbej,imae->ijab', wmbej, t2)
        term6 =  term6tmp - term6tmp.swapaxes(2, 3)  - term6tmp.swapaxes(0, 1)  + term6tmp.swapaxes(0, 1).swapaxes(2, 3)
        total = term1 + term2 + term3 + term44 + term55 + term6 
        return total 


    def corrected_T2(self, t2, dt2, F):
        o = self.occ
        v = self.vir
        eps, evecs = np.linalg.eigh(F)
        
        Dem = eps[o].reshape(-1, 1, 1, 1)
        Dem = Dem + eps[o].reshape(-1, 1, 1)
        Dem = Dem - eps[v].reshape(-1, 1) 
        Dem = Dem - eps[v]
        Dem = 1/Dem
        self.print_T_amp(Dem)
        t2 = t2 + contract('ijab,ijab->ijab', dt2, Dem)
        return t2

    #Routine for DIIS solver, builds all arrays(maxsize) before B is computed    
    def DIIS_solver(self, t2, F, maxsize, maxiter, E_min):
            print("This is T2")
            self.print_T_amp(t2)
            #Store the maxsize number of t2
            T2rhs = self.T2eq_rhs(t2, F)
            t2 = self.corrected_T2(t2, T2rhs, F)
            t2stored = [t2.copy()]
            errort2 = []
            
            for n in range(1, maxsize+1):  
                T2rhs = self.T2eq_rhs(t2, F)
                t2 = self.corrected_T2(t2, T2rhs, F)
                t2stored.append(t2.copy())
                
                errort2.append(t2stored[n]- t2stored[n-1])

             # Build B
            B = np.ones((maxsize + 1, maxsize + 1)) * -1
            B[-1, -1] = 0
            for z in range(1, maxiter):
                CCSD_E_old = self.CCSD_Corr_E(t2, F)
                for n in range(maxsize):
                    for m in range(maxsize):
                        b = contract('ijab,ijab->', errort2[m], errort2[n])
                        B[n, m] = b
    
                # Build residual vector
                A = np.zeros(maxsize + 1)
                A[-1] = -1

                c = np.linalg.solve(B, A)
                
                # Update t2 
                t2 = 0.0*t2
                for n in range(maxsize):
                    t2 += c[n] * t2stored[n+1]

                oldt2 = t2.copy()
                #test if converged
                CCSD_E = self.CCSD_Corr_E(t2, F)
                diff_E = CCSD_E - CCSD_E_old
                if (abs(diff_E) < E_min):
                    break
                #update t2 list
                T2rhs = self.T2eq_rhs(t2, F)
                t2 = self.corrected_T2(t2, T2rhs, F)
                t2stored.append(t2.copy())
                
                errort2.append(t2 - oldt2)
                
                print("inter =", z,  "\t", "CCSD_E =", CCSD_E,"diff=", diff_E)
                del t2stored[0]
                del errort2[0]
            return CCSD_E, t2
        
    #Calculate the CCSD energy 
    def CCSD_Corr_E(self, t2, F):
        o = self.occ
        v = self.vir
        TEI = self.TEI
        term1 = 0.25*contract('ijab,ijab->', TEI[o, o, v, v], t2)
        return term1
        


    def LSWmnij(self, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = 0.5*TEI[o, o, o, o].copy()
        term2 = 0.25*contract('ijfe,mnfe->ijmn', TEI[o, o, v, v], t2)
        total = term1 + term2 
        return total

    def LSWabef(self, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = 0.5*TEI[v, v, v, v].copy()
        term2 = 0.25*contract('nmab,nmef->efab', TEI[o, o, v, v], t2)
        total = term1 + term2
        return total


    def LSWieam(self,  t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, v, v, o].copy()
        term2 =  -contract('ijab,mjbe->ieam', TEI[o, o, v, v], t2)
        total = term1 + term2 
        return total

    def Gfe(self, t2, lam2):
        return -0.5*contract('mnfb,mneb->fe', lam2, t2)
        
    def Gmn(self, t2, lam2):
        return 0.5*contract('njed,mjed->nm', lam2, t2)

    def lam2eq_rhs(self, t2, lam2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI     
        Feb = self.LRFea(t2, F)
        Fjm = self.LRFim(t2, F)
        Wijmn = self.LSWmnij(t2, F)
        Wefab = self.LSWabef(t2, F)
        Wjebm = self.LSWieam(t2, F)
        Fjb = self.Fme(t2, F)
        Gbe = self.Gfe(t2, lam2)
        Gmj = self.Gmn(t2, lam2) 
        
        term1 = TEI[o, o, v, v]
        term2a = contract('eb,ijae->ijab', Feb, lam2)
        term2 = term2a - term2a.swapaxes(2,3)
        term3a = -contract('jm,imab->ijab', Fjm, lam2)
        term3 = term3a - term3a.swapaxes(0,1)
        term4 = contract('ijmn,mnab->ijab', Wijmn, lam2)
        term5 = contract('efab,ijef->ijab', Wefab, lam2)
        #term8 and 9
        term89a = contract('jebm,imae->ijab', Wjebm, lam2) 
        term89 = term89a 
        term89 = term89 - term89a.swapaxes(2,3) 
        term89 = term89 - term89a.swapaxes(0,1) 
        term89 = term89 + term89a.swapaxes(0,1).swapaxes(2,3) 
        term10a = contract('ijfb,af->ijab', TEI[o, o, v, v], Gbe)
        term10 = term10a - term10a.swapaxes(2,3)
        term11a = -contract('mjab,im->ijab', TEI[o, o, v, v], Gmj)
        term11 = term11a - term11a.swapaxes(0,1)
        total = term1 + term2 + term3 + term4 + (term5 + term6) + term7 + term89
        total = total + term10 + term11
        return total
        
    def CCSD_pseudo_E(self, t2, lam2, F):
        o = self.occ
        v = self.vir
        TEI = self.TEI
        term1 = 0.25*contract('ijab,ijab->', TEI[o, o, v, v], lam2)
        return term1                         
        
    def corrected_lam2(self, lam2, dlam2, F):
        o = self.occ
        v = self.vir
        eps, evecs = np.linalg.eigh(F)
        Dem = eps[o].reshape(-1, 1, 1, 1)
        Dem = Dem + eps[o].reshape(-1, 1, 1)
        Dem = Dem - eps[v].reshape(-1, 1) 
        Dem = Dem - eps[v]
        Dem = 1/Dem
        lam2 = lam2 + contract('ijab,ijab->ijab', dlam2, Dem)
        return lam2

    
    def remove_dup(self, t2):
        s = []
        for i in t2:
            i = round(i,10)
            if i not in s:
                s.append(i)
        return s

    def print_T_amp(self, t2):
        sort_t2 = sorted(t2.ravel())
        sort_t2 = self.remove_dup(sort_t2)
        print("\n   The largest T2 values are:")

        for x in range(len(sort_t2)):
            if (round(sort_t2[x],5) ==0e5 ): #or x % 2 or x > 20):
                pass
            else:
                print('\t', ('% 5.10f' %  sort_t2[x]))  
                
    def print_L_amp(self, lam2):
        sort_lam2 = sorted(lam2.ravel())
        sort_lam2 = remove_dup(sort_lam2)
        
        print("\n   The largest lam2 values are:")
        for x in range(len(sort_lam2)):
            if (round(sort_lam2[x],2) ==0.00 or x % 2 or x > 20):
                pass
            else:
                print('\t', ('% 5.10f' %  sort_lam2[x]))   
                
    def DIIS_solver_Lam(self, t2, lam2, F, maxsize, maxiter, E_min): 
            #Store the maxsize number of  t2
            lam2rhs = self.lam2eq_rhs(t2, lam2, F)
            lam2 = self.corrected_lam2(lam2, lam2rhs, F)
            lam2stored = [lam2.copy()]
            errort2 = []
            
            for n in range(1, maxsize+1):  
                lam2rhs = self.lam2eq_rhs(t2, lam2, F)
                lam2 = self.corrected_lam2(lam2, lam2rhs, F)
                lam2stored.append(lam2.copy())
                
                errort2.append(lam2stored[n]- lam2stored[n-1])

             # Build B
            B = np.ones((maxsize + 1, maxsize + 1)) * -1
            B[-1, -1] = 0
            for z in range(1, maxiter):
                CCSD_E_old  = self.CCSD_pseudo_E(t2, lam2, F)
                for n in range(maxsize):
                    for m in range(maxsize):
                        b = contract('ijab,ijab->', errort2[m], errort2[n])
                        B[n, m] =  b
    
                # Build residual vector
                A = np.zeros(maxsize + 1)
                A[-1] = -1

                c = np.linalg.solve(B, A)
                
                # Update  t2 
                lam2 = 0.0*lam2
                for n in range(maxsize):
                    lam2 += c[n] * lam2stored[n+1]

                oldlam2 = lam2.copy()
                #test if converged
                CCSD_E  = self.CCSD_pseudo_E(t2, lam2, F)
                diff_E = CCSD_E - CCSD_E_old
                if (abs(diff_E) < E_min):
                    break
                #update t2 list
                lam2rhs = self.lam2eq_rhs(t2, lam2, F)
                lam2 = self.corrected_lam2(lam2, lam2rhs, F)
                lam2stored.append(lam2.copy())
                
                errort2.append(lam2 - oldlam2)
                
                print("inter =", z,  "\t", "Pseudo_E =", CCSD_E,"diff=", diff_E)
                del lam2stored[0]
                del errort2[0]
            return CCSD_E, lam2
    #T2 Runge-Kutta function 
    def ft2(self, t, dt, t2, F, Vt):
        k1 = self.T2eq_rhs(t2, F + Vt(t))
        k2 = self.T2eq_rhs(t2 + dt/2.0*k1, F + Vt(t + dt/2.0))  
        k3 = self.T2eq_rhs(t2 + dt/2.0*k2, F + Vt(t + dt/2.0)) 
        k4 = self.T2eq_rhs(t2 + dt*k3,  F + Vt(t + dt)) 
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
                 
   #L2 Runge-Kutta function  
    def fL2(self, t, dt, t2, lam2, F, Vt):
        k1 = self.lam2eq_rhs(t2, lam2, F + Vt(t))
        k2 = self.lam2eq_rhs(t2, lam2 + dt/2.0*k1, F + Vt(t + dt/2.0))  
        k3 = self.lam2eq_rhs(t2, lam2 + dt/2.0*k2, F + Vt(t + dt/2.0)) 
        k4 = self.lam2eq_rhs(t2, lam2 + dt*k3, F + Vt(t + dt)) 
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)      
############END functions for Runge-Kutta#############


    def Save_parameters(self, w0, A, t0, t, dt, precs, i, a):
        save_dat =  pd.DataFrame( columns = ( 'w0', 'A', 't0','dt','precs', 'i', 'a')) 
        save_dat.loc[1] = [w0, A, t, dt, precs, i, a]
        save_dat.to_csv('Parameters.csv',float_format='%.10f')
        
    def write_2data(self, F, FileName, precs):
        with open(FileName, 'w') as outcsv:
        #configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            for i in range(F.shape[0]):
                for a in range(F.shape[1]):
                #Write item to outcsv
                    writer.writerow([i, a, np.around(F[i][a], decimals=precs) ])

    def write_4data(self, F, FileName, precs):
        with open(FileName, 'w') as outcsv:
        #configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            for i in range(F.shape[0]):
                for j in range(F.shape[1]):
                    for a in range(F.shape[2]):
                        for b in range(F.shape[3]):
                        #Write item to outcsv
                            writer.writerow([i, j, a, b, np.around(F[i][j][a][b], decimals=precs) ])




    def Save_data(self, F, t2, lam2, data, timing, precs, restart):
        if restart is None: 
            data.to_csv('H2O.csv')
            timing.to_csv('timing.csv')
        else:
            with open('H2O.csv', 'a') as f:
                data.to_csv(f, header=False)
            with open('timing.csv', 'a') as f:
                timing.to_csv(f, header=False) 
              
        self.write_2data(F.real, 'F_real.dat', precs)
        self.write_2data(F.imag, 'F_imag.dat', precs)
        self.write_4data(t2.real, 't2_real.dat', precs)
        self.write_4data(t2.imag, 't2_imag.dat', precs)
        self.write_4data(lam2.real, 'lam2_real.dat', precs)
        self.write_4data(lam2.imag, 'lam2_imag.dat', precs)


    def Runge_Kutta_solver(self, F, t2, lam2, w0, A, t0, tf, dt, timeout, precs, restart=None):
        #Setup Pandas Data and time evolution
       
        data =  pd.DataFrame( columns = ('time', 'mu_real', 'mu_imag')) 
        timing =  pd.DataFrame( columns = ('total', 't2', 'l2')) 
        
        #        ##Electric field, it is in the z-direction for now      
        def Vt(t):
            mu = self.Defd_dipole()

            return -A*mu[2] #*np.sin(2*np.pi*w0*t)*np.exp(-t*t/5.0)   
        t = t0
        i=0
        start = time.time()
        m=1.0
        #Do the time propagation
        while t < tf:
            L2min = np.around(lam2, decimals=precs) 
            dt = dt/m
            itertime_t2 = 0
            for n in range(int(m)):
                t2min = np.around(t2, decimals=precs) 
                itertime = time.time()
                dt2 = -1j*self.ft2(t, dt, t2, F, Vt) #Runge-Kutta
                itertime_t2 = -itertime + time.time()
            dt = m*dt
            itertime = time.time()
            dL2 = 1j*self.fL2(t, dt, t2, lam2, F, Vt)  #Runge-Kutta
            itertime_l2 = -itertime  + time.time()
            total = itertime_t2 + itertime_l2
            timing.loc[i] = [total, itertime_t2, itertime_l2 ]
            t2 = t2min + dt2
            lam2 = L2min + dL2
            i += 1
            t =t0 + i*dt
            stop = time.time()-start
            mua = self.dipole_moment(t2, lam2, F)
            data.loc[i] = [t, mua[2].real, mua[2].imag  ]
            print(t, mua[2])
            
            if abs(stop)>0.9*timeout*60.0:
                
                self.Save_data(F, t2min, L2min, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t-dt, dt, precs, t2.shape[0], t2.shape[1])
    
                break
            #Calculate the dipole moment using the density matrix

            
            if abs(mua[2].real) > 100:
                self.Save_data(F, t2min, L2min, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t-dt, dt, precs, t2.shape[0], t2.shape[1])
                break
            
        stop = time.time()
        print("total time non-adapative step:", stop-start)
        print("total steps:", i)
        print("step-time:", (stop-start)/i)





class CCD_Calculator(CCD_Helper):
    
    def __init__(self,psi,ndocc=None):
        self.mol = CCD_Helper(psi)
        mol = self.mol
        self.ndocc = mol.ndocc  
            
    def test_MP2(self):
        mol = self.mol
        scf, MP2, T2 = mol.MP2_E('Test')
        return MP2
        #print "This is the energy", MP2
        
    def TDCCD(self, pseudo, timeout):#T1 equation
        
        mol = self.mol
        nmo = mol.nmo
        ndocc = mol.ndocc
        F =  mol.F_MO()
        #TEI = np.longdouble(mol.TEI_MO())  
        v = 2*(nmo-ndocc)
        o = 2*ndocc
        scf, MP2, t2 = mol.MP2_E('Test')
        psienergy = psi4.energy('BCCD')
        #psienergy = psi4.energy('CCSD')
        
        

############################################## 
#
#
#           t2 Amplitude (CCD):
#
#
##################################################       
        #initialize t2
        scf, MP2, t2 = mol.MP2_E('Test')
        print("Escf=", scf)
        print("Emp2=", MP2-scf)
        print("Etot=", MP2)
        
        
        maxsize = 7 # number of t2 to store
        maxiter = 40 #max iterations incase it crashes
        E_min = 1e-15 # minimum energy to match
        
        #DIIS solver
        CC2_E, t2 = mol.DIIS_solver(t2, F, maxsize, maxiter, E_min)
        
        print("E_ccsd_me=", CC2_E + scf)
        print("difference between psi4 and me=", psienergy.real - (CC2_E + scf))
        mol.print_T_amp(t2)
        
        #psi4.driver.p4util.compare_values(psi4.energy('BCCD'), CC2_E+scf, 10, 'CCD Energy')
############################################## 
#
#
#           lam2 Amplitude (CCD):
#
#
##################################################    
        maxiter = 30
        E_min = 1e-15 # minimum energy to match
        #Setup the initial lambda values
        lam2 = t2
        #Solve for the converged  L2
        pseudo_E, lam2 = mol.DIIS_solver_Lam(t2, lam2, F, maxsize, maxiter, E_min)

        ###Print out the L2 amplitudes and Pseudo energy
        print("E_pseudo_me=", pseudo_E) 
        print("E_pseudo_psi4=", pseudo)
        print("difference between psi4 and me=", pseudo - (pseudo_E))
        mol.print_L_amp(lam2)

##############################################
#
#
#           Time-dependent dipole matrix
#
#
##################################################

        mol.Test_T1_rhs(t2, lam2, F)

        #Start parameters
        w0 = 0.968635 #frequency of the oscillation and transition frequency
        A = 0.005#the amplitude of the electric field
        t0 = 0.0000 #the start time
        tf = 0.1 #the stop time, the actual stop time is governed by the timelength of the job
                     #Unless it completes enough steps to get to tf first. 
        dt = 0.0001 #time step
        precs = 8 #precision of the t2, l2 amplitudes
        ####4th-order Rosenbrock "Parallel exponential Rosenbrock methods, 
        #Vu Thai Luana, Alexander Ostermannb"
        ######4th-order Runge-Kutta   
        mol.Runge_Kutta_solver(F, t2, lam2, w0, A, t0, tf, dt, timeout, precs)

        end = start - time.time()
        print("total_time", end)
