################################################################
#
#
#                            Created by: Rachel Glenn
#                                 Date: 12/14/2016
#       This code calculates the converaged CCSD energy, pseudo energy, the t1, t2, lam1, and lam2
#       It also calculates the single particle density matrix using the converged t1, t2, lam1, and lam2 
#
#
#####################################################################
import sys
import os
from copy import deepcopy
import numpy as np
import cmath
import pandas as pd
#sys.path.append(os.environ['HOME']+'/Desktop/workspace/psi411/psi4/objdir/stage/usr/local/lib')
#sys.path.append('/home/rglenn/blueridge/buildpsi/lib')
#sys.path.append('/home/rglenn/newriver/buildpython/pandas/pandas')
from pandas import *
import psi4 as psi4
sys.path.append(os.environ['HOME']+'/miniconda2/lib/python2.7/site-packages')
from opt_einsum import contract
import time
import csv

class CCSD_Helper(object):
    
    def __init__(self,psi,ndocc=None):
       
        self.counter = 0
        self.mol = psi4.core.get_active_molecule()
        mol = self.mol
        self.wfn = psi4.scf_helper('SCF',  return_wfn = True)
        self.scf_e = psi4.energy('scf')
        #self.scf_e = wfn.energy()
        #self.scf_e, self.wfn = psi4.energy('scf', return_wfn = True)
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        self.nmo = self.wfn.nmo()
        self.ccsd_e = psi4.energy('ccsd')
        self.S = np.asarray(self.mints.ao_overlap())
        #print mol.nuclear_repulsion_energy()
        #print mol.nuclear_dipole()
        
        #define ndocc
        # Orthoganlizer
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        self.A = np.asarray(A)
        self.ndocc =int(sum(mol.Z(A) for A in range(mol.natom())) / 2)
      
        self.C = self.wfn.Ca()
        #self.C = self.wfn.Ca_subset("AO", "ALL")
        V = np.asarray(self.mints.ao_potential())
        T = np.asarray(self.mints.ao_kinetic())
        self.H = T + V
        self.occ = slice(2*self.ndocc)
        self.vir = slice(2*self.ndocc, 2*self.nmo)
        print self.vir
        #MO energies
        self.eps = np.asarray(self.wfn.epsilon_a()).repeat(2, axis=0)
        #self.TEI_MO = np.asarray(self.mints.mo_spin_eri(self.C, self.C))
        #self.TEI = self.TEI_MO()
        self.TEI = np.asarray(self.mints.ao_eri())
###############Setup the Fock matrix and TEIs #####################
    def TEI_MO(self, C=None):
        if C is None: C = self.C
        Ca = np.asarray(self.C)
        TEI_AO = self.TEI
        nmo = self.nmo
#        for i in range (nmo):
#            for j in range(nmo):
#                for k in range(nmo):
#                    for l in range(nmo):
#                        for p in range (nmo):
#                            for q in range(nmo):
#                                for r in range(nmo):
#                                    for s in range(nmo):
#                                        TEI_MO[i, j, k, l] +=  C[i, p] * C[j, q] * TEI_AO[p, q, r, s]* C[r, k] * C[s,l]
    
        TEI_MO = (np.transpose(Ca)).dot(TEI_AO).dot(Ca)
        TEI = np.zeros(shape=(2*nmo, 2*nmo, 2*nmo, 2*nmo)) #,dtype=np.complex
        for p in range (0,2*nmo,1):
            for q in range(0,2*nmo,1):
                for r in range(0,2*nmo,1):
                    for s in range(0,2*nmo,1):

                        value1 = TEI_MO[p/2, r/2, q/2, s/2] * (p %2 == r%2) * (q %2 == s%2)
    
                        value2 = TEI_MO[p/2, s/2, q/2, r/2] * (p %2 == s%2) * (q %2 == r%2)
 
                        TEI[p, q, r, s] = (value1 - value2)
        return np.asarray(self.mints.mo_spin_eri(C, C))
                        #return TEI
    
    
    




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
        
    def MO_E(self, H=None, C=None):  
        if H is None: H = self.H
        if C is None: C = self.C 
        F = self.F_MO(H,C)
        evals, evecs = np.linalg.eigh(F)
        return evals
    
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

############################################################       
#                    
#               T1 and T2-equations
#                   By R. Glenn, I used T. Daniel Crawfords equations
#    
#    
#    
############################################################
    
    #Build Fvv
    def Fae(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = F[v, v].copy()

        term2 = - 0.5*contract('me,ma->ae', F[o, v], t1)
        term3 = contract('mafe,mf->ae', TEI[o, v, v, v], t1)
        tau = t2.copy() + contract('ia,jb->ijab', t1, t1)
        term4 =-0.5*contract('mnef,mnaf->ae', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4
        return total
    
    #Build Foo
    def Fmi(self, t1, t2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI 
        term1 = F[o, o].copy()
        term2 =0.5*contract('me,ie->mi', F[o, v], t1)
        term3 = contract('mnie,ne->mi', TEI[o, o, o, v], t1)
        tau = t2.copy() + contract('ia,jb->ijab', t1, t1)
        term4 = 0.5*contract('mnef,inef->mi', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4 
        return total
    
    #Build Fov    
    def Fme(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = F[o, v].copy()
        term2 = contract('mnef,nf->me', TEI[o, o, v, v], t1)
        total = term1 + term2
        return total

##################Build T1 equation###################################
    def Test_T1_rhs(self, t1, t2, lam1, lam2, F):
############check T1 equation########################################
        v = self.vir
        o = self.occ
        TEI = self.TEI
        
        # Setup for testing the t1, t2, and F #
        dipolexyz = self.Defd_dipole()
        t1_cal = t1
        t2_cal = t2
        t1 = t1 + 0.5*1j*t1
        t2 = t2 + 0.5*1j*t2 
        Fa =  F + dipolexyz[2] + 1j*dipolexyz[2]
        
        #print("This is fia [R]")
        #self.print_2(Fa[o, v].real)
        #print("This is fia [I]")
        #self.print_2(Fa[o, v].imag)

        #print("This is fea [R]")
        #self.print_2(Fa[v, v].real)
        #print("This is fea [I]")
        #self.print_2(Fa[v, v].imag)
        
        #print("This is fmi [R]")
        #self.print_2(Fa[o, o].real)
        #print("This is fmi [I]")
        #self.print_2(Fa[o, o].imag)
        
        
        #print("This is T1 [R]")
        #self.print_2(t1.real )
        #print("This is T1 [I]")
        #self.print_2(t1.imag )
        #print("This is T2 [R]")
        #self.print_2(t2.real )
        #print("This is T2 [I]")
        #self.print_2(t2.imag )



        #check tau
        tau = t2.copy() + contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1)
        
        #check taut
        taut = t2.copy() + 0.5*contract('ia,jb->ijab', t1, t1)
        
        #check FME
        FME = self.Fme(t1, t2, Fa)    

        #check FAE
        FAE = self.Fae(t1, t2, Fa)

        #check FMI
        FMI = self.Fmi(t1, t2, Fa)

        ############check T1 rhs################
        term1 = Fa[o, v].copy()
        term2 = contract('ae,ie->ia', FAE, t1)
        term3 = -contract('mi,ma->ia', FMI, t1)
        term4 = contract('me,imae->ia', FME, t2)
        extra1 = -contract('naif,nf->ia', TEI[o, v, o, v], t1)
        extra2 = -0.5*contract('nmei,mnae->ia', TEI[o, o, v, o], t2) 
        extra3 = -0.5*contract('maef,imef->ia', TEI[o, v, v, v], t2)
        t1_rhs = term1 + term2 + term3 + term4 + extra1 + extra3 + extra2

        # Check using the definition to get the converged t1 t2 values
        t1_rhs = self.T1eq_rhs(t1, t2, Fa)
        #print("This is T1[R]")
        #self.print_2(t1_rhs.real)
        #print("This is T1[I]")
        #self.print_2(t1_rhs.imag)
        
        ########check T2 equation##########
        #check DT2
        term1 = TEI[o, o, v, v].copy()
        
        #check T2Fae_build
        term2tmp = FAE - 0.5 *contract('me,mb->be', FME, t1)
        
        #check FAE_T2_build
        term2a = contract('be,ijae->ijab', term2tmp, t2) 
        term2 = term2a.copy() - term2a.swapaxes(2, 3) #swap ab
        
        #check T2FMI_build
        term3temp = FMI + 0.5 *contract('me,je->mj', FME, t1)
        
        #check FMI_T2_build
        term3a = -contract('mj,imab->ijab', term3temp, t2) 
        term3 = term3a.copy() - term3a.swapaxes(0, 1) #swap ij
        t2_rhs = term1 + term2 + term3  
        
        #check Wmnij
        term1 = TEI[o, o, o, o].copy()
        term2a = contract('mnie,je->mnij', TEI[o, o, o, v], t1)
        term2 = term2a - term2a.swapaxes(2,3) #swap ij
        tau = 0.5*t2 + 0.5*contract('ia,jb->ijab', t1, t1) - 0.5*contract('ib,ja->ijab', t1, t1)
        term3 = contract('mnef,ijef->mnij', TEI[o, o, v, v], tau)  
        Wmnij = term1 + term2 + term3
        tau = 0.5*t2.copy() + 0.5*contract('ia,jb->ijab', t1, t1) - 0.5*contract('ib,ja->ijab', t1, t1)
        term3 = contract('mnef,ijef->mnij', TEI[o, o, v, v], tau)
        Wmnij_2 = term1 + term2 + term3
        t1t1 =  0.5*contract('ia,jb->ijab', t1, t1) - 0.5*contract('ib,ja->ijab', t1, t1)
        
        #check Wmnij*tau
        temp = 0.5*contract('mnij,mnab->ijab', Wmnij, t2)
        temp += contract('mnij,mnab->ijab', Wmnij, t1t1)
        t2_rhs = t2_rhs + temp
    
        #check P(ij)P(ab) tma tie <mb||je> [R] [R]
        term6tmp = contract('mbej,ie,ma->ijab', TEI[o, v, v, o], t1, t1)
        term6tmp = term6tmp +  contract('maei,je,mb->ijab', TEI[o, v, v, o], t1, t1)
        term6tmp = term6tmp - contract('maej,ie,mb->ijab', TEI[o, v, v, o], t1, t1)
        term6tmp = term6tmp - contract('mbei,je,ma->ijab', TEI[o, v, v, o], t1, t1)
        t2_rhs = t2_rhs - term6tmp

        #check the other extra terms
        term7tmp = contract('abej,ie->ijab', TEI[v ,v, v, o], t1) 
        term7 =  term7tmp.copy() - term7tmp.swapaxes(0, 1) #swap ij
        term8 = -contract('mbij,ma->ijab', TEI[o, v, o, o], t1) 
        term8 += -contract('amij,mb->ijab', TEI[v, o, o, o], t1) #swap ab
        t2_rhs = t2_rhs + term7 + term8

        #Check Wabef
        term1 = TEI[v, v, v, v].copy()
        tau = t2.copy() + contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1)
        term2tmp= -contract('amef,mb->abef', TEI[v, o, v, v], t1) 
        term2 = term2tmp - term2tmp.swapaxes(0,1) #swap ab
        Wabef = term1 + term2
        t2_rhs = t2_rhs + 0.5*contract('abef,ijef->ijab', Wabef, tau)
        
        #check Wmbej
        term1 = TEI[o, v, v, o].copy()
        term2 = -contract('mnej,nb->mbej', TEI[o, o, v, o], t1)
        # the t1 t1 term below doesn't match the 1670, 1680, 1640, 1650,
        t2t1 = 0.5*t2 + contract('jf,nb->jnfb', t1, t1)
        term34 = -contract('mnef,jnfb->mbej', TEI[o, o, v, v], t2t1)
        term5 = contract('mbef,jf->mbej', TEI[o, v, v, v], t1)
        Wmbej = term1 + term2 + term34 + term5

        #check Wmbej_T2
        term6tmp = contract('mbej,imae->ijab', Wmbej, t2)
        t2_rhs = t2_rhs +  term6tmp
        t2_rhs = t2_rhs - term6tmp.swapaxes(2, 3)
        t2_rhs = t2_rhs - term6tmp.swapaxes(0, 1)
        t2_rhs = t2_rhs + term6tmp.swapaxes(0, 1).swapaxes(2, 3)
        
        #print("This is T2 [R]")
        #self.print_2(t2_rhs.real )
        #print("This is T2 [I]")
        #self.print_2(t2_rhs.imag )
        ##########Check T2 using the builg in expressions##############
        #check using my built in function
        t2_rhs = self.T2eq_rhs(t1, t2, Fa)
        #print("This is T2 [R]")
        #self.print_2(t2_rhs.real )
        #print("This is T2 [I]")
        #self.print_2(t2_rhs.imag )

#################check lam1 eq #########################################
        #setup lam1 and lam2 to check lam1 and lam2 equations
        E_test = 2.4
        lam1_cal = lam1
        lam2_cal = lam2
        lam1 = lam1.real + 1j*t1.imag*E_test
        lam2 = lam2.real + 1j*t2.imag*E_test

        #print("This is L1 [R]")
        #self.print_2(lam1.real )
        #print("This is L1 [I]")
        #self.print_2(lam1.imag )
        #print("This is L2 [R]")
        #self.print_2(lam2.real )
        #print("This is L2 [I]")
        #self.print_2(lam2.imag )

        # check Fia
        #Fia = Fa[o, v].copy()
        #Fia = Fia + contract('mnef,nf->me', TEI[o, o, v, v], t1)
        #lam1_rhs = Fia.copy()

        FME = self.Fme(t1, t2, Fa)
        lam1_rhs = FME.copy()
        


        #check Lam LFea
        #term1 = Fa[v, v].copy()
        #term3 = -contract('ma,me->ea', FME, t1)
        #term2 = contract('emaf,mf->ea', TEI[v, o, v, v], t1)
        #tau = t2 + 0.5*contract('ia,jb->ijab', t1, t1) - 0.5*contract('ib,ja->ijab', t1, t1)
        #term4 =-0.5*contract('mnaf,mnef->ea', TEI[o, o, v, v], tau)
        #Fea = term1 + term2 + term3 + term4
        Fea = self.LRFea(t1, t2, Fa)
        lam1_rhs = lam1_rhs + contract('ea,ie->ia', Fea, lam1)
        

        
        
        #check Lam LFim
        #term1 = Fa[o, o].copy()
        #term2 = contract('ie,me->im', FME, t1)
        #term3 = contract('inmf,nf->im', TEI[o, o, o, v], t1)
        ##tau = 0.5*t2 + contract('ia,jb->ijab', t1, t1)
        #term4 = 0.5*contract('inef,mnef->im', TEI[o, o, v, v], tau)
        #Fim = term1 + term2 + term3 + term4
        
        Fim = self.LRFim(t1, t2, Fa)
        lam1_rhs = lam1_rhs - contract('im,ma->ia', Fim, lam1)
    
        Gmn = self.Gmn(t2, lam2)
        Gef = self.Gfe(t2, lam2)

        Weifa = self.LWfiea(t1)
        Wmina = self.LWmina(t1)

        lam1_rhs = lam1_rhs - contract('nm,mina->ia', Gmn, Wmina)
        lam1_rhs = lam1_rhs - contract('fe,fiea->ia', Gef, Weifa)

        #match Wmbej
        #** Wmbej = <mb||ej> + t_j^f <mb||ef> - t_n^b <mn||ej>
        #**         - { t_jn^fb + t_j^f t_n^b } <mn||ef>
        Wmbej = TEI[o, v, v, o].copy()
        Wmbej = Wmbej + contract('mbef,jf->mbej', TEI[o, v, v, v], t1)
        Wmbej = Wmbej - contract('mnej,nb->mbej', TEI[o, o, v, o], t1)
        tau = t2 + contract('ia,jb->ijab', t1, t1)
        Wmbej = Wmbej - contract('mnef,jnfb->mbej', TEI[o, o, v, v], tau)
        
        
        term = contract('ieam,me->ia', Wmbej, lam1)
        lam1_rhs = lam1_rhs + term


        #Check Wabei or Wefam
        # Term I
        term1 = -TEI[v, v, v, o].copy()
        
        # Term II
        #- Fme t(mi,ab) -1/2 F_ME[R] t_Mi^Ab[R]
        Fme = self.Fme(t1, t2, Fa)
        term2 = -contract('na,mnef->efam', Fme, t2)

        # Term IIIa
        #+ t(i,f) <ab||ef>
        term3a = -contract('abef,if->abei', TEI[v, v, v, v], t1)

        #Term IIIc + IIId + IV
        #+ 1/2 t(mn,ab) <mn||ef> t(i,f)              + 1/2 P(ab) t(m,a) t(n,b) <mn||ef> t(i,f)
        #+ 1/2 t(mn,ab) <mn||ei>                     + 1/2 P(ab) t(m,a) t(n,b) <mn||ei>
        tau = t2 + contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1)
        term4a = 0.5*contract('mnab,mnie->abei', tau, Wmina)
        
 
        #Term IIIb + V
        #- P(ab) t(i,f) t(m,b) <am||ef> + P(ab) t(mi,fb) <am||ef>   IIIB + V
        tau = t2 + contract('ia,jb->ijab', t1, t1) #-  contract('ib,ja->ijab', t1, t1)

        term5a = contract('imfb,amef->abei', tau, TEI[v, o, v, v])
        tau = t2 + contract('ia,jb->ijab', t1, t1) #- contract('ib,ja->ijab', t1, t1)
        term5b = term5a - contract('imfb,amef->baei', tau, TEI[v, o, v, v])

        #Term VI and VII
        #- P(ab) t(m,a) <mb||ei> - P(ab) t(m,a) t(ni,fb) <mn||ef>
        Zeimb = contract('mnef,nifb->eimb', TEI[o, o, v, v], t2)
        Zeimb = Zeimb + TEI[v, o, o, v].copy()
        term6a = contract('eimb,ma->abei', Zeimb, t1)
        Zeiam = contract('mnef,niaf->eiam', TEI[o, o, v, v], t2)
        Zeiam =  Zeiam + TEI[v, o, v, o].copy()
        term6b = contract('eiam,mb->abei', Zeiam, t1)
        Wabei = -(term1 + term2 + term3a  + term4a + term5b  + term6a + term6b)

        #Webei = self.LRWefam(t1, t2, Fa)
        term = 0.5*contract('efam,imef->ia', Wabei, lam2)
        #term = term - 0.5*contract('efam,imef->ia', Wabei.real, lam2.real)
        lam1_rhs = lam1_rhs + term


        #Match Wmnij
        Wmnij = TEI[o, o, o, o]
        Wmnij = Wmnij + contract('mnie,je->mnij', TEI[o, o, o, v], t1)
        Wmnij = Wmnij - contract('mnje,ie->mnij', TEI[o, o, o, v], t1)
        tau = t2.copy() + contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1)
        Wmnij = Wmnij - 0.5*contract('ijef,mnfe->ijmn', TEI[o, o, v, v], tau)
        



        #Match Wmbij
        #Wmbij = <mb||ij> - Fme t_ij^be - t_n^b Wmnij + 1/2 <mb||ef> tau_ij^ef
        #+ P(ij) <mn||ie> t_jn^be + P(ij) t_i^e { <mb||ej> - t_nj^bf <mn||ef> }
        #Z(ME,jb)[R] = { <Mb|Ej> + t_jN^bF [2 <Mn|Ef> - <Mn|Fe>] - t_jN^Fb <Mn|Ef> }
        #/* W(Mb,Ij) <-- Z(ME,jb)[R] t_I^E[R] */

        Wmbij = TEI[o, v, o, o].copy()
        Wmbij = Wmbij - contract('me,ijbe->mbij', Fme, t2)
        Wmbij = Wmbij - contract('nb,mnij->mbij', t1, Wmnij)
        Wmbij = Wmbij + 0.5*contract('mbef,ijef->mbij', TEI[o, v, v, v], tau)
        Zmejb = TEI[o, v, v, o].copy()
        Zmejb = Zmejb - contract('mnef,njbf->mbej', TEI[o, o, v, v], t2)
        Zmejb_T1 = contract('mbei,je->mbij', Zmejb, t1)
        Zmejb_T1 = Zmejb_T1  - contract('mbej,ie->mbij', Zmejb, t1)
        Wmbij = Wmbij - Zmejb_T1
        
        # <mn||ie> tjneb
        #Zmijb = contract('inem,jneb->mbij', TEI[o, o, v, o], t2)
        #Zmijb = Zmijb - contract('inem,jneb->mbji', TEI[o, o, v, o], t2)
        
        Zmijb = contract('mnie,jneb->mbij', TEI[o, o, o, v], t2)
        Zmijb = Zmijb - contract('mnie,jneb->mbji', TEI[o, o, o, v], t2)
        Wmbij = Wmbij - Zmijb
        
        term = - 0.5*contract('iemn,mnae->ia', Wmbij, lam2)
        lam1_rhs = lam1_rhs + term
##################check L1 equations I used####################################

        lam1_rhs = self.lam_1eq_rhs(t1, t2, lam1, lam2, Fa)
        #print("L1 [R]")
        #self.print_2(lam1_rhs.real)
        
        #print("L1 [I]")
        #self.print_2(lam1_rhs.imag)
        

##################check L2 equations####################################

        # RHS = <ij||ab>
        lam2_rhs = TEI[o, o, v, v].copy()
        
        # RHS += Lmnab Wijmn
        lam2_rhs = lam2_rhs + 0.5*contract('mnab,ijmn->ijab',lam2, Wmnij)

        #Check Wefab
        #Wefab = <ef||ab> - P(ef) t_m^f <em||ab> + 1/2 tau_mn^ef <mn||ab>
        #tau_mn^ef = t_mn^ef + t_m^e t_n^f - t_m^f t_n^e.
        #first term  <ef||ab> * Lijef
        Wefab = TEI[v, v, v, v].copy()
        lam2_rhs =  lam2_rhs + 0.5*contract('efab,ijef->ijab', Wefab, lam2)

        #second term P(ef) t_m^f <em||ab> *Lijef
        Zmfij = contract('mf,ijef->ijem',t1, lam2)
        lam2_rhs = lam2_rhs- 0.5*contract('emab,ijem->ijab', TEI[v, o, v, v], Zmfij)
        Zmeij = contract('me,ijef->ijfm',t1, lam2)
        lam2_rhs = lam2_rhs + 0.5*contract('fmab,ijfm->ijab', TEI[v, o, v, v], Zmeij)
        
        #third term 1/2 tau_mn^ef <mn||ab> Lijef
        #not what they programmed
        tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) #- 0.25*contract('ia,jb->ijab', t1, t1)
        Zijmn = contract('mnef,ijef->ijmn', tau, lam2)
        lam2_rhs = lam2_rhs + contract('mnab,ijmn->ijab', TEI[o, o, v, v], Zijmn)

        #Check Wejab
        #RHS += P(ij) Lie[R] * Wejab[R] Weifa = self.LWfiea(t1)
        lam2_rhs = lam2_rhs + contract('ejab,ie->ijab', Weifa, lam1)
        lam2_rhs = lam2_rhs - contract('ejab,ie->jiab', Weifa, lam1)

        #Check Wijmb
        #RHS += -P(ab) Lma * Wijmb
        lam2_rhs = lam2_rhs - contract('ijmb,ma->ijab', Wmina, lam1)
        lam2_rhs = lam2_rhs + contract('ijmb,ma->ijba', Wmina, lam1)

        #Check Gae
        #RHS += P(ab) <ij||ae> Gbe
        lam2_rhs = lam2_rhs + contract('ijae,be->ijab', TEI[o, o, v, v], Gef)
        lam2_rhs = lam2_rhs - contract('ijae,be->ijba', TEI[o, o, v, v], Gef)
        
        #Check Gmj
        #RHS -= P(ij) * <im||ab> * Gmj
        lam2_rhs = lam2_rhs - contract('mjab,im->ijab', TEI[o, o, v, v], Gmn)
        lam2_rhs = lam2_rhs + contract('mjab,im->jiab', TEI[o, o, v, v], Gmn)
        
        #Check Fae
        #RHS += P(ab) Lijae * Feb
        Fea = self.LRFea(t1, t2, Fa)
        lam2_rhs = lam2_rhs + contract('ijab,eb->ijab', lam2, Fea)
        lam2_rhs = lam2_rhs - contract('ijab,eb->ijba', lam2, Fea)

        #Check Fmi
        #RHS -= P(ij)*Limab*Fjm
        Fjm = self.LRFim(t1, t2, Fa)
        lam2_rhs = lam2_rhs - contract('imab,jm->ijab', lam2, Fjm)
        lam2_rhs = lam2_rhs + contract('imab,jm->jiab', lam2, Fjm)
        
        #Check Wjebm
        #RHS += P(ij)P(ab)Limae * Wjebm
        lam2_rhs = lam2_rhs + contract('imae,jebm->ijab', lam2, Wmbej)
        lam2_rhs = lam2_rhs - contract('imae,jebm->ijba', lam2, Wmbej)
        lam2_rhs = lam2_rhs - contract('imae,jebm->jiab', lam2, Wmbej)
        lam2_rhs = lam2_rhs + contract('imae,jebm->jiba', lam2, Wmbej)

        #Check L_ij^ab <-- P(ij) P(ab) L_i^a Fjb
        #where Fjb = fjb + t_n^f <jn||bf>
        #Fjb = Fa[o, v].copy() + contract('nf,jnbf->jb', t1, TEI[o, o, v, v ])
        Fme = self.Fme(t1, t2, Fa)
        lam2_rhs = lam2_rhs + contract('ia,jb->ijab', lam1, Fme)
        lam2_rhs = lam2_rhs - contract('ia,jb->ijba', lam1, Fme)
        lam2_rhs = lam2_rhs - contract('ia,jb->jiab', lam1, Fme)
        lam2_rhs = lam2_rhs + contract('ia,jb->jiba', lam1, Fme)
        
        #print("This is L2 rhs")
        #self.print_2(lam2_rhs.real)

        ########Match using my equations#########
        lam2_rhs = self.lam2eq_rhs(t1, t2, lam1, lam2, Fa)
        #print("L2 [R]")
        #self.print_2(lam2_rhs.real)
        
        #print("L2 [I]")
        #self.print_2(lam2_rhs.imag)
    
    ##########################CHECK THE DENSITY MATRIX#######################
        #check  DIJ
        term1 = contract('je,ie->ij', lam1, t1)
        term2 = 0.5*contract('mjea,miea->ij', lam2, t2)
        DIJ = -(term1 + term2)
        
        term1 = contract('na,nb->ab', t1, lam1)
        term2 = 0.5*contract('mnea,mneb->ab', t2, lam2)
        DAB = (term1 + term2)
        
        term1 = t1
        #DIA += timae[R] Lme[R]
        term2 = contract('miea,me->ia', t2, lam1)
        #DIA += -tie tma Lme
        Zim = contract('ie,me->im', t1, lam1)
        term2 = term2 - contract('im,ma->ia', Zim, t1)
        #DIA += -1/2 L^mnef tinef tma
        Zim = contract('mnef,inef->mi', lam2, t2)
        term3 = -0.5*contract('mi,ma->ia', Zim, t1)
        #ZAE[R] = tmnaf Lmnef
        Zae = contract('mnaf,mnef->ae', t2, lam2)
        term3 = term3 -0.5*contract('ae,ie->ia', Zae, t1)
        DIA = term1  + term2 + term3

        DAI = lam1

        #Build the four blocks of the density matrix
        DIA = self.Dai(t1, t2, lam1, lam2)
        DAI = lam1
        DAB = self.Dab(t1, t2, lam1, lam2)
        DIJ = self.Dij(t1, t2, lam1, lam2)
        
        #print("DIJ [R]")
        #self.print_2(DIJ.real)
        #print("DIJ [I]")
        #self.print_2(DIJ.imag)
        
        #print("DAB [R]")
        #self.print_2(DAB.real)
        #print("DAB [I]")
        #self.print_2(DAB.imag)
        
        #print("DIA [R]")
        #self.print_2(DIA.real)
        #print("DIA [I]")
        #self.print_2(DIA.imag)
     
        #print("DAI [R]")
        #self.print_2(DAI.real)
        #print("DAI [I]")
        #self.print_2(DAI.imag)

        
        #Build the correlated density matrix
        #left_p = np.vstack((pij, pai))
        #right_p = np.vstack((pia, pab))
        #corr_p = np.hstack((left_p, right_p))
        
        #Build the Hartree Fock Density matrix
        #HF_p = self.Buildpho(F)
        #HF_p = self.pholowdinbasis(HF_p)
        
        #Calculate the corr dipole moment
        #dip_xyz_corr = []
        #for i in range(3):
        #    temp = contract('ij,ij->', dipolexyz[i], HF_p + corr_p)
            #temp = contract('ij,ij->ij', dipolexyz[i], HF_p + corr_p)
            #temp = contract('ii', temp)
        #    dip_xyz_corr.append(temp)
        


#dipolexyz = self.Defd_dipole()
        
        
        
        
        
        
        
        


        #print("Z [R]")
        #self.print_2(Zim.real)
        #print("Z [I]")
        #self.print_2(Zim.imag)
    
    #####################CHECK RUNGE KUTTA T1 ########################################
        t1 = t1_cal
        t2 = t2_cal
        E0= 0.8
        t=0.5
        w0=2.05
        dt = 0.02
        def Vt(t):
            mu = self.Defd_dipole()
            return -E0*cmath.exp(1j*w0*2.0*np.pi*t)*mu[2]
        #print("This is Vt", Vt(t))
        
        # Setup for testing the t1, t2, and F #
        dipolexyz = self.Defd_dipole()
        t1 = t1 + 0.5*1j*t1
        t2 = t2 + 0.5*1j*t2
        k1 = self.T1eq_rhs(t1, t2, F + Vt(t))
        k2 = self.T1eq_rhs(t1 + dt/2.0*k1, t2, F + Vt(t + dt/2.0))
        k3 = self.T1eq_rhs(t1 + dt/2.0*k2, t2, F + Vt(t + dt/2.0))
        k4 = self.T1eq_rhs(t1 + dt*k3, t2, F + Vt(t + dt))
        newt1 = dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
        #print ("RK T1[R]")
        #self.print_2(t1.real - newt1.imag)
        #print ("RK T1[I]")
        #self.print_2(t1.imag + newt1.real)
    
    ############################CHECK RUNGE KUTTA T2 ######################################
        k1 = self.T2eq_rhs(t1, t2, F + Vt(t))
        k2 = self.T2eq_rhs(t1, t2 + dt/2.0*k1, F + Vt(t + dt/2.0))
        k3 = self.T2eq_rhs(t1, t2 + dt/2.0*k2, F + Vt(t + dt/2.0))
        k4 = self.T2eq_rhs(t1, t2 + dt*k3,  F + Vt(t + dt))
        newt2 = dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
        #print ("RK T2[R]")
        #self.print_2(t2.real - newt2.imag)
        #print ("RK T2[I]")
        #self.print_2(t2.imag + newt2.real)
        #print ("RK T1[R]")
        #self.print_2(t2.real)
        #print ("RK T1[I]")
        #self.print_2(t2.imag)
    
        E_test = 2.4
        #lam1 = lam1_cal
        #lam2 = lam2_cal
        #lam1 = lam1.real + 1j*t1.imag*E_test
        #lam2 = lam2.real + 1j*t2.imag*E_test
        #print "Vt", Vt(t).real, Vt(t).imag
        k1 = self.lam_1eq_rhs(t1, t2, lam1, lam2, F + Vt(t))
        k2 = self.lam_1eq_rhs(t1, t2, lam1 + dt/2.0*k1, lam2, F + Vt(t + dt/2.0))
        k3 = self.lam_1eq_rhs(t1, t2, lam1 + dt/2.0*k2, lam2, F + Vt(t + dt/2.0))
        k4 = self.lam_1eq_rhs(t1, t2, lam1 + dt*k3, lam2, F + Vt(t + dt))
        newL1 = dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
    
        #print ("RK k1 L1[R]")
        #self.print_2(k1.real)
        #print ("RK k1 L1[I]")
        #self.print_2(k1.imag)
        #print ("RK k2 L1[R]")
        #self.print_2(k2.real)
        #print ("RK k2 L1[I]")
        #self.print_2(k2.imag)
        #print ("RK k3 L1[R]")
        #self.print_2(k3.real)
        #print ("RK k3 L1[I]")
        #self.print_2(k3.imag)
        #print ("RK k4 L1[R]")
        #self.print_2(k4.real)
        #print ("RK k4 L1[I]")
        #self.print_2(k4.imag)
        #print ("RK delta L1[R]")
        #self.print_2(lam1.real - newL1.imag)
        #print ("RK delta L1[I]")
        #self.print_2(lam1.imag + newL1.real)

        
        #self.check_T1_T2_L1_L2(t1, t2, lam1, lam2, F)
        Fb = F + Vt(t)
        #print (" F[R]")
        #self.print_2(Fb.real)
        #print (" F[I]")
        #self.print_2(Fb.imag)
        
        k1 = self.lam2eq_rhs(t1, t2, lam1, lam2, F + Vt(t))
        k2 = self.lam2eq_rhs(t1, t2, lam1, lam2 + dt/2.0*k1, F + Vt(t + dt/2.0))
        k3 = self.lam2eq_rhs(t1, t2, lam1, lam2 + dt/2.0*k2, F + Vt(t + dt/2.0))
        k4 = self.lam2eq_rhs(t1, t2, lam1, lam2 + dt*k3, F + Vt(t + dt))
        new_L2 = dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
    
        #print ("RK k1 L2[R]")
        #self.print_2(k1.real)
        #print ("RK k1 L2[I]")
        #self.print_2(k1.imag)
        
        #print ("RK k2 L2[R]")
        #self.print_2(k2.real)
        #print ("RK k2 L2[I]")
        #self.print_2(k2.imag)
    
        #print ("RK k3 L2[R]")
        #self.print_2(k3.real)
        #print ("RK k3 L2[I]")
        #self.print_2(k3.imag)
        
        #print ("RK k4 L2[R]")
        #self.print_2(k4.real)
        #print ("RK k4 L2[I]")
        #self.print_2(k4.imag)
    
        #print ("RK delta L2[R]")
        #self.print_2(new_L2.real)
        #print ("RK delta L2[I]")
        #self.print_2(new_L2.imag)
    
        #print ("RK L2[R]")
        #self.print_2(lam2.real - new_L2.imag)
        #print ("RK L2[I]")
        #self.print_2(lam2.imag + new_L2.real)

    def check_T1_T2_L1_L2(self, t1, t2, lam1, lam2, F):
        print (" T1[R]")
        self.print_2(t1.real)
        print (" T1[I]")
        self.print_2(t1.imag)
        print (" T2[R]")
        self.print_2(t2.real)
        print (" T2[I]")
        self.print_2(t2.imag)
        print (" L1[R]")
        self.print_2(lam1.real)
        print (" L1[I]")
        self.print_2(lam1.imag)
        print (" L2[R]")
        self.print_2(lam2.real)
        print (" L2[I]")
        self.print_2(lam2.imag)
    
        print (" F[R]")
        self.print_2(F.real)
        print (" F[I]")
        self.print_2(F.imag)
    
####################################################################
#
#
#
#####################################################
    def T1eq_rhs(self, t1, t2, F):        
        #All terms in the T1 Equation
        v = self.vir
        o = self.occ
        TEI = self.TEI
        fae = self.Fae(t1, t2, F) 
        fmi = self.Fmi(t1, t2, F)  
        fme = self.Fme(t1, t2, F) 
              
        term1 = F[o, v].copy()
        term2 = contract('ae,ie->ia', fae, t1)
        term3 = -contract('mi,ma->ia', fmi,t1)
        term4 = contract('me,imae->ia', fme, t2)
        #extra terms   
        extra1 = -contract('naif,nf->ia', TEI[o, v, o, v], t1)
        extra2 = -0.5*contract('nmei,mnae->ia', TEI[o, o, v, o], t2)
        extra3 = -0.5*contract('maef,imef->ia', TEI[o, v, v, v], t2)
        
        total = term1 + term2 + term3  + term4  + extra1  + extra2  + extra3
        return total
     
   #Build Woooo for t2 terms 
    def Wmnij(self, t1 ,t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, o, o, o].copy()
        term2a = contract('mnie,je->mnij', TEI[o, o, o, v], t1)
        term2 = term2a - term2a.swapaxes(2,3) #swap ij
        #tau = contract('ia,jb->ijab', t1, t1) 
        #term3 = 0.25*contract('mnef,ijef->mnij', TEI[o, o, v, v], t2)
        #term4a = 0.5*contract('mnef,ijef->mnij', TEI[o, o, v, v], tau)  
        #term4 = term4a - term4a.swapaxes(2,3)
        
        tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) - 0.5*contract('ib,ja->ijab', t1, t1) 
        term3 = contract('mnef,ijef->mnij', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 
        return total    
     
    #Build Woooo for t1 * t1 like terms       
    def Wmnij_2(self, t1 ,t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, o, o, o].copy()
        term2a = contract('mnie,je->mnij', TEI[o, o, o, v], t1)
        term2 = term2a - term2a.swapaxes(2,3) #swap ij
        tau = contract('ia,jb->ijab', t1, t1) 
        #term3 = 0.25*contract('mnef,ijef->mnij', TEI[o, o, v, v], t2)
        term4a = 0.25*contract('mnef,ijef->mnij', TEI[o, o, v, v], tau)  
        term4 = term4a - term4a.swapaxes(2,3)
        #tau = contract('ia,jb->ijab', t1, t1) + contract('ib,ja->ijab', t1, t1) 
        #term4 = 0.25*contract('mnef,ijef->mnij', TEI[o, o, v, v], tau)  
        total = term1 + term2 + term4
        return total 
     
    #Build Wvvvv for t2 terms                                                                                                                                                                                                                                                                     
    def Wabef(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[v, v, v, v].copy()
        term2tmp = -contract('amef,mb->abef', TEI[v, o, v, v], t1) 
        term2 = term2tmp - term2tmp.swapaxes(0,1) #swap ab
        tau = contract('ia,jb->ijab', t1, t1) #- contract('ib,ja->ijab', t1, t1) 
        term3 = 0.25*contract('mnef,mnab->abef', TEI[o, o, v, v], t2)
        term4a = 0.5*contract('mnef,mnab->abef', TEI[o, o, v, v], tau)
        term4 = term4a - term4a.swapaxes(0,1)
        #tau = t2 + contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1) 
        #term3 = 0.5*contract('mnef,mnab->abef', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4
        return total

    #Build Wvvvv for t1 * t1 like terms
    def Wabef_2(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[v, v, v, v].copy()
        term2tmp = -contract('amef,mb->abef', TEI[v, o, v, v], t1) 
        term2 = term2tmp - term2tmp.swapaxes(0,1) #swap ab
        #tau = contract('ia,jb->ijab', t1, t1) #- contract('ib,ja->ijab', t1, t1) 
        ##term3 = 0.25*contract('mnef,mnab->abef', TEI[o, o, v, v], t2)
        #term4a = 0.25*contract('mnef,mnab->abef', TEI[o, o, v, v], tau)
        #term4 = term4a - term4a.swapaxes(0,1)
        tau = contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1) 
        term4 = 0.25*contract('mnef,mnab->abef', TEI[o, o, v, v], tau)
        total = term1 + term2 + term4
        return total
    
    #Build Wovvo                                                                                                                                                                                                                                                                    
    def Wmbej(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, v, v, o].copy()
        term2 = -contract('mnej,nb->mbej', TEI[o, o, v, o], t1)
        t2t1 = 0.5*t2 + contract('jf,nb->jnfb', t1, t1)
        term34 = -contract('mnef,jnfb->mbej', TEI[o, o, v, v], t2t1)
        term5 = contract('mbef,jf->mbej', TEI[o, v, v, v], t1)
        total = term1 + term2 + term34 + term5 
        return total
 
########### Build T2 Equation################################                                                       
    def T2eq_rhs(self, t1, t2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI         
        fae = self.Fae(t1, t2, F) 
        fmi = self.Fmi(t1, t2, F)  
        fme = self.Fme(t1, t2, F) 
        wmnij = self.Wmnij(t1, t2, F)
        wabef = self.Wabef(t1, t2, F)
        wmbej = self.Wmbej(t1, t2, F)
        wabef_2 = self.Wabef_2(t1 ,t2, F)
        wmnij_2 = self.Wmnij_2(t1 ,t2, F)
        #All terms in the T2 Equation
        term1 = TEI[o, o, v, v].copy()
        
        term2tmp = fae - 0.5 *contract('me,mb->be', fme, t1)
        term2a = contract('be,ijae->ijab', term2tmp, t2) 
        term2 = term2a - term2a.swapaxes(2, 3) #swap ab
        
        term3temp = fmi + 0.5 *contract('me,je->mj', fme, t1)
        term3a = -contract('mj,imab->ijab', term3temp, t2) 
        term3 = term3a - term3a.swapaxes(0, 1) #swap ij
             
        tau = contract('ma,nb->mnab', t1, t1) - contract('na,mb->mnab', t1, t1)
        term44 = 0.5*contract('mnij,mnab->ijab', wmnij, t2)
        term55 = 0.5*contract('abef,ijef->ijab', wabef, t2)   
        term44 += 0.5*contract('mnij,mnab->ijab', wmnij_2, tau) 
        term55 +=  0.5*contract('abef,ijef->ijab', wabef_2, tau)  
 
        term6tmp = contract('mbej,imae->ijab', wmbej, t2)
        term6tmp = term6tmp - contract('mbej,ie,ma->ijab', TEI[o, v, v, o], t1, t1)
        term6 =  term6tmp - term6tmp.swapaxes(2, 3)  - term6tmp.swapaxes(0, 1)  + term6tmp.swapaxes(0, 1).swapaxes(2, 3)

                                             
        term7tmp = contract('abej,ie->ijab', TEI[v ,v, v, o], t1) 
        term7 =  term7tmp - term7tmp.swapaxes(0, 1) #swap ij 
                             
        term8tmp = -contract('mbij,ma->ijab', TEI[o, v, o, o], t1) 
        term8 =  term8tmp - term8tmp.swapaxes(2, 3) #swap ab
    
        total = term1 + term2 + term3 + term44 + term55 + term6 + term7 + term8
        return total
    
    #Calculate the CCSD energy 
    def CCSD_Corr_E(self, t1, t2, F):
        o = self.occ
        v = self.vir
        TEI = self.TEI
        term1 = contract('ia,ia->',F[o, v], t1)
        term2 = 0.25*contract('ijab,ijab->', TEI[o, o, v, v], t2)
        term3 = 0.5*contract('ijab,ia,jb->', TEI[o, o, v, v], t1, t1)
        total = term1 + term2
        total = total + term3 
        return total                           
    
    # update the T2 iteratively
    def corrected_T2(self, t2, dt2, F):
        o = self.occ
        v = self.vir
        eps, evecs = np.linalg.eigh(F)
        Dem = eps[o].reshape(-1, 1, 1, 1)
        Dem = Dem + eps[o].reshape(-1, 1, 1)
        Dem = Dem - eps[v].reshape(-1, 1) 
        Dem = Dem - eps[v]
        Dem = 1/Dem
        t2 = t2 + contract('ijab,ijab->ijab', dt2, Dem)
        return t2
     
     # update the T1 iteratively    
    def corrected_T1(self, t1, dt1, F):
        o = self.occ
        v = self.vir
        eps, evecs = np.linalg.eigh(F)
        Dem =  eps[o].reshape(-1, 1) - eps[v]
        Dem = 1/Dem
        t1 = t1 + contract('ia,ia->ia', dt1, Dem)
        return t1
    
    #Routine for DIIS solver, builds all arrays(maxsize) before B is computed    
    def DIIS_solver(self, t1, t2, F, maxsize, maxiter, E_min):
            #Store the maxsize number of t1 and t2
            T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
            T2rhs = self.T2eq_rhs(t1, t2, np.longdouble(F))
            t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
            t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
            t1stored = [t1.copy()]
            t2stored = [t2.copy()]
            errort1 = []
            errort2 = []
            
            for n in range(1, maxsize+1):  
                T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
                T2rhs = self.T2eq_rhs(t1, t2, np.longdouble(F))
                t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
                t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
                t1stored.append(t1.copy())
                t2stored.append(t2.copy())
                
                errort1.append(t1stored[n]- t1stored[n-1])
                errort2.append(t2stored[n]- t2stored[n-1])

             # Build B
            B = np.ones((maxsize + 1, maxsize + 1)) * -1
            B[-1, -1] = 0
            for z in range(1, maxiter):
                CCSD_E_old = self.CCSD_Corr_E( t1, t2, F)
                for n in range(maxsize):
                    for m in range(maxsize):
                        a = contract('ia,ia->',errort1[m], errort1[n])
                        b = contract('ijab,ijab->', errort2[m], errort2[n])
                        B[n, m] = a + b
    
                # Build residual vector
                A = np.zeros(maxsize + 1)
                A[-1] = -1

                c = np.linalg.solve(B, A)
                
                # Update t1 and t2 
                t1 = 0.0*t1
                t2 = 0.0*t2
                for n in range(maxsize):
                    t1 += c[n] * t1stored[n+1]
                    t2 += c[n] * t2stored[n+1]

                oldt1 = t1.copy()
                oldt2 = t2.copy()
                #test if converged
                CCSD_E = self.CCSD_Corr_E( t1, t2, F)
                diff_E = CCSD_E - CCSD_E_old
                if (abs(diff_E) < E_min):
                    break
                #update t1 and t2 list
                T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
                T2rhs = self.T2eq_rhs(t1, t2, np.longdouble(F))
                t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
                t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
                t1stored.append(t1.copy())
                t2stored.append(t2.copy())
                
                errort1.append(t1 - oldt1)
                errort2.append(t2 - oldt2)
                
                print("inter =", z,  "\t", "CCSD_E =", CCSD_E,"diff=", diff_E)
                del t1stored[0]
                del t2stored[0]
                del errort1[0]
                del errort2[0]
            return CCSD_E, t1, t2
    
    #a regular iterative solver, Slow, don't use        
    def NO_DIIS_solver(self, t1, t2, F, maxsize, maxiter, E_min):    
        i=0
        for x in range (maxiter):
            CCSDE_Em = self.CCSD_Corr_E(t1, t2, F)
            T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
            T2rhs = self.T2eq_rhs(t1, t2, np.longdouble(F))
            t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
            t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
            CCSD_E = self.CCSD_Corr_E(t1, t2, F)
            diff_E = np.abs( CCSD_E -CCSDE_Em )
            i+=1
            if (abs(diff_E) < E_min):
                break
            print("inter =", i,  "\t", "CCSD_E =", CCSD_E,"diff=", diff_E)
        return CCSD_E, t1, t2
 
##############################################################################
#    
#     
#        
#                       Lambda Equations:
#                       Derived by R. Glenn
#     
#   
#      
#######################################################################     

    # Build Fvv for L1 and L2 
    def LRFea(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = F[v, v].copy()
        term2 = - contract('ma,me->ea', F[o, v], t1)
        term3 = contract('emaf,mf->ea', TEI[v, o, v, v], t1)
        tau = 0.5*t2 + contract('ia,jb->ijab', t1, t1)
        term4 =-contract('mnaf,mnef->ea', TEI[o, o, v, v], tau)
        #Fea = term1 + term2 + term3 + term4
        
        total = term1 + term2 + term3 + term4
        return total
        
    #Build Foo for L1 and L2     
    def LRFim(self, t1, t2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI 
        term1 = F[o, o].copy()
        term2 = contract('ie,me->im', F[o, v], t1)
        term3 = contract('inmf,nf->im', TEI[o, o, o, v], t1)
        tau = 0.5*t2 + contract('ia,jb->ijab', t1, t1) 
        term4 = contract('inef,mnef->im', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4 
        return total
        
    #Build Wovvo          
    def LSWieam(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, v, v, o].copy()
        term2 = contract('eifa,mf->ieam', TEI[v, o, v, v], t1)
        term3 = -contract('nima,ne->ieam', TEI[o, o, o, v], t1)
        tau = t2 + contract('ia,jb->ijab', t1, t1) 
        #term4 =  contract('ijab,mjeb->ieam', TEI[o, o, v, v], tau)
        #should be the same but below gives several sig figs more accurate?
        term4 =  -contract('ijab,mjbe->ieam', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4 
        ###########Stanton ############
        #Wmbej = self.Wmbej(t1, t2, F)
        #term1 = -0.5*contract('mnef,jnfb->mbej', TEI[o, o, v, v], t2)
        #totals = Wmbej + term1
        return total
            
    #Build Wvvvo    
    def LRWefam(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        #Build Fme-Later change to use self.Fme
        #Fme = F[o, v].copy() + contract('njab,jb->na', TEI[o, o, v, v], t1)
        Fme = self.Fme(t1, t2, F)
        #Build Wooo
        
        #term1 = 0.5*TEI[v, v, v, v].copy()
        #term2 = -contract('jfab,je->efab', TEI[o, v, v, v], t1)
        ##term2 = term2a + term2a.swapaxes(2,3) 
        #tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) # - contract('ib,ja->ijab', t1, t1)
        #term3 =contract('jnab,jnef->efab', TEI[o, o, v, v], tau)
        #Wabef =  term1 + term2 + term3  
        Wabef = self.LSWabef(t1, t2, F)
        

        term1 = 0.5*TEI[v, v, v, o].copy()
        term2 = 0.5*contract('na,mnef->efam', Fme, t2)
        term3 = contract('efab,mb->efam', Wabef, t1)
        term4a = -TEI[o, v, v, o].copy() + contract('jnab,nmfb->jfam', TEI[o, o, v, v], t2)  
        term4 = contract('jfam,je->efam', term4a, t1)
        tau =0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) #- contract('ib,ja->ijab', t1, t1)
        term5 = contract('jnam,jnef->efam', TEI[o, o, v, o], tau)
        term6 = -contract('jfab,jmeb->efam', TEI[o, v, v, v], t2) 
        total = term1 + (term2 + term3 + term4 + term5 + term6) #+ extra
        return total
    
       #Build Wovoo                     
    def LRWibjm(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        #Build Fme -Later change to use self.Fme
        #Fme = F[o, v].copy() + contract('inef,nf->ie', TEI[o, o, v, v], t1)
        Fme = self.Fme(t1, t2, F)
        #Build Wmnij
        #term1 = 0.5*TEI[o, o, o, o].copy()
        #term2 = contract('inem,je->injm', TEI[o, o, v, o], t1) 
        #tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) #- contract('ib,ja->ijab', t1, t1)
        #term3 = contract('inef,jmef->injm', TEI[o, o, v, v], tau)
        #Wmnij = term1 + term2 + term3 
        Wmnij = self.LSWmnij(t1, t2, F)
        
        term1 = -0.5*TEI[o, v, o, o].copy()
        term2 = 0.5*contract('ie,jmbe->ibjm', Fme, t2)
        term3 = contract('injm,nb->ibjm', Wmnij, t1)
        term4a = -TEI[o, v, v, o].copy() - contract('inef,nmfb->ibem', TEI[o, o, v, v], t2) 
        term4 = contract('ibem,je->ibjm', term4a, t1)
        tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) #-contract('ib,ja->ijab', t1, t1)
        term5 = -contract('ibef,jmef->ibjm', TEI[o, v, v, v], tau)
        term6 = contract('inem,jneb->ibjm', TEI[o, o, v, o], t2)
        total = term1 + (term2 + term3 + term4 + term5 + term6) 
        return total
                                                                                                                                          
    def Gfe(self, t2, lam2):
        return -0.5*contract('mnfb,mneb->fe', lam2, t2)
        
    def Gmn(self, t2, lam2):
        return 0.5*contract('njed,mjed->nm', lam2, t2)
             
    #Build Wvovv       
    def LWfiea(self, t1):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[v, o, v, v].copy()
        term2 = -contract('jiea,jf->fiea', TEI[o, o, v, v], t1)
        total = term1 + term2
        return total
        
     #Build Wooov   
    def LWmina(self, t1):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, o, o, v].copy()
        term2 = contract('mifa,nf->mina', TEI[o, o, v, v], t1)
        total = term1 + term2
        return total
        
###############Lam1 Equation#####################
 

    def lam_1eq_rhs(self, t1, t2, lam1, lam2, F):   
        v = self.vir
        o = self.occ
        TEI = self.TEI
        Fia = self.Fme(t1, t2, F)
        Fea = self.LRFea(t1, t2, F)
        Fim = self.LRFim(t1, t2, F)
        Wieam = self.LSWieam(t1, t2, F)
        Wefam = self.LRWefam(t1, t2, F)
        Wibjm= self.LRWibjm(t1, t2, F)
                    
        Gef = self.Gfe(t2, lam2)
        Gmn = self.Gmn(t2, lam2)
        Weifa = self.LWfiea(t1)
        Wmina = self.LWmina(t1)
        
        
        term1 = Fia.copy()
        term2 = contract('ea,ie->ia', Fea, lam1)
        term3 = -contract('im,ma->ia', Fim, lam1)
        term4 = contract('ieam,me->ia', Wieam, lam1)
        term5 = contract('efam,imef->ia', Wefam, lam2)
        term6 = contract('ibjm,jmab->ia', Wibjm, lam2) 
        term7 = -contract('fe,fiea->ia', Gef, Weifa)
        term8 = -contract('nm,mina->ia', Gmn, Wmina)
        total = (term1 + (term2 + term3 + term4) + term5 + term6 + term7 + term8) 
        return total
################################################################

    
    # Build Woooo 
    def LSWmnij(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = 0.5*TEI[o, o, o, o].copy()
        term2 = contract('ijme,ne->ijmn', TEI[o, o, o, v], t1)
        tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1)
        term3 = contract('ijfe,mnfe->ijmn', TEI[o, o, v, v], tau)
        total = (term1 + term2 + term3)
        return total
             
    #Build Wvvvv          
    def LSWabef(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = 0.5*TEI[v, v, v, v].copy()
        term2 = -contract('emab,mf->efab', TEI[v, o, v, v], t1)
        tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) 
        term3 = contract('nmab,nmef->efab', TEI[o, o, v, v], tau)
        total = (term1 + term2 + term3  )
        return total
                                
########################Lam 2 Equations################
    def lam2eq_rhs(self, t1, t2, lam1, lam2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI     
        Feb = self.LRFea(t1, t2, F)
        Fjm = self.LRFim(t1, t2, F)
        Wijmn = self.LSWmnij(t1, t2, F)
        Wefab = self.LSWabef(t1, t2, F)
        Wjebm = self.LSWieam(t1, t2, F)
        Wejab = self.LWfiea(t1)
        Wijmb = self.LWmina(t1)
        Fjb = self.Fme(t1, t2, F)
        Gbe = self.Gfe(t2, lam2)
        Gmj = self.Gmn(t2, lam2) 
        
        term1 = TEI[o, o, v, v]
        term2a = contract('eb,ijae->ijab', Feb, lam2)
        term2 = term2a - term2a.swapaxes(2,3)
        term3a = -contract('jm,imab->ijab', Fjm, lam2)
        term3 = term3a - term3a.swapaxes(0,1)
        term4 = contract('ijmn,mnab->ijab', Wijmn, lam2)
        term5 = contract('efab,ijef->ijab', Wefab, lam2)
        term6a = contract('ejab,ie->ijab', Wejab, lam1)
        term6 = term6a - term6a.swapaxes(0,1)
        term7a = -contract('ijmb,ma->ijab', Wijmb, lam1)
        term7 = term7a - term7a.swapaxes(2,3)
        #term8 and 9
        term89a = contract('jebm,imae->ijab', Wjebm, lam2) + contract('jb,ia->ijab', Fjb, lam1)
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
        
    def CCSD_pseudo_E(self, t1, t2, lam1, lam2, F):
        o = self.occ
        v = self.vir
        TEI = self.TEI
        term1 = contract('ia,ia->', F[o, v], lam1)
        term2 = 0.25*contract('ijab,ijab->', TEI[o, o, v, v], lam2)
        return term1, term2                         
        
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
         
    def corrected_lam1(self, lam1, dlam1, F):
        o = self.occ
        v = self.vir
        eps, evecs = np.linalg.eigh(F)
        Dem =  eps[o].reshape(-1, 1) - eps[v]
        Dem = 1/Dem
        lam1 = lam1 + contract('ia,ia->ia', dlam1, Dem)
        return lam1

    
    def NO_DIIS_solve_lamr(self, t1, t2, lam1, lam2, F, maxsize, maxiter, E_min):    
        i=0
        print("this is the convergence", E_min)
        for x in range (maxiter):
            E1, E2 = self.CCSD_pseudo_E(t1, t2, lam1, lam2, F)
            pseudo_Em = E1 +E2
            lam1rhs = self.lam_1eq_rhs(t1, t2, lam1, lam2, np.longdouble(F))
            lam2rhs = self.lam2eq_rhs(t1, t2, lam1, lam2 ,np.longdouble(F))
            lam1 = np.longdouble(self.corrected_lam1(lam1, lam1rhs, F))
            lam2 = np.longdouble(self.corrected_lam2(lam2, lam2rhs, F))
            E1, E2 = self.CCSD_pseudo_E(t1, t2, lam1, lam2, F)
            pseudo_E = E1 +E2
            diff_E = np.abs( pseudo_E -pseudo_Em )
            i+=1
            
            if (abs(diff_E) < E_min):
                break
                #pass
            print("inter =", i,  "\t", "pseudo_E =", pseudo_E,"diff=", diff_E)
        print(E1, E2)
        return pseudo_E, lam1, lam2
        
    def DIIS_solver_Lam(self, t1, t2, lam1, lam2, F, maxsize, maxiter, E_min): 
            #Store the maxsize number of t1 and t2
            lam1rhs = self.lam_1eq_rhs(t1, t2, lam1, lam2, np.longdouble(F))
            lam2rhs = self.lam2eq_rhs(t1, t2, lam1, lam2 ,np.longdouble(F))
            lam1 = np.longdouble(self.corrected_lam1(lam1, lam1rhs, F))
            lam2 = np.longdouble(self.corrected_lam2(lam2, lam2rhs, F))
            lam1stored = [lam1.copy()]
            lam2stored = [lam2.copy()]
            errort1 = []
            errort2 = []
            
            for n in range(1, maxsize+1):  
                lam1rhs = self.lam_1eq_rhs(t1, t2, lam1, lam2, np.longdouble(F))
                lam2rhs = self.lam2eq_rhs(t1, t2, lam1, lam2 ,np.longdouble(F))
                lam1 = np.longdouble(self.corrected_lam1(lam1, lam1rhs, F))
                lam2 = np.longdouble(self.corrected_lam2(lam2, lam2rhs, F))
                lam1stored.append(lam1.copy())
                lam2stored.append(lam2.copy())
                
                errort1.append(lam1stored[n]-lam1stored[n-1])
                errort2.append(lam2stored[n]- lam2stored[n-1])

             # Build B
            B = np.ones((maxsize + 1, maxsize + 1)) * -1
            B[-1, -1] = 0
            for z in range(1, maxiter):
                E1, E2 = self.CCSD_pseudo_E(t1, t2, lam1, lam2, F)
                CCSD_E_old = E1 + E2
                for n in range(maxsize):
                    for m in range(maxsize):
                        a = contract('ia,ia->',errort1[m], errort1[n])
                        b = contract('ijab,ijab->', errort2[m], errort2[n])
                        B[n, m] = a + b
    
                # Build residual vector
                A = np.zeros(maxsize + 1)
                A[-1] = -1

                c = np.linalg.solve(B, A)
                
                # Update t1 and t2 
                lam1 = 0.0*lam1
                lam2 = 0.0*lam2
                for n in range(maxsize):
                    lam1 += c[n] * lam1stored[n+1]
                    lam2 += c[n] * lam2stored[n+1]

                oldlam1 = lam1.copy()
                oldlam2 = lam2.copy()
                #test if converged
                E1, E2 = self.CCSD_pseudo_E(t1, t2, lam1, lam2, F)
                CCSD_E = E1 + E2
                diff_E = CCSD_E - CCSD_E_old
                if (abs(diff_E) < E_min):
                    break
                #update t1 and t2 list
                lam1rhs = self.lam_1eq_rhs(t1, t2, lam1, lam2, np.longdouble(F))
                lam2rhs = self.lam2eq_rhs(t1, t2, lam1, lam2 ,np.longdouble(F))
                lam1 = np.longdouble(self.corrected_lam1(lam1, lam1rhs, F))
                lam2 = np.longdouble(self.corrected_lam2(lam2, lam2rhs, F))
                lam1stored.append(lam1.copy())
                lam2stored.append(lam2.copy())
                
                errort1.append(lam1 - oldlam1)
                errort2.append(lam2 - oldlam2)
                
                print("inter =", z,  "\t", "Pseudo_E =", CCSD_E,"diff=", diff_E)
                #print("inter =", z,  "\t", "CCSD_E =", CCSD_E,"diff=", diff_E, "lam1E=", E1, "lam2E=", E2
                del lam1stored[0]
                del lam2stored[0]
                del errort1[0]
                del errort2[0]
            print("Lambda1 energy =", E1)
            print("Lambda2 energy =", E2)
            return CCSD_E, lam1, lam2

    def print_2(self, t1):
        #print("\n   The test function values:")
        #for i in range(F.shape[0]):
        #    for a in range(F.shape[1]):
        #        print i,"\t",  a, "\t", F[i][a]
        t1_tmp = t1.ravel()
        #sort_t1 = sorted(t1_tmp, key=lambda v: -v if v <0 else v, reverse=True) 
        sort_t1 = sorted(t1_tmp, reverse=True)
        for x in range(len(sort_t1)-1):
            
            if (round(sort_t1[x], 10) ==0e10 or round(sort_t1[x+1], 10) == round(sort_t1[x],10)):
                 
                pass
            else:
                print '\t', ('% 5.10f' %  sort_t1[x])
        print '\t', ('% 5.10f' %  sort_t1[-1])

    def print_T_amp(self, t1, t2):
        sort_t1 = sorted(t1.ravel())
        sort_t2 = sorted(t2.ravel())

        print("\n   The largest T1 values:")
        for x in range(len(sort_t1)):
            if (round(sort_t1[x], 5) ==0e5 or x % 2 or 30< x < 60 ):
                pass
            else: 
                print('\t', ('% 5.10f' %  sort_t1[x]))
       
        print("\n   The largest T2 values are:")

        for x in range(len(sort_t2)):
            if (round(sort_t2[x],2) ==0.00 or x % 2 or x > 20):
                pass
            else:
                print('\t', ('% 5.10f' %  sort_t2[x]))  
                
    def print_L_amp(self, lam1, lam2):
        sort_lam1 = sorted(-abs(lam1.ravel()))
        sort_lam2 = sorted(lam2.ravel())

        print("\n   The largest lam1 values:")
        for x in range(len(sort_lam1)):
            if (round(sort_lam1[x], 5) ==0e5 or x % 2 or x >20):
                pass
            else: 
                print('\t', ('% 5.10f' %  sort_lam1[x]))
        
        print("\n   The largest lam2 values are:")
        for x in range(len(sort_lam2)):
            if (round(sort_lam2[x],2) ==0.00 or x % 2 or x > 20):
                pass
            else:
                print('\t', ('% 5.10f' %  sort_lam2[x]))   
                
                  
 ##################################################################
 #
 #
 #                  Single-electron density matrix equations-derived by R. Glenn
 #
 #
 #####################################################################         
    #Dipoles in the MO basis
    def Defd_dipole(self):
        C = np.asarray(self.C)
        nmo = self.nmo
        tmp_dipoles = self.mints.so_dipole()
        dipoles_xyz = []
        for n in range(3):
            temp = contract('li,lk,kj->ij',C,tmp_dipoles[n],C)
            temp = temp.repeat(2, axis=1).repeat(2, axis=0)
            temp = temp*np.tile(np.identity(2),(nmo,nmo))
            dipoles_xyz.append(temp)
        return dipoles_xyz
    
    

    
    
    #Build Dvv 
    def Dij(self, t1, t2, lam1, lam2):
        term1 = contract('je,ie->ij', lam1, t1)
        term2 = 0.5*contract('jmea,imea->ij', lam2, t2)
        total = -(term1 + term2)
        return total
    
      #Build Doo 
    def Dab(self, t1, t2, lam1, lam2):
        term1 = contract('nb,na->ab', lam1, t1)
        term2 = 0.5*contract('mneb,mnea->ab', lam2, t2)
        total = term1 + term2
        return total  
        
      #Build Dvo
    def Dai(self, t1, t2, lam1, lam2):
        term1a = t1 
        term1 = contract('ia->ai', term1a)
        term2 = contract('me,miea->ai', lam1, t2)
        term3 = -contract('me,ma,ie->ai', lam1, t1, t1)
        term4 = -0.5*contract('mnef,mnaf,ie->ai', lam2, t2, t1)
        term5 = -0.5*contract('mnef,inef,ma->ai', lam2, t2, t1)
        total = term1 + term2 + term3 + term4 + term5
        return total
    #Dov is equal to lam1

    def Buildpho(self, F):
        o =self.occ
        S12, S12plus = self.GenS12()
        evals, evecs = np.linalg.eigh(F)
        C = contract('ij,jk->ik', S12, evecs)
        pho = contract('ik,jk->ij', C[:, o], np.conj(C[:, o]))
        return pho

    def pholowdinbasis(self, pho):
        S12, S12plus = self.GenS12()
        pholowdin = contract('il,lk,jk->ij', S12plus, pho, S12plus)
        return pholowdin 
                                     
    #For testing purposes only, to check my density as a function of time 
    def pho_checks(self, HF_p, corr_p, dip_xyz_corr):
        
        ##################################
        #
        #       Check the density, dipole, trace, idempotency
        #
        ####################################
        
        #get the correlated dipoles from psi to compare to
        dip_x = np.asarray(psi4.core.get_variable('CC DIPOLE X'))
        dip_y = np.asarray(psi4.core.get_variable('CC DIPOLE Y'))
        dip_z = np.asarray(psi4.core.get_variable('CC DIPOLE Z'))
        fac = 0.393456#The conversion factor from dybe to a.u.
        x_nuclear_dipole = 0.0 #H2O
        y_nuclear_dipole = 0.0 #H2O
        z_nuclear_dipole = 1.1273 #H2O
        dip_x = dip_x*fac -x_nuclear_dipole
        dip_y = dip_y*fac -y_nuclear_dipole
        dip_z = dip_z*fac -z_nuclear_dipole        
        

        #Compare calculated CC dipole to psi4
        print("This is the calculated electric in a. u. dipole \n", "x=", dip_xyz_corr[0], "y=", dip_xyz_corr[1], "z=", dip_xyz_corr[2])
        print("\n This is the psi4 electric in a. u. units dipole \n", "x=", dip_x, "y=", dip_y, "z=", dip_z)
        
        #Check that the p_trace_corr = 0, and p_trace_Hf =0
        p_trace_corr = np.sum(contract('ii->i', corr_p))
        #p_trace_tot = np.sum(contract('ii->i', ptot))  
        p_trace_HF = np.sum(contract('ii->i', HF_p)) 
        print("The trace of pho corr is", p_trace_corr,"\n")
        #print "The trace of pho is", p_trace_tot,"\n"
        print("The trace of pho HF is", p_trace_HF,"\n")       
        
        #Check the idempotency of HF
        p_sqd = contract('ij,kj->ik', HF_p, HF_p)
        #print "This is HF Density \n", HF_p, "\n This is HF p^2 \n", p_sqd, "\n"
        print("The difference between HF density p and p^2 should be zero \n", HF_p-p_sqd, "\n")

        #Check the idempotency of the total density ( It is not idempotent )
        ptot = HF_p + corr_p
        p_sqd = contract('ij,kj->ik', ptot, ptot)
        np.set_printoptions(precision=3)
        #print "This is total Density \n", ptot, "\n This is total p^2 \n", p_sqd, "\n"
        print("The difference between the total p and p^2 should be zero \n", ptot-p_sqd, "\n")
        
    #Build the expectation value of the dipole moment
    def dipole_moment(self, t1, t2, lam1, lam2, F):
        #Build the four blocks of the density matrix
        pai = self.Dai(t1, t2, lam1, lam2)
        pia = lam1 
        pab = self.Dab(t1, t2, lam1, lam2)
        pij = self.Dij(t1, t2, lam1, lam2)
        dipolexyz = self.Defd_dipole() 
        
        #Build the correlated density matrix
        left_p = np.vstack((pij, pai))
        right_p = np.vstack((pia, pab))
        corr_p = np.hstack((left_p, right_p))
        
        #Build the Hartree Fock Density matrix
        HF_p = self.Buildpho(F)
        HF_p = self.pholowdinbasis(HF_p)
        
        #Calculate the corr dipole moment
        dip_xyz_corr = []
        for i in range(3):
            temp = contract('ij,ij->', dipolexyz[i], HF_p + corr_p)
            #temp = contract('ij,ij->ij', dipolexyz[i], HF_p + corr_p)
            #temp = contract('ii', temp)
            dip_xyz_corr.append(temp)   
        
        #Check important characteristics before moving on
        #self.pho_checks(HF_p, corr_p, dip_xyz_corr)     
        return dip_xyz_corr             
        
########################################################
#
#
#
#       ###Functions for doing the Time integration
#
#
###########################################################
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




    def Save_data(self, F, t1, t2, lam1, lam2, data, timing, precs, restart):
        if restart is None: 
            data.to_csv('H2O.csv')
            timing.to_csv('timing.csv')
        else:
            with open('H2O.csv', 'a') as f:
                data.to_csv(f, header=False)
            with open('timing.csv', 'a') as f:
                timing.to_csv(f, header=False) 
              
        #np.savetxt('t1.dat', t1, fmt='%.10e%+.10ej '*t1.shape[1])
        #np.savetxt('t2.dat', t2.flatten(), fmt='%.10e%+.10ej ')
        #np.savetxt('lam1.dat', lam1, fmt='%.10e%+.10ej '*lam1.shape[1])
        #np.savetxt('lam2.dat', lam2.flatten(), fmt='%.10e%+.10ej ')
        ##############save the data values plus the indices##################
        self.write_2data(F.real, 'F_real.dat', precs)
        self.write_2data(F.imag, 'F_imag.dat', precs)
        self.write_2data(t1.real, 't1_real.dat', precs)
        self.write_2data(t1.imag, 't1_imag.dat', precs)
        self.write_4data(t2.real, 't2_real.dat', precs)
        self.write_4data(t2.imag, 't2_imag.dat', precs)
        self.write_2data(lam1.real, 'lam1_real.dat', precs)
        self.write_2data(lam1.imag, 'lam1_imag.dat', precs)
        self.write_4data(lam2.real, 'lam2_real.dat', precs)
        self.write_4data(lam2.imag, 'lam2_imag.dat', precs)
                ######save just the data values##############
        #np.savetxt('F.dat', F, fmt='%.10e '*F.shape[1])
        #np.savetxt('t1_real.dat', t1.real.flatten(), fmt='%.10f')
        #np.savetxt('t1_imag.dat', t1.imag.flatten(), fmt='%.10f')
        #np.savetxt('t2_real.dat', t2.real.flatten(), fmt='%.10f')
        #np.savetxt('t2_imag.dat', t2.imag.flatten(), fmt='%.10f')
        #np.savetxt('lam1_real.dat', lam1.real.flatten(), fmt='%.10f')
        #np.savetxt('lam1_imag.dat', lam1.imag.flatten(), fmt='%.10f')
        #np.savetxt('lam2_real.dat', lam2.real.flatten(), fmt='%.10f')
        #np.savetxt('lam2_imag.dat', lam2.imag.flatten(), fmt='%.10f')



###############################################
#        
#            
    ##Rosenbrock Integrator 4th-order
#
#    "Parallel exponential Rosenbrock methods, 
        #Vu Thai Luana, Alexander Ostermannb"
################################################

 
 ########################################################
#       ###F, G, and A terms of the RHS of t1, t2, l1, l2 for doing the 
#                RosenbrockTime integration
###########################################################

    def T1_OSC_terms(self, t1,F):
        TEI = self.TEI
        ndocc = 2*self.ndocc
        nmo = 2*self.nmo
        cons = np.zeros(shape=(ndocc, nmo-ndocc))
        for i in range(ndocc):
            for a in range(ndocc, nmo):
                cons[i, a-ndocc] = F[a, a]- F[i, i] - TEI[i, a, i, a]
        return cons
        
    def T2_OSC_terms(self, t2, F): 
        TEI = self.TEI
        ndocc = 2*self.ndocc
        nmo = 2*self.nmo
        cons = np.zeros(shape=(ndocc, ndocc, nmo-ndocc, nmo-ndocc))
        for i in range(ndocc):
            for j in range(ndocc):
                for a in range(ndocc, nmo):
                    for b in range(ndocc, nmo):
                        cons[i][j][a-ndocc][b-ndocc] = F[b, b] + F[a, a] - F[j, j] - F[i, i] + 0.5*TEI[i, j, i, j] \
                                         + 0.5*TEI[a, b, a, b] + 0.5*TEI[j, b, b, j] + 0.5*TEI[i, b, b, i]\
                                         + 0.5*TEI[i, a, a, i] + 0.5*TEI[j, a, a, j]
        return cons
        
    def T1eq_rhs_TD(self, t1, t2, F, Vt):
        v = self.vir
        o = self.occ  
        #All rhs terms:
        t1 = self.T1eq_rhs(t1, t2, F + Vt) 
        #constant terms
        #t1_cons = F[o, v].copy()
        #constant *T_i^a
        t1_cons_t1 = contract('ia, ia->ia', self.T1_OSC_terms(t1, F), t1)
        #function * T_i^a
        return t1 - t1_cons_t1 #- t1_cons_t1
        
        
    def T2eq_rhs_TD(self, t1, t2, F, Vt):
        v = self.vir
        o = self.occ  
        TEI = self.TEI 
        #All rhs terms:
        t2 = self.T2eq_rhs(t1, t2, F + Vt) 
        #constant terms
        #t2_cons = TEI[o, o, v, v].copy()
        #constant *T_ij^ab
        t2_cons_t2  = contract('ijab, ijab->ijab', self.T2_OSC_terms(t2, F), t2)
        #function * T_ij^ab
        return t2 - t2_cons_t2 #- t2_cons_t2

                       
    def L1eq_rhs_TD(self, t1, t2, lam1, lam2, F, Vt):
        v = self.vir
        o = self.occ  
        #All rhs terms: 
        lam1 = self.lam_1eq_rhs(t1, t2, lam1, lam2, F + Vt) 
        #constant terms
        #lam1_cons = F[o, v].copy()
        #constant *L_i^a
        lam1_cons_lam1  = contract('ia, ia->ia', self.T1_OSC_terms(lam1, F), lam1)
        #function * L_i^a
        return lam1 - lam1_cons_lam1 #- lam1_cons_lam1
    
    def L2eq_rhs_TD(self, t1, t2, lam1, lam2, F, Vt):
        v = self.vir
        o = self.occ  
        TEI = self.TEI 
        #All rhs terms:
        lam2 = self.lam2eq_rhs(t1, t2, lam1, lam2, F + Vt) 
        #constant terms
        #lam2_cons = TEI[o, o, v, v].copy()
        #constant *L_ij^ab
        lam2_cons_lam2  = contract('ijab, ijab->ijab', self.T2_OSC_terms(lam2, F), lam2)
        #function * L_ij^ab
        return lam2 - lam2_cons_lam2 #- lam2_cons_lam2                

########END F, G, A parameters####################


############Time propagation###################
    def Rosenbrock(self, F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, restart=None):
        
        #propagates any of the  T1, T2, L1  or L2 functions: 
        
         #Setup Pandas Data and time evolution
        data =  pd.DataFrame( columns = ( 'time', 'mu_real', 'mu_imag')) 
        timing =  pd.DataFrame( columns = ( 'total','t1', 't2', 'l1','l2')) 
        m=10.0
        #        ##Electric field, it is in the z-direction for now      
        def Vt(t):
            mu = self.Defd_dipole()
            return -A*mu[2]*np.sin(2*np.pi*w0*t)*np.exp(-t*t/5.0)   
        t = t0       
        
        ##################functions for Rosenbrock###############
        c2 =0.5
        c3 =1.0
        #Needed functions 
        
        def phi1(z):
            return (cmath.exp(z) -1)/z
        #print phi1(A)
        def phi2(z):
            return (cmath.exp(z) -1-z)/z**2
        def phi3(z):
            return -(2 - 2*cmath.exp(z)  +2*z+np.power(z, 2))/( 2*np.power(z, 3))
        def phi4(z):
            return -(6 - 6*cmath.exp(z)  + 6*z+3*np.power(z, 2)+np.power(z, 3))/( 6*np.power(z, 4))
        def b2(z):
            return 16*phi3(z) - 48*phi4(z)
        def b3(z):
            return -2*phi3(z) + 12*phi4(z)   
        def b2_bar(z):
            return 16*phi3(z)
        def b3_bar(z):
            return -2*phi3(z) 
        Aia = -1j*self.T1_OSC_terms(t1, F)
        
        occ = 2*self.ndocc
        vir = 2*(-self.ndocc + self.nmo)
        pphi1 = np.zeros(shape=(occ, vir), dtype=np.complex) 
        pphi1c2 = np.zeros(shape=(occ, vir), dtype=np.complex)
        pphi1c3 = np.zeros(shape=(occ, vir), dtype=np.complex)
        pphi3 = np.zeros(shape=(occ, vir), dtype=np.complex) 
        pphi4= np.zeros(shape=(occ, vir), dtype=np.complex) 
        for i in range(Aia.shape[0]):
            for a in range(Aia.shape[1]):
                x =  abs(Aia[i, a])           
                if  np.around(x, decimals=6) == 0.00000000000:
                    pphi1[i,a] = 1.0
                    pphi1c2[i,a] = 0.0
                    pphi1c3[i,a] = 0.0
                    pphi3[i,a] = 1.0
                    pphi4[i,a] = 1.0   
                else:   
                    pphi1[i,a] = phi1(dt*Aia[i,a])
                    pphi1c2[i,a] = phi1(c2*dt*Aia[i,a])
                    pphi1c3[i,a] = phi1(c3*dt*Aia[i,a])
                    pphi3[i,a] = phi3(dt*Aia[i,a])
                    pphi4[i,a] = phi4(dt*Aia[i,a])  

        Aijab = -1j*self.T2_OSC_terms(t2, F)
        phhi1 = np.zeros(shape=(occ, occ, vir, vir), dtype=np.complex) 
        phhi1c2 = np.zeros(shape=(occ, occ, vir, vir), dtype=np.complex)
        phhi1c3 = np.zeros(shape=(occ, occ, vir, vir), dtype=np.complex) 
        phhi3 = np.zeros(shape=(occ, occ, vir, vir), dtype=np.complex) 
        phhi4= np.zeros(shape=(occ, occ, vir, vir), dtype=np.complex)
        for i in range(occ):
            for j in range(occ):
                for a in range(vir):
                    for b in range(vir):   
                        x =  abs(Aijab[i, j, a, b])           
                        if  np.around(x, decimals=6) == 0.00000000000:
                            phhi1[i, j, a, b] = 1.0
                            phhi1c2[i, j, a, b] = 1.0
                            phhi1c3[i, j, a, b] = 1.0
                            phhi3[i, j, a, b] = 1.0
                            phhi4[i, j, a, b] = 1.0   
                        else:   
                            phhi1[i, j, a, b] = phi1(dt*Aijab[i, j, a, b])
                            phhi1c2[i, j, a, b] = phi1(c2*dt*Aijab[i, j, a, b])
                            phhi1c3[i, j, a, b] = phi1(c3*dt*Aijab[i, j, a, b])
                            phhi3[i, j, a, b] = phi3(dt*Aijab[i, j, a, b])
                            phhi4[i, j, a, b] = phi4(dt*Aijab[i, j, a, b])     
        
        b_12 = 16*pphi3 - 48*pphi4
        b_13 = -2*pphi3 + 12*pphi4
        b_22 = 16*phhi3 - 48*phhi4
        b_23 = -2*phhi3 + 12*phhi4  
        del pphi3
        del pphi4
        del phhi3
        del phhi4

        ##################END functions for Rosenbrock###############
        
        i=0
        start = time.time()
        #Do the time propagation
        while t < tf:
            L1min = np.around(lam1, decimals=precs) 
            L2min = np.around(lam2, decimals=precs) 
            dt = dt/m
            itertime_t1 = itertime_t2 = 0
            for n in range(int(m)):
                t1min = np.around(t1, decimals=precs) 
                t2min = np.around(t2, decimals=precs) 
                itertime = time.time()        
        
        
                ####T1 propagation##########
                un = np.around(t1, decimals=precs)           
                Fa = -1j*self.T1eq_rhs(un, t2, F + Vt(t))
                def gn(t, un):
                    return -1j*self.T1eq_rhs_TD(un, t2, F, Vt(t))   
            
                Un2 = un+ c2*dt*contract('ia,ia->ia', pphi1c2, Fa)
                Un3 = un + c3*dt*contract('ia,ia->ia', pphi1c3, Fa)
                Dn2 = gn(t, Un2) - gn(t, un) 
                Dn3 = gn(t, Un3) - gn(t, un)  
                dt1 = dt*contract('ia,ia->ia', pphi1, Fa) 
                dt1 = dt1 + dt*contract('ia,ia->ia', b_12, Dn2) + dt*contract('ia,ia->ia', b_13, Dn3)
                
                itertime_t1 = -itertime + time.time()
                itertime = time.time()
                #####T2 propagation
                un = np.around(t2, decimals=precs) 
                Fa = -1j*self.T2eq_rhs(t1, un, F + Vt(t))
                def gn(t, un):
                    return -1j*self.T2eq_rhs_TD(t1, un, F, Vt(t))           
            
                Un2 = un + c2*dt*contract('ijab,ijab->ijab', phhi1c2, Fa)
                Un3 = un + c3*dt*contract('ijab,ijab->ijab', phhi1c3, Fa)
                Dn2 = gn(t, Un2) - gn(t, un)  
                Dn3 = gn(t, Un3) - gn(t, un) 
                dt2 = dt*contract('ijab,ijab->ijab', phhi1, Fa) 
                dt2 = dt2 + dt*contract('ijab,ijab->ijab', b_22, Dn2) + dt*contract('ijab,ijab->ijab', b_23, Dn3)

                itertime_t2 = -itertime + time.time()
            
            dt = m*dt
            itertime = time.time()
            ########L1 propagation
            un = np.around(lam1, decimals=precs) 
            Fa =1j*self.lam_1eq_rhs(t1, t2, un, lam2, F + Vt(t))
            def gn(t, un):
                return 1j*self.L1eq_rhs_TD(t1, t2, un, lam2, F, Vt(t))
            
            Un2 = un+ c2*dt*contract('ia,ia->ia', np.conjugate(pphi1c2), Fa)
            Un3 = un + c3*dt*contract('ia,ia->ia', np.conjugate(pphi1c3), Fa)
            #Dn2 =1j*self.lam_1eq_rhs(t1, t2, Un2, lam2, F + Vt(t)) - contract('ia,ia->ia', Aia, Un2)
            #Dn2 = Dn2 - Fa - contract('ia,ia->ia', Aia, un)
            #Dn3 =1j*self.lam_1eq_rhs(t1, t2, Un3, lam2, F + Vt(t)) - contract('ia,ia->ia', Aia, t1) 
            #Dn3 = Dn3 - Fa - contract('ia,ia->ia', Aia, un)
            Dn2 = gn(t, Un2) - gn(t, un) 
            Dn3 = gn(t, Un3) - gn(t, un) 
            dL1 = dt*contract('ia,ia->ia', np.conjugate(pphi1), Fa) 
            dL1 = dL1 + dt*contract('ia,ia->ia', np.conjugate(b_12), Dn2) + dt*contract('ia,ia->ia', np.conjugate(b_13), Dn3)
                
            itertime_l1 = -itertime  + time.time()
            itertime = time.time()   
            ##########L2 propagation
            un = np.around(lam2, decimals=precs) 
            Fa = 1j*self.lam2eq_rhs(t1, t2, lam1, un, F + Vt(t))
            def gn(t, un):
                return 1j*self.L2eq_rhs_TD(t1, t2, lam1, un, F, Vt(t))

            Un2 = un+ c2*dt*contract('ijab,ijab->ijab', np.conjugate(phhi1c2), Fa)
            Un3 = un + c3*dt*contract('ijab,ijab->ijab',np.conjugate(phhi1c3), Fa)
            Dn2 = gn(t, Un2) - gn(t, un)  
            Dn3 = gn(t, Un3) - gn(t, un)  
            dL2 = dt*contract('ijab,ijab->ijab', np.conjugate(phhi1), Fa) 
            dL2 = dL2 + dt*contract('ijab,ijab->ijab', np.conjugate(b_22), Dn2) + dt*contract('ijab,ijab->ijab', np.conjugate(b_23), Dn3)
            
            itertime_l2 = -itertime  + time.time()
            total = itertime_t1 + itertime_t2 + itertime_l1 + itertime_l2
            timing.loc[i] = [total, itertime_t1, itertime_t2, itertime_l1, itertime_l2 ]
            ####Update t1, t2, l1, l2
            t1 = t1min + dt1
            t2 = t2min + dt2
            lam1 = L1min + dL1
            lam2 = L2min + dL2
            i += 1
            t =t0 + i*dt
            stop = time.time()-start
            
            if abs(stop)>0.9*timeout*60.0:
                print('The file timed out just before',0.9*timeout*60.0, 'sec')
                self.Save_data(F, t1, t2, lam1, lam2, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t, dt, precs, t1.shape[0], t1.shape[1])
                
                break
            #Calculate the dipole moment using the density matrix
            mua = self.dipole_moment(t1, t2, lam1, lam2, F)
            data.loc[i] = [ t, mua[2].real, mua[2].imag  ]
            print(t, mua[2])
            
            if abs(mua[2].imag) > 100:
                print('The dipole was greater than 100, UNSTABLE, file timed out at approx.',0.9*timeout*60.0, 'sec')
                self.Save_data(F, t1, t2, lam1, lam2, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t, dt, precs, t1.shape[0], t1.shape[1])
                break

        stop = time.time()
        print("total time non-adapative step:", stop-start)
        print("total steps:", i)
        print("step-time:", (stop-start)/i)
        self.Save_data(F, t1, t2, lam1, lam2, data, timing, precs, restart)
        self.Save_parameters(w0, A, t0, t, dt, precs,  t1.shape[0], t1.shape[1])
                      

        
###############################################
#        
#            
#     Runge-Kutta time dependent propagator
#
#    
################################################
 
 ########Functions for Runge-Kutta#################
 #########for T1, T2, L1, L2#######################
       #T1 Runge-Kutta function 
    def ft1(self, t, dt, t1, t2, F, Vt):  
        k1 = self.T1eq_rhs(t1, t2, F + Vt(t))
        k2 = self.T1eq_rhs(t1 + dt/2.0*k1, t2, F + Vt(t + dt/2.0)) 
        k3 = self.T1eq_rhs(t1 + dt/2.0*k2, t2, F + Vt(t + dt/2.0))
        k4 = self.T1eq_rhs(t1 + dt*k3, t2, F + Vt(t + dt))  
        #k1 = self.T1eq_rhs_TD(t1, t2, F, Vt(t))
        #k2 = self.T1eq_rhs_TD(t1 + dt/2.0*k1, t2, F, Vt(t + dt/2.0)) 
        #k2 = self.T1eq_rhs_TD(t1 + dt/2.0*k1, t2, F, Vt(t + dt/2.0))  
        #k3 = self.T1eq_rhs_TD(t1 + dt/2.0*k2, t2, F, Vt(t + dt/2.0))
        #k4 = self.T1eq_rhs_TD(t1 + dt*k3, t2, F, Vt(t + dt)) 
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
         
    #T2 Runge-Kutta function 
    def ft2(self, t, dt, t1, t2, F, Vt):
        k1 = self.T2eq_rhs(t1, t2, F + Vt(t))
        k2 = self.T2eq_rhs(t1, t2 + dt/2.0*k1, F + Vt(t + dt/2.0))  
        k3 = self.T2eq_rhs(t1, t2 + dt/2.0*k2, F + Vt(t + dt/2.0)) 
        k4 = self.T2eq_rhs(t1, t2 + dt*k3,  F + Vt(t + dt)) 
        #k1 = self.T2eq_rhs_TD(t1, t2, F, Vt(t))
        #k2 = self.T2eq_rhs_TD(t1, t2 + dt/2.0*k1, F, Vt(t + dt/2.0))  
        #k3 = self.T2eq_rhs_TD(t1, t2 + dt/2.0*k2, F, Vt(t + dt/2.0)) 
        #k4 = self.T2eq_rhs_TD(t1, t2 + dt*k3,  F, Vt(t + dt)) 
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
                 
    #L1 Runge-Kutta function 
    def fL1(self, t, dt, t1, t2, lam1, lam2, F, Vt):
        k1 = self.lam_1eq_rhs(t1, t2, lam1, lam2, F + Vt(t))
        k2 = self.lam_1eq_rhs(t1, t2, lam1 + dt/2.0*k1, lam2, F + Vt(t + dt/2.0))  
        k3 = self.lam_1eq_rhs(t1, t2, lam1 + dt/2.0*k2, lam2, F + Vt(t + dt/2.0)) 
        k4 = self.lam_1eq_rhs(t1, t2, lam1 + dt*k3, lam2, F + Vt(t + dt)) 
        #k1 = self.L1eq_rhs_TD(t1, t2, lam1, lam2, F, Vt(t))
        #k2 = self.L1eq_rhs_TD(t1, t2, lam1 + dt/2.0*k1, lam2, F, Vt(t + dt/2.0))  
        #k3 = self.L1eq_rhs_TD(t1, t2, lam1 + dt/2.0*k2, lam2, F, Vt(t + dt/2.0)) 
        #k4 = self.L1eq_rhs_TD(t1, t2, lam1 + dt*k3, lam2, F, Vt(t + dt)) 
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)  
           
   #L2 Runge-Kutta function  
    def fL2(self, t, dt, t1, t2, lam1, lam2, F, Vt):
        k1 = self.lam2eq_rhs(t1, t2, lam1, lam2, F + Vt(t))
        k2 = self.lam2eq_rhs(t1, t2, lam1, lam2 + dt/2.0*k1, F + Vt(t + dt/2.0))  
        k3 = self.lam2eq_rhs(t1, t2, lam1, lam2 + dt/2.0*k2, F + Vt(t + dt/2.0)) 
        k4 = self.lam2eq_rhs(t1, t2, lam1, lam2 + dt*k3, F + Vt(t + dt)) 
        #k1 = self.L2eq_rhs_TD(t1, t2, lam1, lam2, F, Vt(t))
        #k2 = self.L2eq_rhs_TD(t1, t2, lam1, lam2 + dt/2.0*k1, F, Vt(t + dt/2.0))  
        #k3 = self.L2eq_rhs_TD(t1, t2, lam1, lam2 + dt/2.0*k2, F, Vt(t + dt/2.0)) 
        #k4 = self.L2eq_rhs_TD(t1, t2, lam1, lam2 + dt*k3, F, Vt(t + dt)) 
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)      
############END functions for Runge-Kutta#############
     
    ####Time propagator#############
    def Runge_Kutta_solver(self, F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, restart=None):
        #Setup Pandas Data and time evolution
       
        data =  pd.DataFrame( columns = ('time', 'mu_real', 'mu_imag')) 
        timing =  pd.DataFrame( columns = ('total','t1', 't2', 'l1','l2')) 
        
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
            L1min = np.around(lam1, decimals=precs) 
            L2min = np.around(lam2, decimals=precs) 
            dt = dt/m
            itertime_t1 = itertime_t2 = 0
            for n in range(int(m)):
                t1min = np.around(t1, decimals=precs) 
                t2min = np.around(t2, decimals=precs) 
                itertime = time.time()
                dt1 = -1j*self.ft1(t, dt, t1, t2, F, Vt) #Runge-Kutta
                itertime_t1 = -itertime + time.time()
                itertime = time.time()
                dt2 = -1j*self.ft2(t, dt, t1, t2, F, Vt) #Runge-Kutta
                itertime_t2 = -itertime + time.time()
            dt = m*dt
            itertime = time.time()
            dL1 = 1j*self.fL1(t, dt, t1, t2, lam1, lam2, F, Vt) #Runge-Kutta
            itertime_l1 = -itertime  + time.time()
            itertime = time.time()
            dL2 = 1j*self.fL2(t, dt, t1, t2, lam1, lam2, F, Vt)  #Runge-Kutta
            itertime_l2 = -itertime  + time.time()
            total = itertime_t1 + itertime_t2 + itertime_l1 + itertime_l2
            timing.loc[i] = [total, itertime_t1, itertime_t2, itertime_l1, itertime_l2 ]
            t1 = t1min + dt1
            t2 = t2min + dt2
            lam1 = L1min + dL1
            lam2 = L2min + dL2
            i += 1
            t =t0 + i*dt
            stop = time.time()-start
            mua = self.dipole_moment(t1, t2, lam1, lam2, F)
            data.loc[i] = [t, mua[2].real, mua[2].imag  ]
            print(t, mua[2])
            
            if abs(stop)>0.9*timeout*60.0:
                
                #self.Save_data(F, t1, t2, lam1, lam2, data, timing, restart)
                self.Save_data(F, t1min, t2min, L1min, L2min, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t-dt, dt, precs, t1.shape[0], t1.shape[1])
    
                break
            #Calculate the dipole moment using the density matrix

            
            if abs(mua[2].real) > 100:
                #self.Save_data(F, t1, t2, lam1, lam2, data, timing, restart)
                self.Save_data(F, t1min, t2min, L1min, L2min, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t-dt, dt, precs, t1.shape[0], t1.shape[1])
                break
            
        stop = time.time()
        print("total time non-adapative step:", stop-start)
        print("total steps:", i)
        print("step-time:", (stop-start)/i)
        #self.Save_data(F, t1, t2, lam1, lam2, data, timing, restart)
        #self.Save_data(F, t1min, t2min, L1min, L2min, data, timing, restart)
#self.Save_parameters(w0, A, t0, t-dt, dt, precs, t1.shape[0], t1.shape[1])











#
#rhs_L2(int L_irr, double E0_Real, double E0_Imag)


#L2_plus_delta_L2(int L_irr)



#init_io_L2()
#exit_io_L2()
#init_io_onepdm()
#exit_io_onepdm()


#RK_TO_RHS_io_L2(int L_irr)

#RHS_to_RK_io_L2(int L_irr)






























