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
       

    def Fkc(self, t1, t2, F): 
        v = self.vir
        o = self.occ
        term1 = F[o, v]
        term2 = contract('klcd,ld->kc', TEI[o, o, v, v],t1)
        return term1 + term2


    #Build Fvv
    def Fae(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        fkc = F[o, v]
        Fkc = self.Fkc(t1, t2, F)
        term1 = F[v, v].copy()
        term2 = contract('akcd,kd->ca', TEI[v, o, v, v], t1)
        term3= -0.5*contract('kc,ka->ca' Fkc, t1)
        term4 = -0.5*contract('kc,ka->ca', fkc, t1)
        tau = 
        term5 = -0.5*contract('klcd,klad->ca', TEI[o, o, v, v], tau)



        term2 = - 0.5*contract('me,ma->ae', F[o, v], t1)
        term3 = contract('mafe,mf->ae', TEI[o, v, v, v], t1)
        tau = t2.copy() + contract('ia,jb->ijab', t1, t1)
        term4 =-0.5*contract('mnef,mnaf->ae', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4
        return total
