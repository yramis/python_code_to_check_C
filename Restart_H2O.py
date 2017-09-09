import sys
import os
sys.path.insert(0,'./..')
import psi4 as psi4
from CCSD_Helper import *
import csv
import pandas as pd

psi4.core.set_memory(int(62e9), False) #blueridge

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
  "basis": "6-31g",
  "reference": "RHF",
  "mp2_type": "conv",
  "roots_per_irrep": [40],
  "scf_type": "pk",
  "e_convergence": 1e-14,
  "r_convergence": 1e-14
}
psi4.set_options(opt_dict)
psi4.core.set_output_file('output.dat', False)

param = pd.read_csv('Parameters.csv')     
#Start parameters
w0 = param['w0'][0]#frequency of the oscillation and transition frequency
A = param['A'][0]#the amplitude of the electric field    
t0 = param['t0'][0]  #the start time
tf = 50.0 + t0 #the stop time, the actual stop time is governed by the timelength of the job
                     #Unless it completes enough steps to get to tf first. 
dt =  param['dt'][0] #time step
precs = int(param['precs'][0])
i = int(param['i'][0])
a = int(param['a'][0])

def convert_2data(Filename, i, a):
    F = pd.read_csv(Filename,sep="\t",header=None, names=['i', 'a', 'F'])
    #F = np.genfromtxt(("\t".join(i) for i in csv.reader(open(Filename))), delimiter="\t")
    Freshape = np.zeros(shape=(i, a))
    x = np.around(F['i'].tolist())
    y = np.around(F['a'].tolist())
    Fa = F['F'].tolist()
    for n in range(len(Fa)):
        ni = int(x[n])
        na = int(y[n])
        Freshape[ni][na] = Fa[n]
    return Freshape

def convert_4data(Filename, i, a):
    F = pd.read_csv(Filename,sep="\t",header=None, names=['i', 'j', 'a', 'b', 'F'])
    #F = np.genfromtxt(("\t".join(i) for i in csv.reader(open(Filename))), delimiter="\t")
    Freshape = np.zeros(shape=(i, i, a, a))
    x = np.around(F['i'].tolist())
    y = np.around(F['j'].tolist())
    t = np.around(F['a'].tolist())
    z = np.around(F['b'].tolist())
    Fa = F['F'].tolist()
    for n in range(len(Fa)):
        ni = int(x[n])
        nj = int(y[n])
        na = int(t[n])
        nb = int(z[n])
        Freshape[ni][nj][na][nb] = Fa[n]
    return Freshape


#######The data for t1 is "i", "a", "t1-flattend"############################
t1_real = convert_2data("t1_real.dat", i, a)
t1_imag = convert_2data("t1_imag.dat", i, a)
t1 = t1_real + 1.0*1j*t1_imag

lam1_real = convert_2data("lam1_real.dat", i, a)
lam1_imag = convert_2data("lam1_imag.dat", i, a)
lam1 = lam1_real + 1.0*1j*lam1_imag

t2_real = convert_4data("t2_real.dat", i, a)
t2_imag = convert_4data("t2_imag.dat", i, a)
t2 = t2_real + 1.0*1j*t2_imag

lam2_real = convert_4data("lam2_real.dat", i, a)
lam2_imag = convert_4data("lam2_imag.dat", i, a)
lam2 = lam2_real + 1.0*1j*lam2_imag


F_real = convert_2data("F_real.dat", i+a, i+a )
F_imag = convert_2data("F_imag.dat", i+a, i+a)
F = F_real #+ 1.0*1j*F_imag
mol = CCSD_Helper(psi4)

mol.Runge_Kutta_solver(F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, 'restart')


