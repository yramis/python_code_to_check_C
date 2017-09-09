import sys
import os
sys.path.insert(0,'./..')
sys.path.append(os.environ['HOME']+'/Desktop/workspace/psi411/psi4/objdir/stage/usr/local/lib')
sys.path.append(os.environ['HOME']+'/miniconda2/lib/python2.7/site-packages')
sys.path.append('/usr/local/psi4/lib')
sys.path.append('/home/rglenn/blueridge/buildpsi/lib')
import psi4 as psi4
from CCSD_Helper import *
import csv
import pandas as pd
#if os.environ['SYSNAME']=='blueridge':
psi4.core.set_memory(int(62e9), False) #blueridge
#psi4.core.set_memory(int(3.5e9), False) 

timeout = float(sys.argv[1])/60
print("allocated time in minutes is:", timeout)


#psi4.core.set_memory(int(100.e6), False) #my laptop
#psi4.core.clean()
numpy_memory = 2
#psi4.core.clean()
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

opt_dict = {
  "basis": '6-31g',
  "reference": "RHF",
  "mp2_type": "conv",
  "roots_per_irrep": [40],
  "scf_type": "pk",
  'e_convergence': 1e-14,
  'r_convergence': 1e-14
}
psi4.set_options(opt_dict)
#psi('ccsd', properties=['dipole'])
#psi4.property('eom-cc2', properties=['oscillator_strength'])
psi4.core.set_output_file('output.dat', False)

pseudo = -0.068888224492060 #H2O sto-3g
pseudo = -0.140858583055215 #'3-21g
pseudo = -0.148311233718836 #'6-31g

-0.16484162084478067637
-0.079490341352383462

#reading in all necessary data
#t1 = np.loadtxt("t1.dat", dtype = np.complex128)
#n=t1.shape[0]
#m=t1.shape[1]
#t2 = np.loadtxt("t2.dat", dtype = np.complex128)
#t2= t2.reshape(n,n,m,m)
#lam1 = np.loadtxt("lam1.dat", dtype = np.complex128)
#lam2 = np.loadtxt("lam2.dat", dtype = np.complex128)
#lam2= lam2.reshape(n,n,m,m)
#TEI = np.loadtxt("TEI.dat")
#TEI = TEI.reshape(n,n,m,m)



#########################3
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
###################################'
#t1_real = np.loadtxt("t1_real.dat")
#t1_real = t1_real.reshape(i, a)
#t1_imag = np.loadtxt("t1_imag.dat")
#t1_imag = t1_imag.reshape(i, a)
#t1 = t1_real + 1.0*1j*t1_imag

#t2_real = np.loadtxt("t2_real.dat")
#t2_real = t2_real.reshape(i, i, a, a)
#t2_imag = np.loadtxt("t2_imag.dat")
#t2_imag = t2_imag.reshape(i, i, a, a)
#t2 = t2_real + 1.0*1j*t2_imag

#lam1_real = np.loadtxt("lam1_real.dat")
#lam1_real = lam1_real.reshape(i, a)
#lam1_imag = np.loadtxt("lam1_imag.dat")
#lam1_imag = lam1_imag.reshape(i, a)
#lam1 = lam1_real + 1.0*1j*lam1_imag

#lam2_real = np.loadtxt("lam2_real.dat")
#lam2_real = lam2_real.reshape(i, i, a, a)
#lam2_imag = np.loadtxt("lam2_imag.dat")
#lam2_imag = lam2_imag.reshape(i, i, a, a)
#lam2 = lam2_real + 1.0*1j*lam2_imag

################################
#F = np.loadtxt("F.dat")

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
#F = np.loadtxt("F.dat")


#
#np.savetxt('shape4.dat', shape4.flatten(), fmt='%2.i' )
####4th-order Rosenbrock "Parallel exponential Rosenbrock methods, 
#Vu Thai Luana, Alexander Ostermannb"
#t0 = t0 - dt
mol = CCSD_Helper(psi4)
#mol.Rosenbrock(F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, 'restart')   
######4th-order Runge-Kutta   
mol.Runge_Kutta_solver(F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, 'restart')
#for x in range(5):
#    for y in range(5):
#        print x, y, t1.real[x][y]
#for x in range(5):
#    for y in range(5):
#        for t in range(5):
#            for z in range(5):
#                print x, y, t, z, t2.real[x][y][t][z]

