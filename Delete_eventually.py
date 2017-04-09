import sys
import os
sys.path.insert(0,'./..')
sys.path.append(os.environ['HOME']+'/Desktop/workspace/psi411/psi4/objdir/stage/usr/local/lib')
sys.path.append(os.environ['HOME']+'/miniconda2/lib/python2.7/site-packages')
sys.path.append('/usr/local/psi4/lib')
sys.path.append('/home/rglenn/blueridge/buildpsi/lib')
import psi4 as psi4
from CCSD_Helper import *
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
#psi4.property('ccsd', properties=['dipole'])
#psi4.property('eom-cc2', properties=['oscillator_strength'])
psi4.core.set_output_file('output.dat', False)

pseudo = -0.068888224492060 #H2O sto-3g
pseudo = -0.140858583055215 #'3-21g
pseudo = -0.148311233718836 #'6-31g


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
t1_real = np.loadtxt("t1_real.dat") 
t1_real = t1_real.reshape(i, a)
t1_imag = np.loadtxt("t1_imag.dat") 
t1_imag = t1_imag.reshape(i, a)
t1 = t1_real + 1.0*1j*t1_imag

t2_real = np.loadtxt("t2_real.dat") 
t2_real = t2_real.reshape(i, i, a, a)
t2_imag = np.loadtxt("t2_imag.dat") 
t2_imag = t2_imag.reshape(i, i, a, a)
t2 = t2_real + 1.0*1j*t2_imag

lam1_real = np.loadtxt("lam1_real.dat") 
lam1_real = lam1_real.reshape(i, a)
lam1_imag = np.loadtxt("lam1_imag.dat") 
lam1_imag = lam1_imag.reshape(i, a)
lam1 = lam1_real + 1.0*1j*lam1_imag

lam2_real = np.loadtxt("lam2_real.dat") 
lam2_real = lam2_real.reshape(i, i, a, a)
lam2_imag = np.loadtxt("lam2_imag.dat") 
lam2_imag = lam2_imag.reshape(i, i, a, a)
lam2 = lam2_real + 1.0*1j*lam2_imag

################################
F = np.loadtxt("F.dat")
shape2i = np.zeros(shape=(i,a))
shape2a = np.zeros(shape=(i,a))
for x in range(i):
    for y in range(a):
        shape2i[x][y] = x
        shape2a[x][y] = y

shape4i = np.zeros(shape=(i,i,a,a))
shape4a = np.zeros(shape=(i,i,a,a))
shape4ii = np.zeros(shape=(i,i,a,a))
shape4aa = np.zeros(shape=(i,i,a,a))
for x in range(i):
    for y in range(i):
        for t in range(a):
            for z in range(a):
                shape4i[x][y][t][z] = x
                shape4ii[x][y][t][z] = y
                shape4a[x][y][t][z] = t
                shape4aa[x][y][t][z] = z


import csv
from io import BytesIO
with open('shape2.dat', 'a') as outcsv:   
        #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                #writer.writerow(['number', 'text', 'number'])
    for x in range(i*a):
                    #Write item to outcsv
        writer.writerow([shape2i.flatten()[x], shape2a.flatten()[x]])
#np.savetxt('shape2.dat', shape2.flatten(), fmt='%04i')

with open('shape4.dat', 'a') as outcsv:   
        #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                #writer.writerow(['number', 'text', 'number'])
    for x in range(i*a*i*a):
                    #Write item to outcsv
        writer.writerow([shape4i.flatten()[x], shape4i.flatten()[x],shape4a.flatten()[x], shape4aa.flatten()[x]])

#######The data for t1 is "i", "a", "t1-flattend"############################
t1_real = np.genfromtxt(("\t".join(i) for i in csv.reader(open('shape2.dat'))), delimiter="\t")
t1_real = np.loadtxt("t1_real.dat")
t1_real = t1_real.reshape(i, a)
t1_imag = np.loadtxt("t1_imag.dat")
t1_imag = t1_imag.reshape(i, a)
t1 = t1_real + 1.0*1j*t1_imag

#print data[1,:]
#
#np.savetxt('shape4.dat', shape4.flatten(), fmt='%2.i' )
####4th-order Rosenbrock "Parallel exponential Rosenbrock methods, 
#Vu Thai Luana, Alexander Ostermannb"
#t0 = t0 - dt
mol = CCSD_Helper(psi4)
#mol.Rosenbrock(F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, 'restart')   
######4th-order Runge-Kutta   
#mol.Runge_Kutta_solver(F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, 'restart')    

