# -*- coding: utf-8 -*-
import sys
import os
import cmath
import numpy as np
sys.path.append('/home/rglenn/newriver/buildpython/pandas')
import pandas as pd
import matplotlib.pyplot as plt
import time

############Testing a approximated integrator (First-order) and the density matrix###########
###Setup parameters:
w10=w0 = 0.476347 #frequency of the oscillation and transition frequency
A = 0.05#the amplitude of the electric field    
t0 = 0.0 #the start time
tf = 200.0 #the stop time
dt =1.0 #time step
t=t0
##Electric field, it is in the z-direction for now

def Vt(t):
    return -A*np.sin(w0*t) #*mu[2]
p0 = 1
p1 = 0
#Build the rhs of the density matrix for p10,p11 
def phi1(z):
    return (cmath.exp(z)-1)/z 
def phi2(z):
    return (cmath.exp(z)-1-z)/np.power(z, 2)
def phi3(z):
    return -(2 - 2*cmath.exp(z) +2*z+np.power(z, 2))/(2*np.power(z, 3))
def phi4(z):
    return -(6 - 6*cmath.exp(z) + 6*z+3*z*z+np.power(z, 3))/(6*np.power(z, 4))

def b2(z):
    return 16*phi3(z) - 48*phi4(z)
def b3(z):
    return -2*phi3(z) + 12*phi4(z)        
 ##F(t):          
def F( t, p0, p1):
    return 1j*Vt(t)*(p1-(1-p1))-1j*w10*p0        
def g( t, p0, p1):
    return 1j*Vt(t)*(p1-(1-p1))  #-1j*w10*p0
def p11rhs( t, p0):
    return -1j*(Vt(t)*np.conj(p0)-p0*np.conj(Vt(t)))                                              

L= -1j*w10
c2=0.5
c3 =1.0   
def Un(t, p0, p1, c):
    return p0 + c*dt*phi1(c*dt*A)*F(t, p0, p1) 

def Dn(t, p0, p1, c):
    return g( t, Un(t, p0, p1, c), p1) - g( t, p0, p1)   

def ka(t, p0, p1):
    return p0 + dt*phi1(dt*L)*F( t, p0, p1) + dt * b2(c2*dt*A) * Dn(t, p0, p1, c2)\
    + dt * b3(c3*dt*A)*  Dn(t, p0, p1, c3)
     
def kb(t, p0, p1): 
    return p1 + dt*p11rhs( t, p0)                                                                                            
i=0
datas =  pd.DataFrame( columns = ( 'time', 'p10', 'p11', 'p01', 'p00')) 
while t < tf:
    p0min = p0
    p1min = p1
    dp0=  ka( t, p0, p1)
    dp1=  kb( t, p0, p1)
    t += dt
    p0 = dp0
    p1 = dp1
    #print p0, p1
    i +=1
    datas.loc[i] = [ t, p0, p1, np.conj(p0), 1-p1 ]                                               
    #print pf(t0 + dt/2.0, p + dt/2.0*pf(t0,p) )
i=0 
       
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(datas['time'].real, datas['p00'].real, datas['time'].real, datas['p11'].real)
axarr[0].set_title('Density Matrix-4th order Rosenbrock')
axarr[1].plot(datas['time'].real, datas['p01'].imag, datas['time'].real, datas['p10'].imag)
plt.show()



#exit()
















############Testing Rosenbrock–Euler Exponential integrator (First-order) and the density matrix###########

###Setup parameters:
t=t0
p0 = 1
p1 = 0
#Build the rhs of the density matrix for p10,p11 
def phi1(z):
    return (cmath.exp(z)-1)/z
        
            
def p10rhs( t, p0, p1):
    return 1j*Vt(t)*(p1-(1-p1))  #-1j*w10*p0
def p11rhs( t, p0):
    return -1j*(Vt(t)*np.conj(p0)-p0*np.conj(Vt(t)))                                              
L= -1j*w10
def ka(t, p0, p1):
    return cmath.exp(-dt*L)*p0 + dt*phi1(-dt*L)*p10rhs( t, p0, p1)      
def kb(t, p0, p1): 
    return p1 + dt*p11rhs( t, p0)                                                                                            
i=0
datas =  pd.DataFrame( columns = ( 'time', 'p10', 'p11', 'p01', 'p00')) 
while t < tf:
    p0min = p0
    p1min = p1
    dp0=  ka( t, p0, p1)
    dp1=  kb( t, p0, p1)
    t += dt
    p0 = dp0
    p1 = dp1
    #print p0, p1
    i +=1
    datas.loc[i] = [ t, p0, p1, np.conj(p0), 1-p1 ]                                               
    #print pf(t0 + dt/2.0, p + dt/2.0*pf(t0,p) )
i=0 
       
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(datas['time'].real, datas['p00'].real, datas['time'].real, datas['p11'].real)
axarr[0].set_title('Density Matrix-1st order Euler')
axarr[1].plot(datas['time'].real, datas['p01'].imag, datas['time'].real, datas['p10'].imag)
plt.show()


############Testing Trapezoidal Exponential integrator (2nd-order) and the density matrix###########

###Setup parameters:
t=t0
p0 = 1
p1 = 0
#Build the rhs of the density matrix for p10,p11 
def phi1(z):
    return (cmath.exp(z)-1)/z
def phi2(z):
    return (cmath.exp(z)-1-z)/z**2    
def b1(z):
    return phi1(z) -phi2(z)
def b2(z):
    return phi2(z)
               
def p10rhs( t, p0, p1):
    return 1j*Vt(t)*(p1-(1-p1)) 
def p11rhs( t, p0):
    return -1j*(Vt(t)*np.conj(p0)-p0*np.conj(Vt(t)))                                              
def L(p0):
    return -1j*w10
        
def ka(t, p0, p1):
    return cmath.exp(-dt*L(p0))*p0 + dt*b1(-dt*L(p0))*p10rhs( t, p0, p1) \
    + dt*b2(-dt*L(p0))*p10rhs( t+dt, p0, p1)       
def kb(t, p0, p1): 
    return p1 + dt*p11rhs( t, p0) #+ dt*p11rhs( t+dt, p0)                                                                                          
i=0
datas =  pd.DataFrame( columns = ( 'time', 'p10', 'p11', 'p01', 'p00')) 
while t < tf:
    p0min = p0
    p1min = p1
    dp0=  ka( t, p0, p1)
    dp1=  kb( t, p0, p1)
    t += dt
    p0 = dp0
    p1 = dp1
    #print p0, p1
    i +=1
    datas.loc[i] = [ t, p0, p1, np.conj(p0), 1-p1 ]                                               
       #print pf(t0 + dt/2.0, p + dt/2.0*pf(t0,p) )
i=0 
       
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(datas['time'].real, datas['p00'].real, datas['time'].real, datas['p11'].real)
axarr[0].set_title('Density Matrix-2nd order Trap. rule')
axarr[1].plot(datas['time'].real, datas['p01'].imag, datas['time'].real, datas['p10'].imag)
plt.show()














############Testing Runge-Kutta with the Density Matrix###########
###Setup parameters:
t=t0
p0 = 1
p1 = 0

def Vt(t):
    return -A*np.sin(w0*t) #*mu[2]
p0 = 1
p1 = 0
#Build the rhs of the density matrix for p10,p11 
def p10rhs( t, p0, p1):
    return -1j*w10*p0+1j*Vt(t)*(p1-(1-p1))
def p11rhs( t, p0):
    return -1j*(Vt(t)*np.conj(p0)-p0*np.conj(Vt(t)))                                              
def ka(t, p0, p1):
    k1 = p10rhs( t, p0, p1)
    k2 = p10rhs( t + dt/2.0, p0 + dt/2.0*k1, p1)   
    k3 = p10rhs( t + dt/2.0, p0 + dt/2.0*k2, p1)     
    k4 = p10rhs( t + dt, p0 + dt*k3, p1+ dt*k3)   
    return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)      
def kb(t, p0, p1):
    k1 = p11rhs( t, p0)
    k2 = p11rhs( t + dt/2.0, p0+ dt/2.0*k1)   
    k3 = p11rhs( t + dt/2.0, p0 + dt/2.0*k2)     
    k4 = p11rhs( t + dt, p0 + dt*k3)  
    return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)                                                                                             
i=0
datas =  pd.DataFrame( columns = ( 'time', 'p10', 'p11', 'p01', 'p00')) 
while t < tf:
    p0min = p0
    p1min = p1
    dp0=  ka( t, p0, p1)
    dp1=  kb( t, p0, p1)
    t += dt
    p0 = p0min + dp0
    p1 = p1min + dp1
    #print p0, p1
    i +=1
    datas.loc[i] = [ t, p0, p1, np.conj(p0), 1-p1 ]                                               
        #print pf(t0 + dt/2.0, p + dt/2.0*pf(t0,p) )
i=0 
        
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(datas['time'].real, datas['p00'].real, datas['time'].real, datas['p11'].real)
axarr[0].set_title('Density Matrix-Runge-Kutta')
axarr[1].plot(datas['time'].real, datas['p01'].imag, datas['time'].real, datas['p10'].imag)

plt.show()

