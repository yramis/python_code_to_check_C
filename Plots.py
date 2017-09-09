import sys
import os
sys.path.insert(0,'./..')
sys.path.append('/Users/silver/Desktop/workspace/python_RTDHF/ps411/myscf_pluginUHF')
#sys.path.append('$HOME/Desktop/workspace/psi411/psi4/objdir/stage/psi4-build/bin/psi4')
sys.path.append(os.environ['HOME']+'/Desktop/workspace/psi411/psi4/objdir/stage/usr/local/lib')
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift    
        
data = pd.read_csv('H2O.csv')
param = pd.read_csv('Parameters.csv')
# Number of samplepoints
N = len(data['mu_real'])
dt =  param['dt'][0] #time step

#print data['time'].tolist()


dt = param['dt'][0] #time step
dt = 0.0241888425*dt # convert a. u. to fs
time = 0.0241888425*np.asarray(data['time'].tolist())
#print time
t = time/ time[-1]

yr = np.asarray(data['mu_real'].tolist())
yi = np.asarray(data['mu_imag'].tolist())
N = len(t) 
k = np.arange(N)
T = N*dt
c = 33356.4095 #fs/cm
xaxis = k/T # two sides frequency range
xaxis = xaxis[range(N/2)]*2*np.pi*c


Yr = np.fft.fft(yr)/N
Yr = Yr[range(N/2)]
Yi = np.fft.fft(yi)/N
Yi = Yi[range(N/2)]

print N
#print '= {:.3e} m^3/min'.format(1235422574.0)
f, axarr = plt.subplots(3)#, sharex=True)
axarr[0].plot(data['time'].tolist(), data['mu_real'].tolist())
axarr[0].set_xlabel('time (fs)')
axarr[0].set_ylabel('Re[$\mu$] (arb. u.)')
axarr[0].set_title('Dipole moment')
axarr[1].set_xlabel('time (fs)')
axarr[1].set_ylabel(' Im[$\mu$](arb. u)')
axarr[1].plot(time, data['mu_imag'])
axarr[2].set_xlabel('wavenumbers (cm^{-1})')
axarr[2].set_ylabel('Fourier Transform \n of Im[$\mu$]')
axarr[2].set_xlim([0, 80895650])
#axarr[2].plot(xaxis,abs(Yr))
axarr[2].plot(xaxis,abs(Yi))

plt.show() 






exit()

Fs = 1500.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 50;   # frequency of the signal
y = np.sin(2*np.pi*ff*t)

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)]# one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
plt.show()


