import cmath
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift    

timing = pd.read_csv('timing.csv')
t = np.asarray(timing['total'].tolist())
#print("the total time is: ", sum(t)/3600.0/24.0, "days")
data = pd.read_csv('H2O.csv')
param = pd.read_csv('Parameters.csv')
yr = np.asarray(data['mu_real'].tolist())
yi = np.asarray(data['mu_imag'].tolist())
time = np.asarray(data['time'].tolist())

N = len(time)
dt = time[1]-time[0]

Yr = np.fft.fft(yr)
Yr = np.fft.fftshift(Yr)
Yi = np.fft.fft(yi)
Yi = np.fft.fftshift(Yi)
xaxis1 = range(-N/2,N/2,1)
xaxis = list(map(lambda x: x*1.0/(N*dt)*(2*np.pi)*27.211, xaxis1))
f, axarr = plt.subplots(3)#, sharex=True)
axarr[0].plot(time, yr,'.', markersize=1,color='black')
axarr[0].set_xlabel('time (a.u.)')
axarr[0].set_ylabel('Re[$\mu$] (a.u.)')
axarr[0].set_title('Dipole moment')
axarr[1].set_xlabel('time (a.u.)')
axarr[1].set_ylabel(' Im[$\mu$](a.u.)')
axarr[1].plot(time, yi,'.', markersize=1,color='black')
axarr[2].set_xlabel('Energy (eV)')
axarr[2].set_ylabel('Fourier Transform \n of Im[$\mu$]')
f.tight_layout()
axarr[0].set_xlim([time[0],time[-1]])
axarr[1].set_xlim([time[0],time[-1]])
axarr[0].set_ylim([min(yr)*1.005, max(yr)*0.999])
axarr[1].set_ylim([min(yi)*1.2, max(yi)*1.2])
axarr[2].set_xlim([0, 30])
l1 = axarr[2].plot(xaxis[N/2+2:],np.real(Yi[N/2+2:])/max(abs(Yi[N/2+2:])),marker='.',label='F$\{$Re[$\mu$]$\}$ ')
#l1 = axarr[2].scatter(xaxis[N/2+2:],abs(Yi[N/2+2:])/max(abs(Yi[N/2+2:])),label='F$\{$Re[$\mu$]$\}$ ',s=1)
l2 = axarr[2].plot(xaxis[N/2+2:],np.real(Yr[N/2+2:])/max(abs(Yr[N/2+2:])),marker='.',label='F$\{$Im[$\mu$]$\}$')
#l2 = axarr[2].scatter(xaxis[N/2+2:],6e-9*abs(Yr[N/2+2:])/max(abs(Yr[N/2+2:]))*2.68e8,label='F$\{$Im[$\mu$]$\}$',s=1)
axarr[2].legend(loc='upper right')
plt.show()
