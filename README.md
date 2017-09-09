# python_code_to_check_C

This is a Time-dependent Coupled Cluster (CCSD or CC2) calculator. 
The converged CCSD and CC2 values have been checked against psi4. 
All individual T1, T2, L1, and L2 intermediate terms for CCSD and CC2 have been checked against code pulled psi4. 
The Fourier transform of the time-dependent simulation has peaks that correspond to an 
EOM-CC2 oscillator_strength calculation. 

## The format to run the simulation is:

```
python H2O.py <time_to_run>
```

where <time_to_run> is the time-length in seconds that the simulation will run for. I recommend using

```
python H2O.py <time_to_run> > <output.dat>
```

where <output.dat> is a file, that contains the psi4-output and also a copy of the time-dependent dipole moment.

## Files output from the simulation are:

H2O.csv          -The time-evolution of the dipole moment.

Parameters.dat   -The parameters needed for the next simulation to start from the end of last one.

timing.csv       -The recorded time to evaluate the right-hand-side of T1, T2, L1, and L2 (including cchbar).

Output files used to restart the simultion from the last point are:

F_real.dat

F_imag.dat

t1_real.dat

t1_imag.dat

t2_real.dat

t2_imag.dat

lam1_real.dat

lam1_imag.dat

lam2_real.dat

lam2_imag.dat

## Parameters:

The Electric Field Function is defined inside the Runge_Kutta defined function. 

w0 - Frequency of the Electric Field.

A  - Amplitude of the Electric Field.

dt - Time-step of the numeric solver.

t0 - Start-time of the simulation.

precs - The accuracy of the numerical solver.

See Restart_H2O.py for an example restart of the simulation. The form of the saved T1, T2, L1, and L2 was done so they could be compared to the C++ plugin.
Restart_H2O.py imports the saved parameters in Parameters.dat and F,T1,T2,L1,L2 to restart the next simulation for the length of time.
The Restart file requires the molecular input so that the numerical calculation can get the two electron integrals directly from psi4 rather than a saved input file.

## Links:
* [C++ Time-dependent CCSD/CC2 code](https://github.com/rachelglenn/tdccsd)

