#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import curie_weiss_periodic as cp
from numpy.linalg import eigh
from scipy.sparse import dia_matrix
from math import *
import matplotlib.pyplot as plt
import time as tm

#At Freezing Resonance
#besj_root = 8.65372791291101 	

fsize = 16

def get_jmat_pbc(lsize, beta):
    N = lsize
    J = dia_matrix((N, N))
    mid_diag = np.floor(N/2).astype(int)
    for i in np.arange(1,mid_diag+1):
        elem = pow(i, -beta)
        J.setdiag(elem, k=i)
        J.setdiag(elem, k=-i)
    for i in np.arange(mid_diag+1, N):
        elem = pow(N-i, -beta)
        J.setdiag(elem, k=i)
        J.setdiag(elem, k=-i)
    return J.toarray()

N = 9
t = np.linspace(0.0,2.0,5000)
beta = 0.0
b = beta
state = np.ones(2**N)/np.sqrt(2**N)
#state = np.zeros(2**N)
#state[0]=1.0
start = tm.time()
a = 25.0
hdc = 0.1
hz = -1.0
kac_norm = 8.0
f = 25.0
#Kac norm
#mid = np.floor(N/2).astype(int)
#sum = np.sum(1/(pow(np.arange(1, mid+1),b).astype(float)))
#kac_norm = 2.0 * sum
#f = (2 * a * kac_norm)/fr 

#Long Range hopping matrix
J = get_jmat_pbc(N, b)

filename1 = "lat_" + str(N) + "_f_" + str(f) + "_amp_" + str(a) + "_hdc_"+str(hdc)+"_hz_"+str(hz)+".txt"
p = cp.ParamData(hopmat = J/kac_norm, lattice_size=N, omega=f, times=t, hz=hz, jx=-1.0, jz=0.0, amp=a, hdc=hdc,norm=kac_norm)
h = cp.Hamiltonian(p)
out = cp.run_dyn(p, initstate=state) #Initial condition is t=0 Hamiltonian ground state

np.savetxt(filename1,np.vstack((np.real(out["t"]), np.real(out["sx"]), np.real(out["sz"]))).T)
time = out["t"]
szvalue = out["sz"]
sxvalue = out["sx"]
#plt.figure(dpi = 200)
plt.plot(time,szvalue,label='sz exact')
plt.plot(time,sxvalue,label='sx exact')
plt.title("l9_hz_-1_a_25_hdc_p1_f_25")
plt.grid()
plt.ylim(-1,1)
plt.legend()
plt.xlabel("time")
plt.ylabel("sz")
#plt.savefig("output.jpeg")
plt.show()
duration = tm.time() - start
print("time taken in second =",duration)
