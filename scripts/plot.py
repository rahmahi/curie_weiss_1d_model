#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 07:36:54 2020

@author: nilu
"""

import numpy as np
import matplotlib.pyplot as plt

d1 = np.loadtxt("8_time_sz_0.0_14.930917708with_dc_part")
d2 = np.loadtxt("8_time_sz_0.0_14.930917708witho_dc_part")

t1, t2, sz1, sz2 = [], [], [], []
for i in xrange(1000):
    t1.append(d1[i][1])
    t2.append(d2[i][1])
    sz1.append(d1[0][1])
    sz2.append(d2[0][1])
    
plt.subplot(211)
plt.title("sz vs time with and without dc field along with rwa")
plt.ylim(-1,1)
plt.ylabel("sz_wdc")
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
plt.plot(t1,sz1)
plt.subplot(212)
plt.ylim(-1,1)
plt.ylabel("sz_wodc")
plt.xlabel("time")
plt.plot(t2,sz2)
plt.show()