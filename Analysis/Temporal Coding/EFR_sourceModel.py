#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 10:30:48 2021

@author: ravinderjit
"""

from scipy.signal.windows import gaussian
from scipy.signal import freqz
import matplotlib.pyplot as plt
import numpy as np


#%% Model each source as simple gaussian

latency = [.0066, .023, .046, .092, .181]
latency2 = [.0066, .023, .046, .080, .192]
width = np.array([.014, .020, .040, .057, .126])
width2 = np.array([.014, .020, .035, .050, .158])
stds = width / 4 #SD of the sources in seconds

fs= 4096
source_mods = []
for source in range(stds.size):
    gauss_source = gaussian(np.round(width[source]*fs),np.round(stds[source]*fs))
    source_mods.append(gauss_source - np.min(gauss_source))
    
plt.figure()
for source in range(stds.size):
    plt.plot(source_mods[source])
    


cur_lat = 0
lat_mod = np.zeros(len(source_mods))
for source in range(len(source_mods)):
    lat_mod[source] = cur_lat + source_mods[source].size/2 /fs
    cur_lat += source_mods[source].size/fs
    
    
#%% Construct full model with desired latencies and weights
    
latency_pad = [0,0,0,0 , 0 ]
weights = [1 , 0.5,1, 1, 1]
weights2 = [0.66, .44, 0.71, 0.87, 1]
weights2 = [1, 1.6, 0.3, 0.8, 0.42]

full_mod = np.array([])
full_mod2 = np.array([])
for source in range(len(source_mods)):
    
    if source == 2:
        continue
    
    if (source == 1): # do mixed source

        s2 = np.concatenate((np.zeros( int(np.round( (np.sum(width[:2]) + width[2]/2 - latency[2]) * fs)) ), source_mods[2]  ))
        s1 = np.concatenate((source_mods[1], np.zeros(s2.size-source_mods[1].size)))
        mixed_source = weights[1]*s1 + weights[2]*s2
        mixed_source2 = weights2[1]*s1 + weights2[2]*s2
        
        full_mod = np.append(full_mod, mixed_source)
        full_mod2 = np.append(full_mod2, mixed_source2)
       
        plt.figure()
        plt.plot(s1)
        plt.plot(s2)
        
        
        
    else:
        full_mod = np.append(full_mod,np.min(source_mods[source])*np.ones(int(np.round(latency_pad[source]*fs))))
        full_mod = np.append(full_mod, weights[source] * source_mods[source])
        
        full_mod2 = np.append(full_mod2,np.min(source_mods[source])*np.ones(int(np.round(latency_pad[source]*fs))))
        full_mod2 = np.append(full_mod2, weights2[source] * source_mods[source])

t_fm = np.linspace(0,full_mod.size/fs,full_mod.size)

plt.figure()
plt.plot(t_fm,full_mod)
plt.plot(t_fm,full_mod2)

[w,h] = freqz(b= full_mod - full_mod.mean() ,a=1,worN=np.arange(0,fs/2,2),fs=fs)
[w2,h2] =  freqz(b= full_mod2 - full_mod2.mean() ,a=1,worN=np.arange(0,fs/2,2),fs=fs)

plt.figure()
plt.plot(w,np.abs(h))
plt.xlabel('Frequency (Hz)')
plt.plot(w2,np.abs(h2))
plt.xlim([0,150])





