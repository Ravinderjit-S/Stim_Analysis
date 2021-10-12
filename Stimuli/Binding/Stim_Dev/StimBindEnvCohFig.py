#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 15:25:24 2021

@author: ravinderjit
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import rcParams

data = sio.loadmat('Binding20tonesExample.mat',squeeze_me=True)

fs = data['fs']
stimABAB = data['stimABAB']
stimABAB_2 = data['stimABAB_2']
envs_1 = data['envs_1']
envs_2 = data['envs_2']
corr_inds = data['Corr_inds']


fig, axs = plt.subplots(2,2)
fig.set_size_inches(11,8)
rcParams.update({'font.size': 15})

t = np.arange(0,stimABAB.size/fs,1/fs)
t_mask = (t>=1) & (t <=2.01)

axs[0,0].plot(t,stimABAB, color='k')
axs[0,0].plot(t[t_mask],envs_1[t_mask,19]*6.5,linestyle='dashed',color='tab:blue')
axs[0,0].set_xlim([0.75,2.25])
#axs[0,0].set_xlabel('Time (s)')
axs[0,0].set_ylabel('Stimulus Amplitdue')
axs[0,0].set_xticks([1,1.5,2])
axs[0,0].set_xticklabels(['0','0.5','1'])
axs[0,0].set_yticks([-4,4])
axs[0,0].legend(['Signal','Coherent Envelope'],fontsize=9,loc=2)


axs[0,1].plot(t,stimABAB_2, color='k')
axs[0,1].plot(t[t_mask],envs_2[t_mask,19]*6.5,linestyle='dashed',color='tab:blue')
axs[0,1].set_xlim([0.75,2.25])
axs[0,1].set_xticks([1,1.5,2])
axs[0,1].set_xticklabels(['0','0.5','1'])
axs[0,1].set_yticks([-4,4])
#axs[0,1].set_xlabel('Time (ms)')

n_seg = int(np.round(0.050*fs))

f,t,Sxx1 = signal.spectrogram(stimABAB,fs,nperseg= n_seg, noverlap = int(np.round(0.9*n_seg)))
f,t,Sxx2 = signal.spectrogram(stimABAB_2,fs,nperseg= n_seg, noverlap = int(np.round(0.9*n_seg)))

Sxx1 = 10*np.log10(Sxx1)
Sxx2 = 10*np.log10(Sxx2)

axs[1,0].pcolormesh(t,f[1:],Sxx1[1:,:], vmin= -40, vmax = -25 ,rasterized = True)
axs[1,0].set_xlim([0.75,2.25])
axs[1,0].set_ylim([150,9000])
axs[1,0].set_yscale('log')
axs[1,0].set_yticks([500, 2000, 8000])
axs[1,0].set_yticklabels(['0.5','2','8'])
axs[1,0].set_xticks([1,1.5,2])
axs[1,0].set_xticklabels(['0','0.5','1'])
axs[1,0].set_xlabel('Time (s)')
axs[1,0].set_ylabel('Frequency (kHz)')

axs[1,1].pcolormesh(t,f[1:],Sxx2[1:,:], vmin= -40, vmax = -25, rasterized= True )
axs[1,1].set_xlim([0.75,2.25])
axs[1,1].set_ylim([150,8500])
axs[1,1].set_yscale('log')
axs[1,1].set_yticks([500, 2000, 9000])
axs[1,1].set_yticklabels(['0.5','2','8'])
axs[1,1].set_xticks([1,1.5,2])
axs[1,1].set_xticklabels(['0','0.5','1'])
axs[1,1].set_xlabel('Time (s)')

plt.savefig( 'BindingEnvCohFig.svg' , format='svg')




