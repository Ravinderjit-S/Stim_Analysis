#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:44:40 2021

@author: ravinderjit
"""


import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt

#%% Subjects

fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/')

Subjects = ['S207', 'S228','S236','S238','S239','S246','S247','S250']

dataPassive_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
picklePassive_loc = dataPassive_loc + 'Pickles/'

#%% Load Data

A_Tot_trials_pass = []
A_Ht_pass = []
A_Htnf_pass = []
A_info_obj_pass = []
A_ch_picks_pass = []

A_Ht_epochs_pass = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials_pass.append(Tot_trials)
    A_Ht_pass.append(Ht)
    A_Htnf_pass.append(Htnf)
    A_info_obj_pass.append(info_obj)
    A_ch_picks_pass.append(ch_picks)
    
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
    
    A_Ht_epochs_pass.append(Ht_epochs)

print('Done loading passive ...')


#%% Average

Cz_sub = []

for sub in range(len(Subjects)):
    Cz_sub.append(A_Ht_pass[sub][-1,:])

Cz_sub = np.array(Cz_sub)
Cz_mean = Cz_sub.mean(axis=0)
Cz_sem = Cz_sub.std(axis=0) / np.sqrt(Cz_sub.shape[0])

plt.figure()
plt.plot(t,Cz_mean)
plt.fill_between(t,Cz_mean-Cz_sem,Cz_mean+Cz_sem)
plt.xlim([-0.05,0.5])

#%% Avg Hf
fs = 4096.0
t_0 = np.where(t>=0)[0][0]
t_1 = np.where(t>=0.5)[0][0]

Cz_hf = np.fft.fft(Cz_sub[:,t_0:t_1],axis=1)
f = np.fft.fftfreq(Cz_hf.shape[1],d=1/fs)
phase = np.unwrap(np.angle(Cz_hf),axis=1)


Cz_hf = np.abs(Cz_hf[:,f>0])
phase = phase[:,f>0]
f = f[f>0]

Cz_hf_sem = Cz_hf.std(axis=0) / np.sqrt(Cz_hf.shape[0])

phase_mn = phase.mean(axis=0)
phase_sem = phase.std(axis=0)

fig,ax = plt.subplots(2,1)

ax[0].plot(t,Cz_mean,color='k')
ax[0].fill_between(t,Cz_mean-Cz_sem,Cz_mean+Cz_sem,color='k',alpha=0.5)
ax[0].set_xlim([-0.05,0.5])

ax[1].plot(f,Cz_hf.mean(axis=0),color='k')
ax[1].fill_between(f,Cz_hf.mean(axis=0)-Cz_hf_sem, Cz_hf.mean(axis=0) + Cz_hf_sem,alpha=0.5,color='k')
ax2 = ax[1].twinx()
ax2.plot(f,phase_mn,color='grey')
ax2.fill_between(f,phase_mn-phase_sem,phase_mn+phase_sem,color='grey',alpha=0.5)
ax[1].set_xlim([0,100])
plt.savefig(os.path.join(fig_path,'ModTRF_avg.svg'),format='svg')
plt.savefig(os.path.join(fig_path,'ModTRF_avg.eps'),format='eps')










