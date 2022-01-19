#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:12:16 2022

@author: ravinderjit

make figues to show what happens to mod-TRF under attention

"""


import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
import scipy.io as sio
from scipy.signal import find_peaks

import sys
sys.path.append(os.path.abspath('../ACRanalysis/'))
import ACR_helperFuncs

sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr

#%% Load mseq
mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)
        
#%% Subjects

Subjects = ['S207', 'S228','S236','S238','S239','S246','S247','S250']
Subjects_sd = ['S207', 'S228', 'S236', 'S238', 'S239', 'S250'] #Leaving out S211 for now


dataPassive_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
picklePassive_loc = dataPassive_loc + 'Pickles/'

dataCount_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/'
pickleCount_loc = dataCount_loc + 'Pickles/'

pickle_loc_sd = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active_harder/Pickles/'


#%% Load Passive Data

A_Tot_trials_pass = []
A_Ht_pass = []
A_Htnf_pass = []
A_info_obj_pass = []
A_ch_picks_pass = []

A_Ht_epochs_pass = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    if subject == 'S250':
        subject = 'S250_visit2'
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

#%% Load Counting Data

A_Tot_trials_count = []
A_Ht_count = []
A_Htnf_count = []
A_info_obj_count = []
A_ch_picks_count = []

A_Ht_epochs_count = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials_count.append(Tot_trials)
    A_Ht_count.append(Ht)
    A_Htnf_count.append(Htnf)
    A_info_obj_count.append(info_obj)
    A_ch_picks_count.append(ch_picks)
    
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    A_Ht_epochs_count.append(Ht_epochs)
    
print('Done Loading Counting data')

#%% Load Shift Detect Data

Subjects_sd = ['S207', 'S228', 'S236', 'S238', 'S239', 'S250'] #Leaving out S211 for now

A_Tot_trials_sd = []
A_Ht_sd = []
A_info_obj_sd = []
A_ch_picks_sd = []

A_Ht_epochs_sd = []

for sub in range(len(Subjects_sd)):
    subject = Subjects_sd[sub]
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder.pickle'),'rb') as file:
        [t, Tot_trials, Ht, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_sd.append(Tot_trials)
    A_Ht_sd.append(Ht)
    A_info_obj_sd.append(info_obj)
    A_ch_picks_sd.append(ch_picks)
    
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    A_Ht_epochs_sd.append(Ht_epochs)
    
print('Done loading shift detect data')

#%% Plot Ch. Cz
    
fig,ax = plt.subplots(nrows=2,ncols=4,sharex=True)
ax = np.reshape(ax,8)

t_0 = np.where(t_epochs>=0)[0][0]

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    
    ch = 31
    
    if sub !=4:
        ax[sub].axes.yaxis.set_visible(False)
        
    if sub < len(Subjects)/2:
        ax[sub].axes.xaxis.set_visible(False)
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_epochs_pass[sub][ch_pass_ind,:,:]
    cz_count = A_Ht_epochs_count[sub][ch_count_ind,:,:]
    
    cz_pass_sem = cz_pass.std(axis=0) / np.sqrt(cz_pass.shape[0])
    cz_count_sem = cz_count.std(axis=0) / np.sqrt(cz_count.shape[0])
    
    cz_pass = cz_pass.mean(axis=0)
    cz_count = cz_count.mean(axis=0)
    
    cz_pass = cz_pass - cz_pass[t_0] #make time 0 have value of 0
    cz_count = cz_count - cz_count[t_0] #make time 0 have value of 0
    
    ax[sub].plot(t_epochs,cz_pass, label='Passive', color='k',linewidth=2)
    ax[sub].fill_between(t_epochs,cz_pass-cz_pass_sem, cz_pass+cz_pass_sem,color='k',alpha=0.5)
    
    ax[sub].plot(t_epochs,cz_count, label='Count', color='tab:blue')
    ax[sub].fill_between(t_epochs,cz_count-cz_count_sem, cz_count+cz_count_sem,color='tab:blue',alpha=0.5)
    
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_epochs_sd[sub_sd][ch_sd_ind,:,:]
        
        cz_sd_sem = cz_sd.std(axis=0) / np.sqrt(cz_sd.shape[0])
        cz_sd = cz_sd.mean(axis=0)
        
        cz_sd = cz_sd - cz_sd[t_0]
        
        ax[sub].plot(t_epochs,cz_sd, label='Shift Detect', color='tab:orange')
        ax[sub].fill_between(t_epochs,cz_sd - cz_sd_sem, cz_sd + cz_sd_sem, color='tab:orange',alpha=0.5)
    
    ax[sub].set_title('S' + str(sub+1))
    ax[sub].set_xlim([-0.010,0.05]) 
    #ax[sub].set_xticks([0,0.050,0.1])
    #ax[sub].set_xticks([0,0.2,0.4])
    
ax[1].legend(fontsize=9)
ax[4].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[4].set_xlabel('Time (sec)')
ax[4].set_ylabel('Amplitude')




