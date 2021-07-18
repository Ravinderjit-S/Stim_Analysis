#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 19:07:52 2021

@author: ravinderjit
"""

import os
import pickle
import numpy as np
import scipy as sp
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
import scipy.io as sio


mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S207', 'S228','S236','S238','S239','S246','S247','S250']

fs = 4096

A_Tot_trials = []
A_Ht = []
A_Htnf = []
A_info_obj = []
A_ch_picks = []

#%% Load Data

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    A_Htnf.append(Htnf)
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
#%% Get average Ht
perCh = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh[ch,0] += np.sum(A_ch_picks[s]==ch)

Avg_Ht = np.zeros([32,t.size])
for s in range(len(Subjects)):
    Avg_Ht[A_ch_picks[s],:] += A_Ht[s]

Avg_Ht = Avg_Ht / perCh



#%% Plot time domain Ht

num_nf = len(A_Htnf[0])

sbp = [4,4]
sbp2 = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for sub in range(len(A_Ht) + 1):
    if sub == len(A_Ht):
        Ht_1 = Avg_Ht
        ch_picks_s = np.arange(32)
    else:
        Ht_1 = A_Ht[sub]
        ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1] + p2
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                if sub == len(A_Ht):
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:],color='k',linewidth=2)
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                else:
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                    
plt.legend(Subjects)
                
    
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for sub in range(len(A_Ht)+1):
    if sub == len(A_Ht):
        Ht_1 = Avg_Ht
        ch_picks_s = np.arange(32)
    else:
        Ht_1 = A_Ht[sub]
        ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                if sub == len(A_Ht):
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:],color='k',linewidth=2)
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                else:
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                
                   
#%% PCA on t_cuts

t_cuts = [.015,.040,.125,.500]
t_cuts = [.015,0.500]


                                
    
    









