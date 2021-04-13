#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:52:24 2021

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



nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];


pickle_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/TMTF/Pickles/'
Subjects = ['Hari 1', 'Hari 2','E002_Visit_1','E002_Visit_2','E002_Visit_3',
            'E004_Visit_1','E004_Visit_2','E004_Visit_3']

A_Tot_trials = []
A_Ht =[]
A_Htnf =[]

A_info_obj = []
A_ch_picks = []



pickle_loc_2 = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits2/Pickles/'
m_bits = [7, 10]
with open(os.path.join(pickle_loc_2,'S211_'+'AMmseqbits2_16384.pickle'),'rb') as file:
    [tdat, Tot_trials, Ht_200, Htnf,
     info_obj, ch_picks] = pickle.load(file)


for sub in range(len(Subjects)):
    subject = Subjects[sub]
    
    with open(os.path.join(pickle_loc,subject+'_TMTF.pickle'),'rb') as file:
       [t, Tot_trials, Ht, info_obj, ch_picks] = pickle.load(file)
       
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
#%% average together two responses. They were played with opposite polarity
    #but can average them together here. 
for sub in range(len(A_Ht)):
    avg = np.zeros(A_Ht[sub][0].shape)
    for rep in range(len(A_Ht[sub])):
        avg += A_Ht[sub][rep]
    avg = avg / len(A_Ht[sub])
    A_Ht[sub] = avg
    
   
#%% Plot time domain Ht vs mseq_150 data
    
sbp = [4,4]
sbp2 = [4,4]

t_1 = np.where(t>=0)[0][0]
t_2 = np.where(t>=0.1)[0][0]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for sub in range(len(Subjects)):
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1]+p2
            ch_ind=cur_ch
            axs[p1,p2].plot(t,A_Ht[sub][ch_ind,:]/np.max(np.abs(A_Ht[sub][ch_ind,t_1:t_2])))
            axs[p1,p2].set_title(ch_picks[ch_ind])
            axs[p1,p2].set_xlim([0,0.1])
            if sub ==len(Subjects)-1:
                t_dat1 = np.where(tdat[1]>=0)[0][0]
                t_dat2 = np.where(tdat[1]>=0.1)[0][0]
                #plot 10 bit 150 max last
                axs[p1,p2].plot(tdat[1],Ht_200[1][ch_ind,:]/np.max(np.abs(Ht_200[1][ch_ind,t_dat1:t_dat2])))
                
            
            
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for sub in range(len(Subjects)):
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            cur_ch = p1*sbp2[1]+p2 + sbp[0]*sbp[1]
            ch_ind=cur_ch
            axs[p1,p2].plot(t,A_Ht[sub][ch_ind,:]/np.max(np.abs(A_Ht[sub][ch_ind,t_1:t_2])))
            axs[p1,p2].set_title(ch_picks[ch_ind])
            axs[p1,p2].set_xlim([0,0.1])



#%% E2 1-3
sbp = [4,4]
sbp2 = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for sub in range(2,5):
    ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1]+p2
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                axs[p1,p2].plot(t,A_Ht[sub][ch_ind,:])
                axs[p1,p2].set_title(ch_picks_s[ch_ind])
                axs[p1,p2].set_xlim([0,0.1])

            
            
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for sub in range(2,5):
    ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            cur_ch = p1*sbp2[1]+p2 + sbp[0]*sbp[1]
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                axs[p1,p2].plot(t,A_Ht[sub][ch_ind,:])
                axs[p1,p2].set_title(ch_picks_s[ch_ind])
                axs[p1,p2].set_xlim([0,0.1])

#%% E4 1-3

sbp = [4,4]
sbp2 = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for sub in range(5,8):
    ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1]+p2
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                axs[p1,p2].plot(t,A_Ht[sub][ch_ind,:])
                axs[p1,p2].set_title(ch_picks_s[ch_ind])
                axs[p1,p2].set_xlim([0,0.1])

            
            
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for sub in range(5,8):
    ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            cur_ch = p1*sbp2[1]+p2 + sbp[0]*sbp[1]
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                axs[p1,p2].plot(t,A_Ht[sub][ch_ind,:])
                axs[p1,p2].set_title(ch_picks_s[ch_ind])
                axs[p1,p2].set_xlim([0,0.1])











   




