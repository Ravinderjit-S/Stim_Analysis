#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:19:53 2021

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
from scipy.signal import find_peaks

import sys
sys.path.append(os.path.abspath('../ACRanalysis/'))
from ACR_helperFuncs import ACR_sourceHf
from ACR_helperFuncs import ACR_model


passive_pickleLoc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/Pickles_full/'
active1_pickleLoc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/Pickles/'
active2_pickleLoc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active_harder/Pickles/'


A_t = []
A_Tot_trials = []
A_Ht = []
A_info_obj = []
A_ch_picks = []


with open(os.path.join(passive_pickleLoc,'S211_'+'AMmseqbits4.pickle'),'rb') as file:
    [tdat, Tot_trials, Ht, Htnf,
     info_obj, ch_picks] = pickle.load(file)
    
A_t.append(tdat[3])
A_Tot_trials.append(Tot_trials[3])
A_Ht.append(Ht[3])
A_info_obj.append(info_obj)
A_ch_picks.append(ch_picks) 
    
with open(os.path.join(active1_pickleLoc,'S211_'+'AMmseq10bits_Active.pickle'),'rb') as file:
    [tdat, Tot_trials, Ht, Htnf,
     info_obj, ch_picks] = pickle.load(file)
    
A_t.append(tdat)
A_Tot_trials.append(Tot_trials)
A_Ht.append(Ht)
A_info_obj.append(info_obj)
A_ch_picks.append(ch_picks)

with open(os.path.join(active2_pickleLoc,'S211_'+'AMmseq10bit_Active_harder.pickle'),'rb') as file:
    [tdat, Tot_trials, Ht, info_obj, ch_picks] = pickle.load(file)

A_t.append(tdat)
A_Tot_trials.append(Tot_trials)
A_Ht.append(Ht)
A_info_obj.append(info_obj)
A_ch_picks.append(ch_picks)


#%% Plot Ht

sbp = [4,4]
colors = ['k','tab:blue','tab:orange']

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for tng in range(len(A_Ht)):
    ch_picks = A_ch_picks[tng]
    Ht1 = A_Ht[tng]
    t = A_t[tng]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1]+p2
            if np.any(cur_ch == ch_picks):
                ch_ind = np.where(cur_ch==ch_picks)[0][0]
                axs[p1,p2].plot(t,Ht1[ch_ind,:],color=colors[tng])
                axs[p1,p2].set_title(ch_picks[ch_ind])    
                axs[p1,p2].set_xlim([0,0.5])

fig.suptitle('Ht ')
    
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for tng in range(len(A_Ht)):
    ch_picks = A_ch_picks[tng]
    Ht1 = A_Ht[tng]
    t = A_t[tng]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1]+p2+sbp[0]*sbp[1]
            if np.any(cur_ch == ch_picks):
                ch_ind = np.where(cur_ch==ch_picks)[0][0]
                axs[p1,p2].plot(t,Ht1[ch_ind,:],color=colors[tng])
                axs[p1,p2].set_title(ch_picks[ch_ind])   
                axs[p1,p2].set_xlim([0,0.5])

fig.suptitle('Ht ')


#%% PCA decomposition

t_cuts = [.016, .066,.250,0.500]

A_pca_sp = []
A_pca_coeffs = []
A_pca_expVar = []


pca = PCA(n_components=2)

for tng in range(len(A_Ht)):
    A_tc_pca_sp = []
    A_tc_pca_coeffs = []
    A_tc_pca_expVar = []
    
    t = A_t[tng]
    Ht = A_Ht[tng]
    A_tcut = []
    
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
            
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        
        pca_sp = pca.fit_transform(Ht[:,t_1:t_2].T)
        pca_coeff = pca.components_
        pca_expVar = pca.explained_variance_ratio_
        
        if pca_coeff[0,-1] < 0:
            pca_coeff = -pca_coeff
            pca_sp = -pca_sp
        
        A_tc_pca_sp.append(pca_sp)
        A_tc_pca_coeffs.append(pca_coeff)
        A_tc_pca_expVar.append(pca_expVar)
        A_tcut.append(t[t_1:t_2])
    
    A_pca_sp.append(A_tc_pca_sp)
    A_pca_coeffs.append(A_tc_pca_coeffs)
    A_pca_expVar.append(A_tc_pca_expVar)
    

plt.figure()
for tng in range(len(A_Ht)):
    for t_c in range(len(t_cuts)):
        plt.plot(A_tcut[t_c],A_pca_sp[tng][t_c][:,0],color=colors[tng])


plt.figure()
for tng in range(len(A_Ht)):
    for t_c in range(len(t_cuts)):
        plt.plot(A_tcut[t_c],A_pca_sp[tng][t_c][:,1],color=colors[tng])



#%% Hf
        
t1 = np.where(t>=0)[0][0]
t2 = np.where(t>=0.500)[0][0] 
ch_ind = np.where(A_ch_picks[0]==31)[0][0]
Cz_avg = (A_Ht[0][ch_ind,t1:t2]) - (A_Ht[0][ch_ind,t1:t2].mean())  
t_Cz = t[t1:t2]

[peaks, prop] = find_peaks(Cz_avg)
[peaks_neg, prop] = find_peaks(-Cz_avg)

plt.figure()
plt.plot(t_Cz,Cz_avg)
plt.scatter(t_Cz[peaks],Cz_avg[peaks],marker='x',color='r')
plt.scatter(t_Cz[peaks_neg],Cz_avg[peaks_neg],marker='x',color='b')

split_locs = [65,247,457,1104]
f1 = [40, 10, 8, 4]
f2 = [80, 25, 22, 10]
fs = 4096

[tpks, pks, pks_Hf, pks_w, pks_phase, 
pks_phaseLine, pks_phaseLineW, pks_gd]  = ACR_sourceHf(split_locs,Cz_avg,t[t1:t2],fs,f1,f2)

[w,h] = freqz(b=Cz_avg[:split_locs[-1]] - Cz_avg[:split_locs[-1]].mean() ,a=1,worN = np.arange(0,fs/2,4),fs=fs)


plt.figure()
for pk in range(len(tpks)):
    plt.plot(tpks[pk],pks[pk])
plt.xlabel('Time (sec)')
plt.title('S211')

plt.figure()
for pk in range(len(tpks)):
    plt.plot(pks_w[pk],np.abs(pks_Hf[pk]))
plt.plot(w,np.abs(h),color='grey',alpha=0.5)
plt.xlim([0,150])
plt.xlabel('Frequency')
plt.title('S211')
plt.legend([1,2,3,4])

plt.figure()
for pk in range(len(tpks)):
    plt.plot(pks_w[pk], pks_phase[pk])
    plt.plot(pks_phaseLineW[pk], pks_phaseLine[pk],color='k')
plt.xlim([0,150])
plt.xlabel('Frequency')
plt.ylabel('Phase')
plt.ylim([-20,10])
plt.legend([1,2,3,4])

#%% ACR model

latency =  [.0066, .023, .046, .092, .181]
width = np.array([.014, .020, .040, .057, .126])
weights = [1,0.5,1,1,1]
latency_pad = [0,0,0,0,0]

latency2 = [.0066, .023, .03, .092, .181]#[.0066, .023, .046, .080, .192]
width2 = np.array([.014, .020, .035, .080, .192])
weights2 = [1,1.5,2,1,1]
latency_pad2 = [0,0,0,0,0]

full_mod = ACR_model(latency,width, weights,latency_pad,fs)
full_mod2 = ACR_model(latency2,width2, weights2,latency_pad2,fs)

[w,h] = freqz(b= full_mod - full_mod.mean() ,a=1,worN=np.arange(0,fs/2,2),fs=fs)
[w2,h2] = freqz(b= full_mod2 - full_mod2.mean() ,a=1,worN=np.arange(0,fs/2,2),fs=fs)


t_fm = np.linspace(0,full_mod.size/fs,full_mod.size)
t_fm2 = np.linspace(0,full_mod2.size/fs,full_mod2.size)

fig,axs = plt.subplots(2,1)
axs[0].plot(t_fm,full_mod)
axs[0].plot(t_fm2,full_mod2)

axs[1].plot(w,np.abs(h))
axs[1].plot(w2,np.abs(h2))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_xlim([0,150])





