#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:57:43 2021

@author: ravinderjit
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import pickle
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs

import sys
sys.path.append(os.path.abspath('../../mseqAnalysis/'))
from mseqHelper import mseqXcorr

Subject = 'S211'
nchans = 34;


refchans = None


direct_IAC = '/media/ravinderjit/Data_Drive/Data/EEGdata/BinauralMseq_reCollect/StandardWay/'
direct_Mseq = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Mseq_4096fs_compensated.mat'
Mseq_mat = sio.loadmat(direct_Mseq)
Mseq = Mseq_mat['Mseq_sig'].T
Mseq = Mseq.astype(float)

exclude = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved



IAC_eeg,IAC_evnt = EEGconcatenateFolder(direct_IAC+Subject+'/',nchans,refchans,exclude)
IAC_eeg.filter(1,1000)

#%% blink removal
blinks_IAC = find_blinks(IAC_eeg, ch_name = ['A1'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
scalings = dict(eeg=40e-6)

blinkIAC_epochs = mne.Epochs(IAC_eeg,blinks_IAC,998,tmin=-0.25,tmax=0.25,proj=False,
                          baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs_IAC = compute_proj_epochs(blinkIAC_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')


eye_projsIAC = [Projs_IAC[0],Projs_IAC[1]]
IAC_eeg.add_proj(eye_projsIAC)

IAC_eeg.plot_projs_topomap()
IAC_eeg.plot(show_options=True)

#%% epoch

IAC_epochs = mne.Epochs(IAC_eeg,IAC_evnt,1,tmin=-0.5,tmax=14,proj=True,baseline=(-0.2, 0.),reject=None)
IAC_evoked = IAC_epochs.average()
IAC_evoked.plot(titles ='IACt_evoked')

#%% Extract epochs when stim is on
t = IAC_epochs.times
fs = IAC_epochs.info['sfreq']
t1 = np.where(t>=0)[0][0]
t2 = t1 + Mseq.size + int(np.round(0.4*fs))
t = t[t1:t2]
t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
ch_picks = np.arange(32)

IAC_ep = IAC_epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)

#%% Remove any epochs with large deflections

reject_thresh = 800e-6
    
Peak2Peak = IAC_ep.max(axis=2) - IAC_ep.min(axis=2)
mask_trials = np.all(Peak2Peak < reject_thresh,axis=0)
print('rejected ' + str(IAC_ep.shape[1] - sum(mask_trials)) + ' IAC trials due to P2P')
IAC_ep = IAC_ep[:,mask_trials,:]
print('Total IAC trials: ' + str(IAC_ep.shape[1]))

Tot_trials_IAC = IAC_ep.shape[1]

#%% Calculate Ht

IAC_Ht = mseqXcorr(IAC_ep,Mseq[:,0])


#%% Plot Ht
    
if ch_picks.size == 31:
    sbp = [5,3]
    sbp2 = [4,4]
elif ch_picks.size == 32:
    sbp = [4,4]
    sbp2 = [4,4]
elif ch_picks.size == 30:
    sbp = [5,3]
    sbp2 = [5,3]

    
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t,IAC_Ht[p1*sbp[1]+p2,:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
        
fig.suptitle('Ht IAC')


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        axs[p1,p2].plot(t,IAC_Ht[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
         
fig.suptitle('Ht IAC')

#%% Ht causal

t1 = np.where(t>=0)[0][0]
t2 = np.where(t>=0.5)[0][0]

Ht_causal = IAC_Ht[:,t1:t2]
t_causal = t[t1:t2]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t_causal,Ht_causal[p1*sbp[1]+p2,:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
        
fig.suptitle('Ht IAC')


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        axs[p1,p2].plot(t_causal,Ht_causal[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
         
fig.suptitle('Ht IAC')

#%% convolve and xcorr w/ mseq --- reconstruct Ht

y_conv = np.zeros([Ht_causal.shape[0],Ht_causal.shape[1]+Mseq.size-1])
for ch in range(Ht_causal.shape[0]):
    y_conv[ch,:] = np.convolve(Mseq[:,0], Ht_causal[ch,:],mode='full')
    
t_conv = np.arange(-Ht_causal.shape[1]/fs,Mseq.size/fs,1/fs)
t_conv = t_conv[1:]


fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t_conv,y_conv[p1*sbp[1]+p2,:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
        
fig.suptitle('Y_conv')


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        axs[p1,p2].plot(t_conv,y_conv[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
         
fig.suptitle('Y_conv')

# Xcorr with Mseq to recover TRF

#t1 = np.where(t_conv>=0)[0][0]


IAC_Ht_reconstruct = np.zeros([y_conv.shape[0],y_conv.shape[1]+Mseq.size-1])
for ch in range(y_conv.shape[0]):
    IAC_Ht_reconstruct[ch,:] = np.correlate(y_conv[ch,:],Mseq[:,0],mode='full')#[mseq[m].size-1:]


t_reconstruct = np.arange(-Mseq.size/fs,y_conv.shape[1]/fs,1/fs)
t_reconstruct = t_reconstruct[1:]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t_reconstruct,IAC_Ht_reconstruct[p1*sbp[1]+p2,:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
        
fig.suptitle('Ht reconstruct')


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        axs[p1,p2].plot(t_reconstruct,IAC_Ht_reconstruct[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
        # for n in range(num_nfs):
        #     axs[p1,p2].plot(t,IAC_Htnf[n][p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='grey',alpha=0.3)
         
fig.suptitle('Ht reconstruct')














