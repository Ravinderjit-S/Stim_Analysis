#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:35:09 2021

@author: ravinderjit
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle
import sys
sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr

# from anlffr.spectral import mtspecraw
# from anlffr.spectral import mtplv
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  
mseq_loc = 'mseqEEG_150_bits10_4096.mat'


file_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/' + mseq_loc
Mseq_dat = sio.loadmat(file_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/Mseq_tone_coherence/'
pickle_loc = data_loc + 'Pickle/'

subject = 'S211'

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
    
datapath =  os.path.join(data_loc, subject)
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=1000)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
ocular_projs = Projs
if subject == 'S211':
    ocular_projs = [Projs[0]]



data_eeg.add_proj(ocular_projs)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=blinks,show_options=True)

#%% Plot data
reject = None

epochs = [];
for ep in range(3):
    epochs.append(mne.Epochs(data_eeg, data_evnt, ep+1, tmin=-0.3, 
                                     tmax=mseq.size/4096+0.6,reject=reject, baseline=(-0.2, 0.)) )
    epochs[ep].average().plot(picks=[31])
info_obj = epochs[0].info


#%% Extract part of response when stim is on
ch_picks = np.arange(32)
remove_chs = []
ch_picks = np.delete(ch_picks,remove_chs)
fs = epochs[0].info['sfreq']
t = epochs[0].times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size + int(np.round(0.4*fs))
epdat = []
for ep in range(3):
    epdat.append(epochs[ep].get_data()[:,ch_picks,t1:t2].transpose(1,0,2))
t = t[t1:t2]
t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))

#%% Remove epochs with large deflections
Reject_Thresh=150-6
Tot_trials = np.zeros(3)
for ep in range(3):
    Peak2Peak = epdat[ep].max(axis=2) - epdat[ep].min(axis=2)
    mask_trials = np.all(Peak2Peak < Reject_Thresh,axis=0)
    print('rejected ' + str(epdat[ep].shape[1] - sum(mask_trials)) + ' trials due to P2P')
    epdat[ep] = epdat[ep][:,mask_trials,:]
    print('Total Trials Left: ' + str(epdat[ep].shape[1]))
    Tot_trials[ep] = epdat[ep].shape[1]

plt.figure()
plt.plot(Peak2Peak.T)

#%% Correlation Analysis

Ht = []
for ep in range(3):
    Ht.append(mseqXcorr(epdat[ep],mseq[0,:]))


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

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for ep in range(3):
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            axs[p1,p2].plot(t,Ht[ep][p1*sbp[1]+p2,:])
            axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
            # for n in range(m*num_nfs,num_nfs*(m+1)):
            #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
fig.suptitle('Ht ')
    
fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)      
for ep in range(3):
    for p1 in range(sbp2[0]):
        for p2 in range(sbp2[1]):
            axs[p1,p2].plot(t,Ht[ep][p1*sbp2[1]+p2+sbp[0]*sbp[1],:])
            axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
            # for n in range(m*num_nfs,num_nfs*(m+1)):
            #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
            
fig.suptitle('Ht ')    
    

#%% PCA decomposition of Ht

pca = PCA(n_components=2)
pca_sp = pca.fit_transform(Ht[2].T)
pca_coeff = pca.components_

fig,axs = plt.subplots(2,1)
axs[0].plot(t,pca_sp)

axs[1].plot(ch_picks,pca_coeff.T)

vmin = pca_coeff.mean() - 2 * pca_coeff.std()
vmax = pca_coeff.mean() + 2 * pca_coeff.std()
plt.figure()
mne.viz.plot_topomap(pca_coeff[0,:], mne.pick_info(info_obj, ch_picks),vmin=vmin,vmax=vmax)
plt.figure()
mne.viz.plot_topomap(pca_coeff[1,:], mne.pick_info(info_obj, ch_picks),vmin=vmin,vmax=vmax)

#%% ICA decomposition of Ht

# ica = FastICA(n_components=2)
# ica_sp = ica.fit_transform(Ht.T)
# ica_coeff = ica.whitening_

# fig,axs = plt.subplots(2,1)
# axs[0].plot(t,ica_sp)

# axs[1].plot(ch_picks,-ica_coeff.T)

# vmin = ica_coeff.mean() - 2 * ica_coeff.std()
# vmax = ica_coeff.mean() + 2 * ica_coeff.std()
# plt.figure()
# mne.viz.plot_topomap(-ica_coeff[0,:], mne.pick_info(epochs.info, ch_picks),vmin=vmin,vmax=vmax)
# plt.figure()
# mne.viz.plot_topomap(-ica_coeff[1,:], mne.pick_info(epochs.info, ch_picks),vmin=vmin,vmax=vmax)

#%% save data
with open(os.path.join(pickle_loc,subject+'_AMmseq10bit_Active.pickle'),'wb') as file:
    pickle.dump([t, Tot_trials, Ht, info_obj, ch_picks],file)
