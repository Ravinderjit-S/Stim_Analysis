#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:04:01 2020

@author: ravinderjit
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import spectralAnalysis as sa
import scipy.io as sio
from anlffr.spectral import mtplv


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  
#data_loc = '/media/ravinderjit/Storage2/EEGdata'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/CMR_active/CMRactive_10Hz'
subject = 'SVarsha'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']

datapath = os.path.join(data_loc,subject)

data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=100)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0, h_freq=10)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = Projs[0]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=blinks,show_options=True)

#%% Plot 
labels = ['40 codev','40 comod']

tmin = -0.5
tmax = 4.5
reject = dict(eeg=150e-6)
baseline = (-0.2,0)

epochs_1 = mne.Epochs(data_eeg,data_evnt,[1],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_2 = mne.Epochs(data_eeg,data_evnt,[2],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)

evkd_1 = epochs_1.average()
evkd_2 = epochs_2.average()

picks = [4,30,31,7,22,8,21]
picks = 30
evkd_1.plot(picks=picks,titles=labels[0])
evkd_2.plot(picks=picks,titles=labels[1])



#%% spectral analysievkd_1.plot(picks=picks,titles='4 coh 0')
fs = evkd_1.info['sfreq']

t = epochs_1.times
t1 = np.where(t>=0)[0][0]
t2 = np.where(t>=4.0)[0][0]
t = t[t1:t2]

dat_epochs_1 = epochs_1.get_data()
dat_epochs_2 = epochs_2.get_data()

dat_epochs_1 = dat_epochs_1[:,0:32,t1:t2].transpose(1,0,2)
dat_epochs_2 = dat_epochs_2[:,0:32,t1:t2].transpose(1,0,2)

trials = np.min([dat_epochs_1.shape[1],dat_epochs_2.shape[1]])

dat_epochs_1 = dat_epochs_1[:,:trials,:]
dat_epochs_2 = dat_epochs_2[:,:trials,:]

params = dict()
params['Fs'] = fs
params['tapers'] = [2,2*2-1]
params['fpass'] = [1,300]
params['itc'] = 0


plvtap_1, f = mtplv(dat_epochs_1,params)
plvtap_2, f = mtplv(dat_epochs_2,params)

sbp = [4,4]
sbp2 = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(f,plvtap_2[p1*sbp[1]+p2,:])
        axs[p1,p2].plot(f,plvtap_1[p1*sbp[1]+p2,:])
        axs[p1,p2].set_xlim((35,45))
        axs[p1,p2].set_title(p1*sbp[1]+p2)    

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(f,plvtap_2[p1*sbp2[1]+p2+sbp[0]*sbp[1],:])
        axs[p1,p2].plot(f,plvtap_1[p1*sbp2[1]+p2+sbp[0]*sbp[1],:])
        axs[p1,p2].set_xlim((35,45))
        axs[p1,p2].set_title(p1*sbp2[1]+p2+sbp[0]*sbp[1])    



fig, ax = plt.subplots(figsize=(5.5,5))
ax.plot(f,plvtap_2.T,label='CORR',linewidth=2,color='b')
ax.plot(f,plvtap_1.T,label='ACORR',linewidth=2,color='r')

np.max(plvtap_2[:,150:167],axis=1)

fig, ax = plt.subplots(figsize=(5.5,5))
fontsize=15
ax.plot(f,plvtap_2[13,:],label='CORR',linewidth=2)
ax.plot(f,plvtap_1[13,:],label='ACORR',linewidth=2)
ax.legend(fontsize=fontsize)
plt.xlabel('Frequency (Hz)',fontsize=fontsize,fontweight='bold')
#plt.ylabel('PLV',fontsize=fontsize,fontweight='bold')
plt.xlim((35,45))
plt.xticks([35,40,45],fontsize=fontsize)
plt.yticks([0,0.04,0.08],fontsize=fontsize)


fig, ax = plt.subplots(figsize=(5.5,5))
fontsize=15
ax.plot(f,plvtap_2.T,label='CORR',linewidth=2,color='b')
ax.plot(f,plvtap_1.T,label='ACORR',linewidth=2,color='r')

plt.figure()
plt.plot(f,plvtap_1.T)
plt.title(labels[0])

plt.figure()
plt.plot(f,plvtap_2.T)
plt.title(labels[1])




