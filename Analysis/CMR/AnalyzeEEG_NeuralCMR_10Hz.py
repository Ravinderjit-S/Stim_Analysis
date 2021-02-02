#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:12:18 2021

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
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/CMR_10Hz/'
subject = 'S_Varsha'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']

datapath = os.path.join(data_loc,subject)

data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=300)

if subject == 'S_Varsha':
    data_eeg.info['bads'] = ['A24','A26']
    


#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0, h_freq=10)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = [Projs[0]]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=blinks,show_options=True)

#%%Epoch and Plot

tmin = -0.5
tmax = 4.5
reject = dict(eeg=200e-6)
baseline = (-0.2,0)

labels = ['40_dev', '40_comod', '223_dev', '223_comod']

epochs_1 = mne.Epochs(data_eeg,data_evnt,[1],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_2 = mne.Epochs(data_eeg,data_evnt,[2],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_3 = mne.Epochs(data_eeg,data_evnt,[3],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)
epochs_4 = mne.Epochs(data_eeg,data_evnt,[4],tmin=tmin,tmax=tmax,
                              baseline=baseline,reject=reject)



evkd_1 = epochs_1.average()
evkd_2 = epochs_2.average()
evkd_3 = epochs_3.average()
evkd_4 = epochs_4.average()

picks = 'A32'
evkd_1.plot(picks=picks,titles=labels[0])
evkd_2.plot(picks=picks,titles=labels[1])
evkd_3.plot(picks=picks,titles=labels[2])
evkd_4.plot(picks=picks,titles=labels[3])

#%% Spectral Analysis: Compute Phase Locking Value (PLV)

t = epochs_1.times
t1 = np.where(t>=0)[0][0]
t2 = np.where(t>=4.0)[0][0]
t = t[t1:t2]

fs = evkd_1.info['sfreq']

dat_epochs_1 = epochs_1.get_data()
dat_epochs_2 = epochs_2.get_data()
dat_epochs_3 = epochs_3.get_data()
dat_epochs_4 = epochs_4.get_data()

dat_epochs_1 = dat_epochs_1[0:279,0:32,t1:t2].transpose(1,0,2)
dat_epochs_2 = dat_epochs_2[0:279,0:32,t1:t2].transpose(1,0,2)
dat_epochs_3 = dat_epochs_3[0:279,0:32,t1:t2].transpose(1,0,2)
dat_epochs_4 = dat_epochs_4[0:279,0:32,t1:t2].transpose(1,0,2)

params = dict()
params['Fs'] = fs
params['tapers'] = [2,2*2-1]
params['fpass'] = [1,300]
params['itc'] = 0

plvtap_1, f = mtplv(dat_epochs_1,params)
plvtap_2, f = mtplv(dat_epochs_2,params)
plvtap_3, f = mtplv(dat_epochs_3,params)
plvtap_4, f = mtplv(dat_epochs_4,params)

fig, ax = plt.subplots()
ax.plot(f,plvtap_1.T)
plt.title(labels[0])

fig, ax = plt.subplots()
ax.plot(f,plvtap_2.T)
plt.title(labels[1])

fig, ax = plt.subplots()
ax.plot(f,plvtap_3.T)
plt.title(labels[2])

fig, ax = plt.subplots()
ax.plot(f,plvtap_4.T)
plt.title(labels[3])

fig,ax = plt.subplots()
ax.plot(f,plvtap_1.T,color='r')
ax.plot(f,plvtap_2.T,color='b')

fig,ax = plt.subplots()
ax.plot(f,plvtap_1[:,:].T,color='r',label=labels[0])
ax.plot(f,plvtap_2[:,:].T,color='b',label=labels[1])


fig,ax = plt.subplots()
ax.plot(f,plvtap_3.T,color='r',label=labels[2])
ax.plot(f,plvtap_4.T,color='b',label=labels[3])










