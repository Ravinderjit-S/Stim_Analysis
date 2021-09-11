#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:04:01 2020

@author: ravinderjit
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  
#data_loc = '/media/ravinderjit/Storage2/EEGdata'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/'
subject = 'CMR_TempCoherence'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']

datapath = os.path.join(data_loc,subject)

data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=150)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = Projs[0]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=data_evnt)

#%% Plot Data

epochs_10 = mne.Epochs(data_eeg,data_evnt,[1],tmin=-1.5,tmax=4.5,
                              baseline=(-0.3,0),reject=dict(eeg=100e-6))
evkd_10 = epochs_10.average();
evkd_10.plot(picks=[31],titles='10')

epochs_40 = mne.Epochs(data_eeg,data_evnt,[2],tmin=-1.5,tmax=4.5,
                              baseline=(-0.3,0),reject=dict(eeg=100e-6))
evkd_40 = epochs_40.average();
evkd_40.plot(picks=31,titles='40')

epochs_110 = mne.Epochs(data_eeg,data_evnt,[3],tmin=-1.5,tmax=4.5,
                              baseline=(-0.3,0),reject=dict(eeg=100e-6))
evkd_110 = epochs_110.average();
evkd_110.plot(picks=31,titles='109')

#%% Spectral Content

freqs = np.arange(1.,130.,1.)
T = 1./2
n_cycles = freqs*T
time_bandwidth = 4
vmin = -0.1
vmax = 0.1

channels = [0,15,4,30,31,25,21,8]

tfr_e10 = mne.time_frequency.tfr_multitaper(epochs_10,freqs=freqs,n_cycles=n_cycles, time_bandwidth = time_bandwidth, 
                                            return_itc = False,picks = channels,decim=8)
tfr_e10.plot_topo(baseline =(-1.0,0),mode='logratio',title='e10',vmin=vmin,vmax=vmax)

tfr_e40 = mne.time_frequency.tfr_multitaper(epochs_40,freqs=freqs,n_cycles=n_cycles, time_bandwidth = time_bandwidth, 
                                            return_itc = False,picks = channels,decim=8)
tfr_e40.plot_topo(baseline =(-1.0,0),mode='logratio',title='e40',vmin=vmin,vmax=vmax)

tfr_e110 = mne.time_frequency.tfr_multitaper(epochs_110,freqs=freqs,n_cycles=n_cycles, time_bandwidth = time_bandwidth, 
                                            return_itc = False,picks = channels,decim=8)
tfr_e110.plot_topo(baseline =(-1.0,0),mode='logratio',title='e110',vmin=vmin,vmax=vmax)








