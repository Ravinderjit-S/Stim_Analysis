#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:33:47 2019

@author: ravinderjit
"""


import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import mne
from anlffr.preproc import find_blinks
from anlffr.spectral import mtcpca
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os


def ITC(X,TW,fs):
    """
    X is time x trial
    TW is half bandwidth product 
    """
    X = X.squeeze()
    ntaps = 2*TW - 1
    dpss = sp.signal.windows.dpss(X.shape[0],TW,ntaps)
    N = int(2**np.ceil(np.log2(X.shape[0])))
    f = np.arange(0,N)*fs/N
    itc_taps = np.zeros([N,ntaps])
    
    for k in range(0,ntaps):
        print('tap:',k+1,'/',ntaps)
        Xf = sp.fft(X*dpss[k,:].reshape(dpss.shape[1],1),axis=0,n=N)
        itc_taps[:,k] = abs(np.mean(Xf,axis=1) / np.mean(abs(Xf),axis=1))

    itc = itc_taps.mean(axis=1)
    return itc,f



nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
    
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/OSCOR'
subject = 'S211_24'

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved


datapath =  os.path.join(data_loc, subject)
# datapath = '/media/ravinderjit/Data_Drive/Data/EEGdata/EFR'
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=300)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = Projs[0]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()

#%% Plot data

epoch_data = mne.Epochs(data_eeg, data_evnt, [1], tmin=-0.2, tmax=1.2,reject=dict(eeg=200e-6), baseline=(-0.2, 0.)) 
evkd_data = epoch_data.average();
evkd_data.plot(titles = 'OSCOR')

#%% Plot PSD
data_eeg.plot_psd(fmin=0,fmax=300,tmin=0.0,tmax=1.0,proj=True,average=False)

#%% ITC
dataAll = epoch_data.get_data(picks = range(0,32))
data32 = epoch_data.get_data(picks='A32').squeeze().T
t = epoch_data.times
t1 = np.where(t>=0)[0][0]
t2 = np.where(t>=1)[0][0]
data32 = data32[t1:t2,:]
fs = data_eeg.info['sfreq']
TW = 2 #fres = 2*TW
itc,f = ITC(data32,TW,fs) 

plt.figure()
plt.plot(f,itc)

data_reshaped = np.zeros([dataAll.shape[1],dataAll.shape[0],dataAll.shape[2]]) #ch x trial x time
for ch in range(0,32):
    data_reshaped[ch,:,:] = dataAll[:,ch,:]
    
params = {}
params['Fs'] = fs
params['tapers'] = [TW,2*TW-1]
params['fpass'] = [1, 100]
params['itc'] = 1

plv,f = mtcpca(data_reshaped,params)
plt.figure()
plt.plot(f,plv)
plt.xlabel('Freq (hz)')





