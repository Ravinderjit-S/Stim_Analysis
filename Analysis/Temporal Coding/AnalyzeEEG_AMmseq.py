#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:21:25 2020

@author: ravinderjit
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle



def PLV_Coh(X,Y,TW,fs):
    """
    X is the Mseq
    Y is time x trials
    TW is half bandwidth product 
    """
    X = X.squeeze()
    ntaps = 2*TW - 1
    dpss = sp.signal.windows.dpss(X.size,TW,ntaps)
    N = int(2**np.ceil(np.log2(X.size)))
    f = np.arange(0,N)*fs/N
    PLV_taps = np.zeros([N,ntaps])
    Coh_taps = np.zeros([N,ntaps])
    
    for k in range(0,ntaps):
        print('tap:',k+1,'/',ntaps)
        Xf = sp.fft(X *dpss[k,:],axis=0,n=N)
        Yf = sp.fft(Y * dpss[k,:].reshape(dpss.shape[1],1),axis=0,n=N)
        XYf = Xf.reshape(Xf.shape[0],1) * Yf.conj()
        PLV_taps[:,k] = abs(np.mean(XYf / abs(XYf),axis=1))
        Coh_taps[:,k] = abs(np.mean(XYf,axis=1) / np.mean(abs(XYf),axis=1))
        
    PLV = PLV_taps.mean(axis=1)
    Coh = Coh_taps.mean(axis=1)
    return PLV, Coh, f



nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  
Mseq_loc = '/media/ravinderjit/Data_Drive/Stim_Analysis/Stimuli/TemporalCoding/Stim_Dev/mseqEEG_4096.mat'  
Mseq_dat = sio.loadmat(Mseq_loc)
mseq = Mseq_dat['mseqEEG_4096']
mseq = mseq.astype(float)

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq'
subject = 'S233'

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved


datapath =  os.path.join(data_loc, subject)
# datapath = '/media/ravinderjit/Data_Drive/Data/EEGdata/EFR'
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=1500)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = Projs[0]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()

#%% Plot data

epoch_data = mne.Epochs(data_eeg, data_evnt, [1], tmin=-0.3, tmax=2.3,reject=None, baseline=(-0.2, 0.)) 
evkd_data = epoch_data.average();
evkd_data.plot(picks = [31], titles = 'AMmseq')

#%% Plot PSD
data_eeg.plot_psd(fmin=0,fmax=1100,tmin=0.0,tmax=1.0,proj=True,average=False)

#%% Extract part of response when stim is on
t = epoch_data.times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size
AMdata = epoch_data.get_data(picks=[31])
AMdata = AMdata.T[:,0,:]
AMdata = AMdata[t1:t2,:]
t = t[t1:t2]

#%% Remove epochs with large deflections
Peak2Peak = AMdata.max(axis=0) - AMdata.min(axis=0)
AMdata = AMdata[:,Peak2Peak*1e6 < 100.]
 
TW = 3
Fres = (1/t[-1]) * TW * 2
fs = epoch_data.info['sfreq']

PLV_AM, Coh_AM, f = PLV_Coh(mseq,AMdata,TW,fs)

fig = plt.figure()
plt.plot(f,Coh_AM,color='k')
plt.title('Coh AM')






