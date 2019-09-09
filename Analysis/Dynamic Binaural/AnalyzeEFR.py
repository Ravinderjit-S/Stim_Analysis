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
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from anlffr import spectral
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
    
direct_ = '/media/ravinderjit/Data_Drive/EEGdata/EFR'


exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
#exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8',
#           'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10', 'A11',
#           'A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22', 
#           'A23','A24','A25','A26','A27','A28','A29','A30','A32']

data_eeg,data_evnt = EEGconcatenateFolder(direct_+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=300)
Trigs = data_evnt[:,2]
Trigs[1001:] = 2
data_evnt[:,2] = Trigs   #ran 1000 trials at mod of 41 Hz and 1000 at mod of 151 but all have the same trigger. Giving 151 mod rate trigg value of 2

epoch_data = mne.Epochs(data_eeg, data_evnt, [2], tmin=-0.1, tmax=1.1,reject=dict(eeg=200e-6)) #, baseline=(-0.2, 0.)) 
evkd_data = epoch_data.average();
evkd_data.plot(titles = 'EFR')

#data_raw = data_eeg.get_data()[0:31,:]
data_raw = evkd_data.data[0:31,:]
data_raw = data_raw[30,:]; #data_raw.mean(axis=0)
data_raw = data_raw - data_raw.mean()
plt.figure()
plt.plot(data_raw)
fs = data_eeg.info['sfreq']

#data_fft = np.fft.fft(data_raw,axis=1)
#freq = np.fft.fftfreq(data_raw.shape[1],1/fs)
data_fft = np.fft.fft(data_raw)
freq = np.fft.fftfreq(data_raw.shape[0],1/fs)
#plt.figure()
#t = np.arange(0,data_raw.size/fs,1./fs)
#plt.plot(t,data_raw)
plt.figure()
#plt.plot(freq,np.abs(data_fft.T))
plt.plot(freq,20*np.log10(np.abs(data_fft.T)))


#params = {"fs": fs, "tapers": [1,1], "fpass": [1, 300]}
#S,N,F = spectral.mtspec(data_raw,params)





