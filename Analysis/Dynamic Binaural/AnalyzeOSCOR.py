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
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
    
direct_ = '/media/ravinderjit/Data_Drive/EEGdata/OSCOR'


exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
#exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8',
#           'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10', 'A11',
#           'A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22', 
#           'A23','A24','A25','A26','A27','A28','A29','A30','A32']

data_eeg,data_evnt = EEGconcatenateFolder(direct_+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=300)


epoch_data = mne.Epochs(data_eeg, data_evnt, [1], tmin=-0.1, tmax=1.1,reject=dict(eeg=200e-6)) #, baseline=(-0.2, 0.)) 
evkd_data = epoch_data.average();
evkd_data.plot(titles = 'OSCOR')

#data_raw = data_eeg.get_data()[0:31,:]
data_raw = evkd_data.data[0:31,:]
data_raw = data_raw.mean(axis=0)
data_raw = data_raw - data_raw.mean()
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

