# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:28:27 2020

@author: StuffDeveloping
"""


import matplotlib.pyplot as plt
from EEGpp import EEGconcatenateFolder
import mne
#import os
import numpy as np
import spectralAnalysis as spA



data_loc = '/media/ravinderjit/Storage2/ChinCap/SAM_noise/AM_'
pathThing = '/'
nchans = 37
refchans = ['EXG1']
exclude = ['A1','EXG2','EXG6','EXG7','EXG8']
data_AM5,evnts_AM5 = EEGconcatenateFolder(data_loc + '5' + pathThing ,nchans,refchans,exclude)
data_AM5.filter(1,40)
#data_eeg.notch_filter(60)
scalings = dict(eeg=20e-6,stim=1)
#data_AM5.plot(events = evnts_AM5, scalings=scalings,show_options=True)


epochs_AM5 = mne.Epochs(data_AM5,evnts_AM5,[255],tmin=-0.2,tmax=1.3)
evoked_AM5 = epochs_AM5.average()
evoked_AM5.plot(titles = 'AM 5')

data_AM40,evnts_AM40 = EEGconcatenateFolder(data_loc + '40' + pathThing ,nchans,refchans,exclude)
data_AM40.filter(1,40)
epochs_AM40 = mne.Epochs(data_AM40,evnts_AM40,[255],tmin=-0.2,tmax=1.3)
evoked_AM40 = epochs_AM40.average()
evoked_AM40.plot(titles = 'AM 40')

data_AM120,evnts_AM120 = EEGconcatenateFolder(data_loc + '120' + pathThing ,nchans,refchans,exclude)
data_AM120.filter(1,40)
epochs_AM120 = mne.Epochs(data_AM120,evnts_AM120,[255],tmin=-0.2,tmax=1.3)
evoked_AM120 = epochs_AM120.average()
evoked_AM120.plot(titles = 'AM 120')



x = evoked_AM5.data[7,:]
x = x - x.mean()
fs = data_AM5.info['sfreq']
nfft = 2**np.ceil(np.log(x.size)/np.log(2))
[f, pxx] = spA.periodogram(x,fs,nfft)

plt.figure()
plt.plot(f,10*np.log10(pxx))
plt.title('AM 5')

x = evoked_AM40.data[7,:]
x = x - x.mean()
fs = data_AM5.info['sfreq']
nfft = 2**np.ceil(np.log(x.size)/np.log(2))
[f, pxx] = spA.periodogram(x,fs,nfft)

plt.figure()
plt.plot(f,10*np.log10(pxx))
plt.title('AM 40')

x = evoked_AM120.data[7,:]
x = x - x.mean()
fs = data_AM5.info['sfreq']
nfft = 2**np.ceil(np.log(x.size)/np.log(2))
[f, pxx] = spA.periodogram(x,fs,nfft)

plt.figure()
plt.plot(f,10*np.log10(pxx))
plt.title('AM 120')



