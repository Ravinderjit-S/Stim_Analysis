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


folder = 'Q394_111120'
data_loc = '/media/ravinderjit/Storage2/ChinCap/SAM_noise/' + folder + '/SAM_'
pathThing = '/'
nchans = 34
refchans = ['EXG1','EXG2']
#exclude = ['A1','A25','A26','A27','A28','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']
data_AM5,evnts_AM5 = EEGconcatenateFolder(data_loc + '5' + pathThing ,nchans,refchans,exclude)
data_AM5.filter(2,40)
#data_eeg.notch_filter(60)
scalings = dict(eeg=20e-6,stim=1)
data_AM5.plot(events = evnts_AM5, scalings=scalings,show_options=True)

bad_chs = [0,5,8,9,16,24,25,26,27]
All_chs = np.arange(32)
channels = np.delete(All_chs,bad_chs)
channels = [1,2,7]

epochs_AM5 = mne.Epochs(data_AM5,evnts_AM5,[255],tmin=-0.3,tmax=1.3)
evoked_AM5 = epochs_AM5.average()
evoked_AM5.plot(titles = 'AM 5',picks=channels)

data_AM40,evnts_AM40 = EEGconcatenateFolder(data_loc + '40' + pathThing ,nchans,refchans,exclude)
data_AM40.filter(2,100)
epochs_AM40 = mne.Epochs(data_AM40,evnts_AM40,[255],tmin=-0.3,tmax=1.3)
evoked_AM40 = epochs_AM40.average()
evoked_AM40.plot(titles = 'AM 40',picks=[2,3,4])

data_AM223,evnts_AM223 = EEGconcatenateFolder(data_loc + '223' + pathThing ,nchans,refchans,exclude)
data_AM223.filter(2,300)
epochs_AM223 = mne.Epochs(data_AM223,evnts_AM223,[255],tmin=-0.3,tmax=1.3)
evoked_AM223 = epochs_AM223.average()
evoked_AM223.plot(titles = 'AM 223',picks=channels)


chan = 2
x = evoked_AM5.data[chan,:]
x = x - x.mean()
fs = data_AM5.info['sfreq']
nfft = 2**np.ceil(np.log(x.size)/np.log(2))
[f, pxx] = spA.periodogram(x,fs,nfft)

plt.figure()
plt.plot(f,10*np.log10(pxx))
plt.title('AM 5')

x = evoked_AM40.data[chan,:]
x = x - x.mean()
fs = data_AM5.info['sfreq']
nfft = 2**np.ceil(np.log(x.size)/np.log(2))
[f, pxx] = spA.periodogram(x,fs,nfft)

plt.figure()
plt.plot(f,10*np.log10(pxx))
plt.title('AM 40')

x = evoked_AM223.data[chan,:]
x = x - x.mean()
fs = data_AM5.info['sfreq']
nfft = 2**np.ceil(np.log(x.size)/np.log(2))
[f, pxx] = spA.periodogram(x,fs,nfft)

plt.figure()
plt.plot(f,10*np.log10(pxx))
plt.title('AM 223')



