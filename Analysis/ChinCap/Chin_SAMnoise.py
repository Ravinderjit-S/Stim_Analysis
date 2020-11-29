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
from anlffr import spectral
import scipy.io as sio


folder = 'Q394_111320'
data_loc = '/media/ravinderjit/Storage2/ChinCap/SAM_noise/' + folder + '/SAM_'
pathThing = '/'
nchans = 37
refchans = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19',
            'A20','A21','A22','A23','A29','A30','A31','A32']
#refchans = ['A4']
#exclude = ['A1','A25','A26','A27','A28','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']
#exclude = ['A24','A25','A26','A27','A28','EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']
exclude = ['EXG6','EXG7','EXG8']
data_AM5,evnts_AM5 = EEGconcatenateFolder(data_loc + '4' + pathThing ,nchans,refchans,exclude)
data_AM5.filter(1,40)
data_AM5.set_channel_types({'EXG4':'eeg','EXG3':'eeg','EXG5':'eeg'})
#data_eeg.notch_filter(60)
scalings = dict(eeg=20e-6,stim=1)
data_AM5.plot(events = evnts_AM5, scalings=scalings,show_options=True)

#bad_chs = [0,5,8,9,16,24,25,26,27]
bad_chs =[23,24,25,26,27,29]
All_chs = np.arange(32)
channels = np.delete(All_chs,bad_chs)

bad_chs = ['A24','A25','A26','A27','A28','EXG1','EXG2','EXG4','EXG5']
data_AM5.drop_channels(bad_chs)
epochs_AM5 = mne.Epochs(data_AM5,evnts_AM5,[255],tmin=-0.5,tmax=1.3,baseline=(-0.3,0))
evoked_AM5 = epochs_AM5.average()
evoked_AM5.plot(titles = 'AM 4',picks=channels)

data_AM40,evnts_AM40 = EEGconcatenateFolder(data_loc + '40' + pathThing ,nchans,refchans,exclude)
data_AM40.filter(1,100)
data_AM40.drop_channels(bad_chs)
epochs_AM40 = mne.Epochs(data_AM40,evnts_AM40,[255],tmin=-0.5,tmax=2.3,baseline=(-0.2,0),reject=dict(eeg=200e-6))
evoked_AM40 = epochs_AM40.average()
evoked_AM40.plot(titles = 'AM 40')

# data_AM223,evnts_AM223 = EEGconcatenateFolder(data_loc + '223' + pathThing ,nchans,refchans,exclude)
# data_AM223.filter(1,300)
# epochs_AM223 = mne.Epochs(data_AM223,evnts_AM223,[255],tmin=-0.3,tmax=1.3)
# evoked_AM223 = epochs_AM223.average()
# evoked_AM223.plot(titles = 'AM 223',picks=channels)





# epochs_data5 = epochs_AM5.get_data()
# epochs_data5 = epochs_data5[:,1:27,:] #dropping stim channel
# epochs_data5 = epochs_data5.reshape(epochs_data5.shape[1],epochs_data5.shape[0],epochs_data5.shape[2])

# params = dict()
# N = epochs_data5.shape[2]
# params['Fs'] = epochs_AM5.info['sfreq'] 
# params['tapers'] = [1.5, 2] #nw = (tapers + 1) / 2
# params['fpass'] = [1,60]
# params['itc'] = 0

# plv_AM4, f = spectral.mtplv(epochs_data5,params)

# plt.figure()
# plt.plot(f,plv_AM4.T)

# epochs_data40 = epochs_AM40.get_data()
# epochs_data40 = epochs_data40[:,1:27,:] #dropping stim channel
# epochs_data40 = epochs_data40.reshape(epochs_data40.shape[1],epochs_data40.shape[0],epochs_data40.shape[2])

# params['fpass'] = [10, 100]
# plv_AM40, f = spectral.mtplv(epochs_data40,params)

# plt.figure()
# plt.plot(f,plv_AM40.T)

# epochs_data223 = epochs_AM223.get_data()
# epochs_data223 = epochs_data223[:,1:27,:] #dropping stim channel
# epochs_data223 = epochs_data223.reshape(epochs_data223.shape[1],epochs_data223.shape[0],epochs_data223.shape[2])

# params['fpass'] = [30, 300]
# plv_AM223, f = spectral.mtplv(epochs_data223,params)

# plt.figure()
# plt.plot(f,plv_AM223.T)



x4 = evoked_AM5.data
x40 = evoked_AM40.data
# x223 = evoked_AM223.data
chs = evoked_AM5.info['ch_names']
chs = np.arange(1,24)
chs = np.append(chs,np.arange(29,33))

sio.savemat(str('SAMnoise_' +folder+ '.mat'), {'x4':x4,'fs':fs})#'x40':x40,'x223':x223,'fs':fs,'chs':chs})

fs = data_AM5.info['sfreq']
nfft = 2**np.ceil(np.log(x4.size)/np.log(2))
[f, pxx] = spA.periodogram(x4.T,fs,nfft)
plt.figure()
plt.plot(f,10*np.log10(pxx))

[f,pxx] = spA.periodogram(x40.T,fs,nfft)
plt.figure()
plt.plot(f,10*np.log10(pxx))

# [f,pxx] = spA.periodogram(x223.T,fs,nfft)
# plt.figure()
# plt.plot(f,10*np.log10(pxx))






# chan = 7
# x = evoked_AM5.data[chan,:]
# x = x - x.mean()
# fs = data_AM5.info['sfreq']
# nfft = 2**np.ceil(np.log(x.size)/np.log(2))
# [f, pxx] = spA.periodogram(x,fs,nfft)

# plt.figure()
# plt.plot(f,10*np.log10(pxx))
# plt.title('AM 4')

# x = evoked_AM40.data[chan,:]
# x = x - x.mean()
# fs = data_AM5.info['sfreq']
# nfft = 2**np.ceil(np.log(x.size)/np.log(2))
# [f, pxx] = spA.periodogram(x,fs,nfft)

# plt.figure()
# plt.plot(f,10*np.log10(pxx))
# plt.title('AM 40')

# x = evoked_AM223.data[chan,:]
# x = x - x.mean()
# fs = data_AM5.info['sfreq']
# nfft = 2**np.ceil(np.log(x.size)/np.log(2))
# [f, pxx] = spA.periodogram(x,fs,nfft)

# plt.figure()
# plt.plot(f,10*np.log10(pxx))
# plt.title('AM 223')


