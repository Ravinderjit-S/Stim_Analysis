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
#import spectralAnalysis as spA
from anlffr import spectral
import scipy.io as sio


folder = 'Chin_SAM_tmtf/'
#data_loc = '/media/ravinderjit/Storage2/ChinCap/'
data_loc = '/home/ravinderjit/Documents/ChinCapData/'
nchans = 35
# refchans = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19',
#             'A20','A21','A22','A23','A29','A30','A31','A32']
refchans = ['EXG1','EXG2']
exclude = ['EXG4','EXG5','EXG6','EXG7','EXG8']
data_eeg,evnts_eeg = EEGconcatenateFolder(data_loc + folder ,nchans,refchans,exclude)
data_eeg.filter(1,80) 
data_eeg.set_channel_types({'EXG3':'eeg'})

AMf = [20,30,40,55,70,90,110,170,250,400,600,800,1000,3000]
bad_chs = ['A1','A25','A26','A27','A28']
data_eeg.drop_channels(bad_chs)
data_eeg.set_eeg_reference(ref_channels='average')

scalings = dict(eeg=20e-6,stim=1)
data_eeg.plot(events = evnts_eeg, scalings=scalings,show_options=True)

epochs_all = []
for m in np.arange(len(AMf)):
    epochs_m = mne.Epochs(data_eeg,evnts_eeg,[m+1],tmin=-0.050,tmax=0.250,baseline=(-0.050,0),reject=dict(eeg=200e-6))
    evoked_m = epochs_m.average()
    evoked_m.plot(titles = str(AMf[m]))
    epochs_all.append(epochs_m)
    


Aud_picks = ['A30', 'A6', 'A29', 'A7', 'A4', 'A17', 'A32', 'A10', 'A3']

t = epochs_all[0].times
t1 = np.where(t>=0)[0][0]
t2 = np.where(t>=0.2)[0][0]
t = t[t1:t2]

epoch_dat = []  
for m in range(len(AMf)):
    epoch_dat_m = epochs_all[m].get_data(picks=Aud_picks)
    epoch_dat_m = epoch_dat_m[:,:,t1:t2].transpose(1,0,2)
    epoch_dat.append(epoch_dat_m)

params = dict()
params['Fs'] = epochs_all[0].info['sfreq']
params['tapers'] = [2,2*2-1]
params['fpass'] = [1,4000]
params['itc'] = 0

for m in np.arange(len(AMf)):
    plvtap, f = spectral.mtplv(epoch_dat[m],params)
    plt.figure()
    plt.plot(f,plvtap.T)
    plt.title(str(AMf[m]))
    plt.xlim((0,AMf[m]*3))

# plvtap_1, f = spectral.mtplv(epoch40_dat,params)
# Aud_inds = np.flip(np.argsort(np.max(plvtap_1[:,75:85],axis=1)))

# Aud_chans = []
# for ind in Aud_inds[0:9]:
#     Aud_chans.append(epochs_AM40.ch_names[ind])

# fig, ax = plt.subplots(figsize=(5,3.3))
# fontsize=10
# ax.plot(f,plvtap_1.T)
# plt.ylabel('PLV',fontsize=fontsize)
# plt.xlabel('Frequency (Hz)',fontsize=fontsize)
# plt.xticks([0,20,40,60,80,100],fontsize=fontsize)
# plt.yticks([0.1,0.2,0.3],fontsize=fontsize)

# fig, ax = plt.subplots(figsize=(5,3.3))
# fontsize=10
# ax.plot(f,plvtap_1[Aud_inds[0:9],:].T)
# plt.ylabel('PLV',fontsize=fontsize)
# plt.xlabel('Frequency (Hz)',fontsize=fontsize)
# plt.xticks([0,20,40,60,80,100],fontsize=fontsize)
# plt.yticks([0.1,0.2,0.3],fontsize=fontsize)


# evoked_dat40 = evoked_AM40.data[5,:]
# fig,ax = plt.subplots(figsize=(5,3.3))
# fontsize=10
# ax.plot(epochs_AM40.times,evoked_dat40*1e6,linewidth=2,color='k')
# plt.xlim([-0.1,2.1])
# plt.xticks([0,0.5,1.0,1.5,2.0],fontsize=fontsize)
# plt.yticks([-3,-2,-1,0,1],fontsize=fontsize)
# plt.ylabel('\u03BCV',fontsize=fontsize)
# plt.xlabel('Time (sec)',fontsize=fontsize)


