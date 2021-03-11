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
from scipy.signal import periodogram 
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
data_eeg.filter(5,3500) 
data_eeg.set_channel_types({'EXG3':'eeg'})

AMf = [20,30,40,55,70,90,110,170,250,400,600,800,1000,3000]
bad_chs = ['A1','A25','A26','A27','A28','EXG3','EXG1','EXG2','A20','A21','A22','A30']
data_eeg.info['bads'] = bad_chs
#data_eeg.drop_channels(bad_chs)
data_eeg.set_eeg_reference(ref_channels='average')

scalings = dict(eeg=20e-6,stim=1)
data_eeg.plot(events = evnts_eeg, scalings=scalings,show_options=True)

epochs_all = []
for m in np.arange(len(AMf)):
    epochs_m = mne.Epochs(data_eeg,evnts_eeg,[m+1],tmin=-0.050,tmax=0.250,baseline=(-0.050,0),reject=dict(eeg=200e-6))
    evoked_m = epochs_m.average()
    evoked_m.plot(titles = str(AMf[m]))
    epochs_all.append(epochs_m)
    


# Aud_picks = ['A30', 'A6', 'A29', 'A7', 'A4', 'A17', 'A32', 'A10', 'A3']
# All_picks = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
#  'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
#  'A21', 'A22', 'A23', 'A24', 'A29', 'A30', 'A31', 'A32']  #Took out A20

t = epochs_all[0].times
t1 = np.where(t>=0)[0][0]
t2 = np.where(t>=0.2)[0][0]
t = t[t1:t2]

epoch_dat = []  
for m in range(len(AMf)):
    epoch_dat_m = epochs_all[m].get_data(picks='data')
    epoch_dat_m = epoch_dat_m[:,:,t1:t2].transpose(1,0,2)
    epoch_dat.append(epoch_dat_m)

AMf = AMf[2:] #first 2 triggers mest up
epoch_dat = epoch_dat[2:]


params = dict()
params['Fs'] = epochs_all[0].info['sfreq']
params['tapers'] = [1,2*1-1]
params['fpass'] = [1,4000]
params['itc'] = 0

All_plvs = []
MaxAud_plvs = np.zeros((epoch_dat[0].shape[0],len(AMf)))
tmtf_plv = np.zeros((epoch_dat[0].shape[0],len(AMf)))
for m in np.arange(len(AMf)):
    plvtap, f = spectral.mtplv(epoch_dat[m],params)
    floc1 = np.where(f>=AMf[m]-10)[0][0]
    floc2 = np.where(f>=AMf[m]+10)[0][0]
    MaxAud_plvs[:,m] = np.flip(np.argsort(np.max(plvtap[:,floc1:floc2],axis=1)))
    tmtf_plv[:,m] = np.max(plvtap[:,floc1:floc2],axis=1)
    plt.figure()
    plt.plot(f,plvtap.T)
    plt.title(str(AMf[m]))
    #plt.xlim((0,AMf[m]*3))
    All_plvs.append(plvtap)
    
AMlabels = []
for AM in AMf: AMlabels.append(str(AM))
plt.figure()
plt.semilogx(AMf,tmtf_plv.T)
plt.xticks(ticks=AMf,labels=AMlabels)
plt.title('TMTF_PLV')
#plt.xlim((0,1000))
    

fs = epochs_all[0].info['sfreq']
nfft = int(2**np.ceil(np.log2(epoch_dat[0].shape[2])))
MaxAud_Axx = np.zeros((epoch_dat[0].shape[0],len(AMf)))
tmtf_Axx = np.zeros((epoch_dat[0].shape[0],len(AMf)))
f = np.arange(0,fs,fs/nfft)
for m in np.arange(len(AMf)):
    Axx = np.fft.fft(epoch_dat[m].mean(axis=1),n=nfft)
    Axx = np.abs(Axx)
    plt.figure()
    floc1 = np.where(f>=AMf[m]-20)[0][0]
    floc2 = np.where(f>=AMf[m]+20)[0][0]
    MaxAud_Axx[:,m] = np.flip(np.argsort(np.max(Axx[:,floc1:floc2],axis=1)))
    tmtf_Axx[:,m] = np.max(Axx[:,floc1:floc2],axis=1)
    plt.plot(f,Axx.T)
    plt.xlim([0,4000])
    plt.title(str(AMf[m]))
    
plt.figure()
plt.semilogx(AMf,tmtf_Axx.T)
plt.xticks(AMf,labels=AMlabels)
plt.title('TMTF_pxx')
plt.xlim((0,1000))
    
    
    
    


