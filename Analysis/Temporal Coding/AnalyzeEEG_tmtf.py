#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 18:11:22 2021

@author: ravinderjit
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle
import sys
sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr

# from anlffr.spectral import mtspecraw
# from anlffr.spectral import mtplv
from sklearn.decomposition import PCA


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/tmtf/'
pickle_loc = data_loc + 'Pickles/'

subject = 'S211'

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
    
datapath =  os.path.join(data_loc, subject)
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=1000)


#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
ocular_projs = Projs

if subject == 'S211':
    ocular_projs = [Projs[0], Projs[2]]



data_eeg.add_proj(ocular_projs)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=blinks,show_options=True)


#%% Plot data
freqs = [2,4,8,16,24,32,48,64]
triggers = np.arange(1,9)
reject = dict(eeg=150e-6)
epochs = []

for tg in range(len(triggers)):    
    epochs_tg = mne.Epochs(data_eeg, data_evnt, [tg+1], tmin=-0.3, 
                                 tmax=1.2,reject=reject, baseline=(-0.2, 0.)) 
    epochs.append(epochs_tg)
    epochs_tg.average().plot(picks=[31])


#%% compute tmtf
t = epochs_tg[0].times
t_1 = np.where(t>=0)[0][0]
t_2 = np.where(t>=1)[0][0]
fs = epochs_tg[0].info['sfreq']
f = np.linspace(0,fs,t_2-t_1)
tmtf_mag = np.zeros(len(freqs))


for tg in range(len(triggers)):
    evkd = epochs[tg].get_data()[:,31,t_1:t_2].mean(axis=0)
    evkd = evkd - evkd.mean()
    ff = np.fft.fft(evkd)
    plt.figure()
    plt.plot(f,np.abs(ff))
    plt.title(freqs[tg])
    f_ind = np.where(f>=freqs[tg]-0.5)[0][0]
    tmtf_mag[tg] = np.abs(ff[f_ind])
    
plt.figure()
tmtf_mag = (tmtf_mag / np.max(tmtf_mag))*0.14
plt.plot(freqs,tmtf_mag,marker='x',color='k')







