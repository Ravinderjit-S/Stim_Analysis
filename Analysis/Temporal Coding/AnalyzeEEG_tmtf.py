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

Subjects = ['S207','S211','S228','S236','S238','S239','S246']

freqs_all = np.array([[3, 6, 9, 14, 18, 30, 48, 55], #S207
             [2, 4, 8, 16, 24, 32, 48, 64], #S211
             [3, 6, 8, 14, 18, 24, 40, 64], #S228
             [4, 6, 8, 10, 14, 18, 26, 40], #S236
             [4, 8, 14, 18, 24, 38, 48, 64], #S238
             [4, 8, 14, 18, 22, 24, 32, 42], #S239
             [4, 6, 10, 14, 20, 32, 42, 54], #S246
             ])

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    
    datapath =  os.path.join(data_loc, subject)
    data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
    data_eeg.filter(l_freq=1,h_freq=1000)
    
    if subject == 'S207':
        data_eeg.info['bads'].append('A8')
        data_eeg.info['bads'].append('A14')
        data_eeg.info['bads'].append('A15')
        data_eeg.info['bads'].append('A16')
        data_eeg.info['bads'].append('A17')
    
    if subject == 'S228':
        data_eeg.info['bads'].append('A25')
        data_eeg.info['bads'].append('A28')
        
    if subject == 'S236':
        data_eeg.info['bads'].append('A7')
        data_eeg.info['bads'].append('A24')
        data_eeg.info['bads'].append('A3')
        data_eeg.info['bads'].append('A6')
        data_eeg.info['bads'].append('A11')


#%% Blink Removal
    blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
    blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                                  baseline=(-0.25,0),reject=dict(eeg=500e-6))
    Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
    ocular_projs = Projs
    
    if subject == 'S211':
        ocular_projs = [Projs[0], Projs[2]]
    if subject == 'S207':
        ocular_projs = [Projs[0]]
    if subject == 'S228':
        ocular_projs = [Projs[0]]
    if subject == 'S236':
        ocular_projs = [Projs[0]]
    if subject == 'S238':
        ocular_projs = [Projs[0], Projs[1]]
    if subject == 'S239':
        ocular_projs = [Projs[0], Projs[2]]
    if subject == 'S246':
        ocular_projs = [Projs[0], Projs[2]]
        
        
    
    data_eeg.add_proj(ocular_projs)
    # data_eeg.plot_projs_topomap()
    # data_eeg.plot(events=blinks,show_options=True)
    
    del ocular_projs, blinks, blink_epochs, Projs


#%% Plot data
    freqs = freqs_all[sub]
    triggers = np.arange(1,9)
    reject = dict(eeg=200e-6)
    epochs = []
    
    
    for tg in range(len(triggers)):    
        epochs_tg = mne.Epochs(data_eeg, data_evnt, [tg+1], tmin=-0.3, 
                                     tmax=1.2,reject=reject, baseline=(-0.2, 0.)) 
        epochs.append(epochs_tg)
        # epochs_tg.average().plot(picks=[31])
        

#%% compute tmtf
    t = epochs_tg[0].times
    t_1 = np.where(t>=0)[0][0]
    t_2 = np.where(t>=1)[0][0]
    fs = epochs_tg[0].info['sfreq']
    f = np.linspace(0,fs,t_2-t_1)
    tmtf_mag = np.zeros(len(freqs))
    Trials_cond = []
    
    for tg in range(len(triggers)):
        evkd = epochs[tg].get_data()[:,31,t_1:t_2].mean(axis=0)
        Trials_cond.append(epochs[tg].get_data()[:,31,:].shape[0])
        evkd = evkd - evkd.mean()
        ff = np.fft.fft(evkd)
        # plt.figure()
        # plt.plot(f,np.abs(ff))
        # plt.title(freqs[tg])
        f_ind = np.where(f>=freqs[tg]-0.5)[0][0]
        tmtf_mag[tg] = np.abs(ff[f_ind])
        
    print(Trials_cond)
        
    plt.figure()
    tmtf_mag_plot = (tmtf_mag / np.max(tmtf_mag))
    plt.plot(freqs,tmtf_mag_plot,marker='x',color='k')

#%% Save data
    with open(os.path.join(pickle_loc,subject+'_tmtf.pickle'),'wb') as file:
       pickle.dump([freqs, tmtf_mag, Trials_cond ],file)
    del data_eeg, data_evnt, epochs





