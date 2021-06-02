#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:58:39 2021

@author: ravinderjit
"""

import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pickle
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/Sentences/'
subject = 'S211'

refchans = ['EXG1','EXG2']
nchans =34
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']

sentEEG, sent_evnt = EEGconcatenateFolder(data_loc+subject+'/',nchans,refchans,exclude)

sentEEG.filter(1,200)

blinks = find_blinks(sentEEG, ch_name = ['A1'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
   
blink_epochs = mne.Epochs(sentEEG,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))

_Projs = compute_proj_epochs(blink_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

ocular_projs = [_Projs[0], _Projs[2]]

scalings = dict(eeg=40e-6)
sentEEG.add_proj(ocular_projs)
sentEEG.plot_projs_topomap()
sentEEG.plot(events=blinks,scalings=scalings, show_options=True)

#%% evoked responses

evkd = []

epochs1 = mne.Epochs(sentEEG,sent_evnt,1,tmin=-0.5,tmax=2.5,reject=dict(eeg=150e-6))
epochs2 = mne.Epochs(sentEEG,sent_evnt,2,tmin=-0.5,tmax=2.9,reject=dict(eeg=150e-6))

evkd1 = epochs1.average()
evkd2 = epochs2.average()

evkd1.plot(titles='Sent1',picks=[30,31])
evkd2.plot(titles='Sent2',picks=[30,31])

with open(os.path.join(data_loc,subject +'_sentEnv.pickle'),'wb') as f:
    pickle.dump([evkd1,evkd2],f)


