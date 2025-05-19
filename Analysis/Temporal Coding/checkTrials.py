#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:41:50 2025

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


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  
mseq_loc = 'mseqEEG_150_bits10_4096.mat'


file_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/' + mseq_loc
Mseq_dat = sio.loadmat(file_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)
mseq_shift = np.roll(mseq,int(np.round(mseq.size/2)))

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active_harder/'

Subjects = ['S207', 'S211', 'S228', 'S236', 'S238', 'S239', 'S250']

trials_activeHarder = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]

    datapath =  os.path.join(data_loc, subject)
    data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
    trials_activeHarder.append(np.sum(data_evnt[:,2]==1) + np.sum(data_evnt[:,2]==2) + np.sum(data_evnt[:,2]==3)  )
    
#%%


data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S207','S211','S228','S236','S238','S239','S246','S247','S250']
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
    
trials_active = []

for subject in Subjects:
    datapath =  os.path.join(data_loc, subject)
    data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
    trials_active.append(np.sum(data_evnt[:,2]==1))







