#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:10:22 2018

@author: ravinderjit
This analysis is for experiments trying to seperate EEG responses to ITD and IPD
"""

#import numpy as np
#import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs

Subject = 'Rav'
nchans = 34;
refchans = ['EXG1','EXG2']
direct_IPDITD = '/media/ravinderjit/Data_Drive/EEGdata/Binaural/IPD_ITD/'

Bin_eeg, Bin_evnt = EEGconcatenateFolder(direct_IPDITD+Subject+'/',nchans,refchans)
Bin_eeg.filter(70,1500)

blinks = find_blinks(Bin_eeg, ch_name = ['A1'], thresh = 100e-6, l_trans_bandwidth=0.5,l_freq=1.0)
scalings = dict(eeg=40e-6)
blink_epochs = mne.Epochs(Bin_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                          baseline=(-0.25,0),reject=dict(eeg=500e-6))
Proj = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
#Bin_eeg.add_proj(Proj)
#Bin_eeg.plot_projs_topomap()

eye_projs = [Proj[0],Proj[2]]
Bin_eeg.add_proj(eye_projs)
Bin_eeg.plot_projs_topomap()
Bin_eeg.plot(events=blinks,scalings=scalings,show_options=True,title = 'IPD ITD')

channels = [31]

stimITD = mne.Epochs(Bin_eeg,Bin_evnt,1,tmin=-0.5,tmax=2.0)
stimITD = stimITD.average()
stimITD.plot(picks=channels,titles = 'ITD')

stimIPD =mne.Epochs(Bin_eeg,Bin_evnt,2,tmin=-0.5,tmax=2.0)
stimIPD = stimIPD.average()
stimIPD.plot(picks=channels,titles= 'IPD')

stimITD2 =mne.Epochs(Bin_eeg,Bin_evnt,3,tmin=-0.5,tmax=2.0)
stimITD2 = stimITD2.average()
stimITD2.plot(picks=channels,titles='ITD2')

stimIPD2 =mne.Epochs(Bin_eeg,Bin_evnt,4,tmin=-0.5,tmax=2.0)
stimIPD2 = stimIPD2.average()
stimIPD2.plot(picks=channels,titles ='IPD2')


