#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:47:40 2019

@author: ravinderjit
ran a pilot with passive and active ABAB stim ... analyzing here
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs

#Corr_inds{1} = [];
#Corr_inds{2} = 15:16; %2
#Corr_inds{3} = 13:16; %4
#Corr_inds{4} = 11:16; %6
#Corr_inds{5} = 9:16;  %8
#Corr_inds{6} = [1,6,11,16];
#Corr_inds{7} = [1,4,7,10,13,16];
#Corr_inds{8} = 3:16;

nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
    
#direct_ = '../../../Data/EEGdata/Binding/BindingPassivePilot'
direct_ = '/media/ravinderjit/Data_Drive/Dropbox/Lab/Data/EEGdata/Binding/BindingBehPilot'


exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved



data_eeg,data_evnt = EEGconcatenateFolder(direct_+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=100)


## blink removal
blinks_eeg = find_blinks(data_eeg, ch_name = ['A1'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
scalings = dict(eeg=40e-6,stim=0.1)

blink_epochs = mne.Epochs(data_eeg,blinks_eeg,998,tmin=-0.25,tmax=0.25,proj=False,
                          baseline=(-0.25,0),reject=dict(eeg=500e-6))

Projs_data = compute_proj_epochs(blink_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

#data_eeg.add_proj(Projs_data)   
#data_eeg.plot_projs_topomap()

#if Subject == 'Rav':                     
#eye_projs = [Projs_data[0],Projs_data[2]]
eye_projs = Projs_data[0]


    
data_eeg.add_proj(eye_projs)
# data_eeg.plot_projs_topomap()
# data_eeg.plot(events=blinks_eeg,scalings=scalings,show_options=True,title = 'BindingData')

channels = [31,4,26,25,30]
ylim_vals = [-3.5,3]
ts = -0.3
te = 3.7  #stim should be 0-2.8

# epochsAll = mne.Epochs(data_eeg, data_evnt, [1, 2, 3,4,5,6,7,8], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6)) #, baseline=(-0.2, 0.)) 
# evokedAll = epochsAll.average()
# evokedAll.plot(picks=channels,titles ='BindingAll_evoked auditory chan')   #ylim = dict(eeg=ylim_vals))   
# evokedAll.plot(picks=[13,14,15],titles ='visual channels')


# epochsAll_noProj = mne.Epochs(data_eeg, data_evnt, [1, 2, 3,4,5,6,7,8], tmin= ts, tmax= te, proj=False,reject=dict(eeg=200e-6)) 
# evoked_noProj = epochsAll_noProj.average()
# evoked_noProj.plot(picks=channels,titles = 'Binding Proj off')

# evk = evokedAll.data[31,:]
# evk_np = evoked_noProj.data[31,:]
# plt.figure()
# t = np.arange(0,evk.size/4096.,1./4096.)
# plt.plot(t,evk,c='b')
# plt.plot(t,evk_np,c='r')


#mod(val,256)

Proj_OnOFF = True #True means projections are on

epochs_e1 = mne.Epochs(data_eeg, data_evnt, [1], tmin= ts, tmax= te, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e1 = epochs_e1.average()

epochs_e2 = mne.Epochs(data_eeg, data_evnt, [2], tmin= ts, tmax= te, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e2 = epochs_e2.average()

epochs_e3 = mne.Epochs(data_eeg, data_evnt, [3], tmin= ts, tmax= te, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e3 = epochs_e3.average()

epochs_e4 = mne.Epochs(data_eeg, data_evnt, [4], tmin= ts, tmax= te, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e4 = epochs_e4.average()

epochs_e5 = mne.Epochs(data_eeg, data_evnt, [5], tmin= ts, tmax= te, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e5 = epochs_e5.average()

epochs_e6 = mne.Epochs(data_eeg, data_evnt, [6], tmin= ts, tmax= te, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e6 = epochs_e6.average()

epochs_e7 = mne.Epochs(data_eeg, data_evnt, [7], tmin= ts, tmax= te, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e7 = epochs_e7.average()

epochs_e8 = mne.Epochs(data_eeg, data_evnt, [8], tmin= ts, tmax= te, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e8 = epochs_e8.average()


#plots
evoked_e1.plot(picks=channels,titles = '1')
evoked_e2.plot(picks=channels,titles = '2')
evoked_e3.plot(picks=channels,titles = '3')
evoked_e4.plot(picks=channels,titles = '4')
evoked_e5.plot(picks=channels,titles = '5')
evoked_e6.plot(picks=channels,titles = '6')
evoked_e7.plot(picks=channels,titles = '7')
evoked_e8.plot(picks=channels,titles = '8')


freqs = np.arange(1.,100.,1.)
T = 1./5
n_cycles = freqs*T
time_bandwidth = 2
vmin = -3
vmax = 3

channels = np.arange(0,32)

# power_e1 = mne.time_frequency.tfr_multitaper(epochs_e1, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
# power_e1.plot_topo(baseline =(-0.3,0),mode= 'zlogratio', title = 'e1', vmin=vmin,vmax=vmax)

# power_e2 = mne.time_frequency.tfr_multitaper(epochs_e2, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
# power_e2.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e2', vmin=vmin,vmax=vmax)

# power_e3 = mne.time_frequency.tfr_multitaper(epochs_e3, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
# power_e3.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e3', vmin=vmin,vmax=vmax)

# power_e4 = mne.time_frequency.tfr_multitaper(epochs_e4, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
# power_e4.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e4', vmin=vmin,vmax=vmax)

# power_e5 = mne.time_frequency.tfr_multitaper(epochs_e5, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
# power_e5.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e5', vmin=vmin,vmax=vmax)

# power_e6 = mne.time_frequency.tfr_multitaper(epochs_e6, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
# power_e6.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e6', vmin=vmin,vmax=vmax)

# power_e7 = mne.time_frequency.tfr_multitaper(epochs_e7, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
# power_e7.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e7', vmin=vmin,vmax=vmax)

# power_e8 = mne.time_frequency.tfr_multitaper(epochs_e8, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
# power_e8.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e8', vmin=vmin,vmax=vmax)




power_e1_in = mne.time_frequency.tfr_multitaper(epochs_e1.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=4)
power_e1_in.plot_topo(baseline =(-0.3,0),mode= 'zlogratio', title = 'e1_induced', vmin=vmin,vmax=vmax)

#power_e2_in = mne.time_frequency.tfr_multitaper(epochs_e2.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
#power_e2_in.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e2_induced', vmin=vmin,vmax=vmax)
#
#power_e3_in = mne.time_frequency.tfr_multitaper(epochs_e3.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
#power_e3_in.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e3_induced', vmin=vmin,vmax=vmax)
#
#power_e4_in = mne.time_frequency.tfr_multitaper(epochs_e4.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
#power_e4_in.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e4_induced', vmin=vmin,vmax=vmax)
#
#power_e5_in = mne.time_frequency.tfr_multitaper(epochs_e5.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
#power_e5_in.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e5_induced', vmin=vmin,vmax=vmax)
#
#power_e6_in = mne.time_frequency.tfr_multitaper(epochs_e6.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
#power_e6_in.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e6_induced', vmin=vmin,vmax=vmax)
#
#power_e7_in = mne.time_frequency.tfr_multitaper(epochs_e7.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
#power_e7_in.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e7_induced', vmin=vmin,vmax=vmax)

power_e8_in = mne.time_frequency.tfr_multitaper(epochs_e8.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=4)
power_e8_in.plot_topo( baseline = (-0.3, 0), mode= 'zlogratio', title = 'e8_induced', vmin=vmin,vmax=vmax)

T0 = 0.0
T1 = 0.7
T2 = 1.4
T3 = 2.1
T4 = 2.8

t = power_e1_in.times
power = power_e1_in.data
power2 = power_e8_in.data

ch = 30
frequns = range(70,90)
look1 = np.subtract(power[ch,frequns,:].T,power[ch,frequns,0:t_0+1].mean(axis=1)).mean(axis=1)
look2 = np.subtract(power2[ch,frequns,:].T,power2[ch,frequns,0:t_0+1].mean(axis=1)).mean(axis=1)

t_0 = np.nonzero(t>=0.0)[0][0]

plt.figure()
plt.plot(t,look1,color='b')
plt.plot(t,look2,color='r')












