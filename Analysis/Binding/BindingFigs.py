#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:40:20 2020

@author: ravinderjit
"""

import numpy as np
# import scipy as sp
# import pylab as pl
import matplotlib.pyplot as plt
# import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
# import pickle

def plot_chAvg_tfr(tfr_obj, picks, vmin,vmax,title,bline):
# This function will average tfr data for the channels in picks. It references to baseline and does 10log10
#tfr_obj = mne tfr object
    

    tfr_data = tfr_obj.data[picks,:,:]
    t = tfr_obj.times
    f = tfr_obj.freqs
    AvgCh = tfr_data.mean(axis=0)
    
    if bline[0] == None:
        AvgCh = 10*np.log10(AvgCh/AvgCh.mean(axis=1).reshape(99,1))
    else:
        b1 = int(np.argwhere(t>=bline[0])[0])
        b2 = int(np.argwhere(t>=bline[1])[0])
        AvgCh = 10*np.log10(AvgCh/AvgCh[:,b1:b2].mean(axis=1).reshape(99,1))
    


    plt.figure(figsize=(5,4))
    Z = AvgCh
    im = plt.pcolormesh(t,f,Z,vmin=vmin,vmax=vmax)
    axx = im.axes
    cbar = plt.colorbar(im, ticks=[-1, 0,1])
    cbar.set_label('dB (re: average)')
    cbar.ax.set_yticklabels(['-1','0','1'])
   
    def format_coord(x, y):
    #finnicky function to get plot to print z value with cursor 
        x0, x1 = axx.get_xlim()
        y0, y1 = axx.get_ylim()
        col = int(np.floor((x-x0)/float(x1-x0)*t.size))
        row = int(np.floor((y-y0)/float(y1-y0)*f.size))
        if col >= 0 and col < Z.shape[1] and row >= 0 and row < Z.shape[0]:
            z = Z[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)
    
    im.axes.format_coord = format_coord
    # plt.title(title)
    plt.show()


data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/Binding/BindingPilot_Spring20/B1B2')
nchans = 34;
refchans = ['EXG1','EXG2']

EEG_type = 'Passive'
subject = 'S233_tones'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
data_path = os.path.join(data_loc, EEG_type, subject)  

data_eeg,data_evnt = EEGconcatenateFolder(data_path+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=100)

## blink removal
blinks_eeg = find_blinks(data_eeg, ch_name = ['A1'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
scalings = dict(eeg=40e-6,stim=0.1)

blink_epochs = mne.Epochs(data_eeg,blinks_eeg,998,tmin=-0.25,tmax=0.25,proj=False,
                      baseline=(-0.25,0),reject=dict(eeg=500e-6))

Projs_data = compute_proj_epochs(blink_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

# data_eeg.add_proj(Projs_data)   
# data_eeg.plot_projs_topomap()

# if subject == 'S132':
eye_projs = Projs_data[0]
data_eeg.add_proj(eye_projs)
data_eeg.plot_projs_topomap()

del Projs_data, blink_epochs, blinks_eeg, eye_projs

ts = -0.5
te = 4.5
channels = [31]

epochs_1 = mne.Epochs(data_eeg, data_evnt, [1], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim = 8) 
S233_16 = epochs_1.average()

epochs_2 = mne.Epochs(data_eeg, data_evnt, [2], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim=8) 
S233_12 = epochs_2.average()

S233_16.plot(picks=channels,titles='S233 16')
S233_12.plot(picks=channels,titles='S233 12')

freqs = np.arange(1.,100.,1.)
T = 1./5
n_cycles = freqs*T
time_bandwidth = 2
vmin = -.1
vmax = .1
bline = (0,4)
channels = np.arange(0,32)

epochs_1.load_data()
S233_16_tfr = mne.time_frequency.tfr_multitaper(epochs_1.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=1)#,average=False)
S233_16_tfr.plot_topo(baseline =bline,mode= 'logratio', title = 'e1', vmin=vmin,vmax=vmax)

picks = [4,25,30,31,8,21]
vmin = -1
vmax = 1
plot_chAvg_tfr(S233_16_tfr,picks,vmin,vmax,title= subject+'_e1_induced_' +EEG_type,bline =bline)
plt.xlabel('Time (sec)')
plt.ylabel( 'Frequency (Hz)')
plt.rcParams["font.family"] = 'Arial'
plt.rcParams['font.size'] = "11"
plt.savefig('/media/ravinderjit/Data_Drive/Data/Figures/BindingPilot/CortOsc.png',format='png')


EEG_type = 'Passive'
subject = 'S132'
data_path = os.path.join(data_loc, EEG_type, subject)  

data_eeg,data_evnt = EEGconcatenateFolder(data_path+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=40)

## blink removal
blinks_eeg = find_blinks(data_eeg, ch_name = ['A1'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
scalings = dict(eeg=40e-6,stim=0.1)

blink_epochs = mne.Epochs(data_eeg,blinks_eeg,998,tmin=-0.25,tmax=0.25,proj=False,
                      baseline=(-0.25,0),reject=dict(eeg=500e-6))

Projs_data = compute_proj_epochs(blink_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

# data_eeg.add_proj(Projs_data)   
# data_eeg.plot_projs_topomap()

# if subject == 'S132':
eye_projs = Projs_data[0]
data_eeg.add_proj(eye_projs)
data_eeg.plot_projs_topomap()


epochs_1 = mne.Epochs(data_eeg, data_evnt, [4], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim = 8) 
S132_16 = epochs_1.average()

epochs_2 = mne.Epochs(data_eeg, data_evnt, [2], tmin= ts, tmax= te, proj=True,reject=dict(eeg=200e-6), baseline=(-0.2, 0.),decim= 8) 
S132_12 = epochs_2.average()

S132_16.plot(picks=channels,titles='S132 16')
S132_12.plot(picks=channels,titles='S132 12')

plt.figure(figsize=(6,4))
t = S233_16.times
y = S233_16.data[31,:]*1e6
y2 = S233_12.data[31,:]*1e6
plt.plot(t,y,linewidth=2,color='b',label='16')
plt.plot(t,y2,linewidth=2,color='r',label='12')
plt.legend()
plt.xticks(ticks = range(0,5))
plt.xlabel('Time (sec)')
plt.ylabel( '\u03BCV')
plt.rcParams["font.family"] = 'Arial'
plt.rcParams['font.size'] = "11"
plt.savefig('/media/ravinderjit/Data_Drive/Data/Figures/BindingPilot/TemporalFeat.png',format='png')







