#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:09:32 2020

@author: ravinderjit
Average across tfr data from EEG data
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
import os
import pickle



def plot_chAvg_tfr(tfr_obj, picks, vmin,vmax,title):
# This function will average tfr data for the channels in picks. It references to baseline and does 10log10
#tfr_obj = mne tfr object
    

    tfr_data = tfr_obj.data[picks,:,:]
    t = tfr_obj.times
    b1 = int(np.argwhere(t>=-0.2)[0])
    b2 = int(np.argwhere(t>=0)[0])
    f = tfr_obj.freqs
    AvgCh = tfr_data.mean(axis=0)
    AvgCh = 10*np.log10(AvgCh/AvgCh[:,b1:b2].mean(axis=1).reshape(119,1))

    plt.figure()
    Z = AvgCh
    im = plt.pcolormesh(t,f,Z,vmin=vmin,vmax=vmax)
    axx = im.axes
    plt.colorbar(im)
   
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
    plt.title(title)
    plt.show()


# data from spring 20 pilot on binding
data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/Binding/BindingPilot_Spring20/Pickles')
subjects = ['S132','S227','S228','S230']
EEG_types = ['Active','Passive']

subjects=['S229']
EEG_type = EEG_types[1]


for m in range(0,len(subjects)):
    subject = subjects[m]
    with open(os.path.join(data_loc,subject+'_'+EEG_type+'_tfr.pickle'),'rb') as f:
        tfr_e1, tfr_e2, tfr_e3 = pickle.load(f)
    if m ==0:   # initialize first tfr object 
        tfr_e1_all = tfr_e1
        tfr_e2_all = tfr_e2
        tfr_e3_all = tfr_e3
    else:
        tfr_e1_all += tfr_e1
        tfr_e2_all += tfr_e2
        tfr_e3_all += tfr_e3
    
    


vmin = -0.1
vmax = 0.1
bline = (-0.2,0)
tfr_e1_all.plot_topo(baseline =bline,mode= 'logratio', title = 'e1_all_' + EEG_type, vmin=vmin,vmax=vmax)
tfr_e2_all.plot_topo(baseline =bline,mode= 'logratio', title = 'e2_all' + EEG_type, vmin=vmin,vmax=vmax)
tfr_e3_all.plot_topo(baseline =bline,mode= 'logratio', title = 'e3_all_' + EEG_type, vmin=vmin,vmax=vmax)

# vmin= 1e-9
# vmax= 5e-8
# fmin = 0
# fmax = 20
# tfr_e1_all.plot_topo(baseline = None,mode= 'mean', title = 'e1_all_' + EEG_type,vmin=vmin,vmax=vmax,fmin=fmin,fmax=fmax)
# tfr_e2_all.plot_topo(baseline = None,mode= 'mean', title = 'e2_all' + EEG_type,vmin=vmin,vmax=vmax,fmin=fmin,fmax=fmax)
# tfr_e3_all.plot_topo(baseline = None,mode= 'mean', title = 'e3_all_' + EEG_type,vmin=vmin,vmax=vmax,fmin=fmin,fmax=fmax)


freqs = np.arange(1.,120.,1.)
T = 1./5
n_cycles = freqs*T
time_bandwidth = 2

         
channels = np.arange(0,32)

for m in range(0,len(subjects)):
    subject = subjects[m]
    with open(os.path.join(data_loc,subject+'_'+EEG_type+'_epochs.pickle'),'rb') as f:
        epochs1, epochs2, epochs3 = pickle.load(f) 
    evoked1 = epochs1.average()
    evoked2 = epochs2.average()
    evoked3 = epochs3.average()
    tfr_e1_evkd = mne.time_frequency.tfr_multitaper(evoked1, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=4)
    tfr_e2_evkd = mne.time_frequency.tfr_multitaper(evoked2, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=4)
    tfr_e3_evkd = mne.time_frequency.tfr_multitaper(evoked3, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels,decim=4)
    
    # tfr_e1_evkd.plot_topo(baseline=(-0.3,0),mode='zlogratio',title='e1_evkd_' + EEG_type + '_'+subject,vmin=vmin,vmax=vmax)
    # tfr_e2_evkd.plot_topo(baseline=(-0.3,0),mode='zlogratio',title='e2_evkd_' + EEG_type + '_'+subject,vmin=vmin,vmax=vmax)
    # tfr_e3_evkd.plot_topo(baseline=(-0.3,0),mode='zlogratio',title='e3_evkd_' + EEG_type + '_'+subject,vmin=vmin,vmax=vmax)

    if m ==0:
        tfr_e1_evkd_all = tfr_e1_evkd
        tfr_e2_evkd_all = tfr_e2_evkd
        tfr_e3_evkd_all = tfr_e3_evkd
    else:
        tfr_e1_evkd_all += tfr_e1_evkd
        tfr_e2_evkd_all += tfr_e2_evkd
        tfr_e3_evkd_all += tfr_e3_evkd
        
    #del tfr_e1_evkd, tfr_e2_evkd, tfr_e3_evkd, evoked1, evoked2,evoked3
    
        
vmin = -1
vmax = 1
tfr_e1_evkd_all.plot_topo(baseline=bline,mode='logratio',title='e1_evkd_All' + EEG_type,vmin=vmin,vmax=vmax)
tfr_e2_evkd_all.plot_topo(baseline=bline,mode='logratio',title='e2_evkd_All' + EEG_type,vmin=vmin,vmax=vmax)
tfr_e3_evkd_all.plot_topo(baseline=bline,mode='logratio',title='e3_evkd_All' + EEG_type,vmin=vmin,vmax=vmax)




# subject= 'S132'
# with open(os.path.join(data_loc,subject+'_'+EEG_type+'_epochs.pickle'),'rb') as f:
#     epochs1, epochs2, epochs3 = pickle.load(f)

# datae1 = epochs3.get_data()
# datae1= datae1*1e6
# ch = 31
# n_epochs = range(0,datae1.shape[0])
# t = epochs1.times
# plt.figure()
# plt.plot(t,datae1[n_epochs,ch,:].T)
# plt.plot(t,datae1[n_epochs,ch,:].mean(axis=0),lw=4,color='k')
# plt.ylabel('uV')
# plt.xlabel('Time (sec)')





picks = [4,25,30,31]
vmin = -8
vmax = 8
plot_chAvg_tfr(tfr_e1_evkd_all,picks,vmin,vmax,title= subject+'_e1_evkd_' +EEG_type)
plot_chAvg_tfr(tfr_e2_evkd_all,picks,vmin,vmax,title= subject+'_e2_evkd_' +EEG_type)
plot_chAvg_tfr(tfr_e3_evkd_all,picks,vmin,vmax,title= subject+'_e3_evkd_' +EEG_type)

vmin = -2
vmax = 2
plot_chAvg_tfr(tfr_e1_all,picks,vmin,vmax,title= subject+'_e1_induced_' +EEG_type)
plot_chAvg_tfr(tfr_e2_all,picks,vmin,vmax,title= subject+'_e2_induced_' +EEG_type)
plot_chAvg_tfr(tfr_e3_all,picks,vmin,vmax,title= subject+'_e3_induced_' +EEG_type)







