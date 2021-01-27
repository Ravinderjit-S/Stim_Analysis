#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:21:25 2020

@author: ravinderjit
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



def PLV_Coh(X,Y,TW,fs):
    """
    X is the Mseq
    Y is time x trials
    TW is half bandwidth product 
    """
    X = X.squeeze()
    ntaps = 2*TW - 1
    dpss = sp.signal.windows.dpss(X.size,TW,ntaps)
    N = int(2**np.ceil(np.log2(X.size)))
    f = np.arange(0,N)*fs/N
    PLV_taps = np.zeros([N,ntaps])
    Coh_taps = np.zeros([N,ntaps])
    Phase_taps = np.zeros([N,ntaps])
    for k in range(0,ntaps):
        print('tap:',k+1,'/',ntaps)
        Xf = sp.fft(X *dpss[k,:],axis=0,n=N)
        Yf = sp.fft(Y * dpss[k,:].reshape(dpss.shape[1],1),axis=0,n=N)
        XYf = Xf.reshape(Xf.shape[0],1) * Yf.conj()
        Phase_taps[:,k] = np.unwrap(np.angle(np.mean(XYf/abs(XYf),axis=1)))
        PLV_taps[:,k] = abs(np.mean(XYf / abs(XYf),axis=1))
        Coh_taps[:,k] = abs(np.mean(XYf,axis=1) / np.mean(abs(XYf),axis=1))
        
    PLV = PLV_taps.mean(axis=1)
    Coh = Coh_taps.mean(axis=1)
    Phase = Phase_taps.mean(axis=1)
    return PLV, Coh, f, Phase



nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  

Mseq_loc = '/media/ravinderjit/Storage2/EEGdata/Vision_mseq_4096.mat'
Mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/Visual_mseq/Vision_mseq_4096.mat'

Mseq_dat = sio.loadmat(Mseq_loc)
mseq = Mseq_dat['mseqEEG_4096']
mseq = mseq.astype(float)

data_loc = '/media/ravinderjit/Storage2/EEGdata/'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/Visual_mseq/'
subject = 'S211'

exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved


datapath =  os.path.join(data_loc, subject)
# datapath = '/media/ravinderjit/Data_Drive/Data/EEGdata/EFR'
data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=80)

#%% Blink Removal
blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
blink_proj = Projs[0]

data_eeg.add_proj(blink_proj)
data_eeg.plot_projs_topomap()
data_eeg.plot()

#%% Plot data

epoch_data_AM = mne.Epochs(data_eeg, data_evnt, [1], tmin=-2, tmax=10,reject=None, baseline=(-0.2, 0.)) 
evkd_data_AM= epoch_data_AM.average();
evkd_data_AM.plot(picks = 15, titles = 'AMmseq')


#%% Plot PSD

#%% Extract part of response when stim is on
t = epoch_data_AM.times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size
Vdata = epoch_data_AM.get_data(picks='A16')
Vdata2 = epoch_data_AM.get_data(picks='A32')
Vdata = Vdata.T[:,0,:]
Vdata = Vdata[t1:t2,:]
Vdata2 = Vdata2.T[:,0,:]
Vdata2 = Vdata2[t1:t2,:]

t = t[t1:t2]



#%% Remove epochs with large deflections
Peak2Peak = Vdata.max(axis=0) - Vdata.min(axis=0)
Vdata = Vdata[:,Peak2Peak*1e6 < 100.]

Peak2Peak = Vdata2.max(axis=0) - Vdata2.min(axis=0)
Vdata2 = Vdata2[:,Peak2Peak*1e6 < 100.]
 

TW = 10
Fres = (1/t[-1]) * TW * 2
fs = epoch_data_AM.info['sfreq']

PLV_V, Coh_V, f,phase_1 = PLV_Coh(mseq,Vdata,TW,fs)
PLV_V2, Coh_V2, f,phase_2 = PLV_Coh(mseq,Vdata2,TW,fs)

f5 = np.where(f>=5)[0][0]
f12 = np.where(f>=12)[0][0]
coeff = np.polyfit(f[f5:f12],phase_1[f5:f12],deg=1)
GD_line1 = coeff[0]*f[f5:f12] + coeff[1]
latency1 = -coeff[0] / (2*np.pi)


fig = plt.figure()
plt.plot(f,Coh_V,color='k',label='Ch. Oz')
plt.plot(f,Coh_V2,color='b',label='Ch. Cz')
plt.title('Coh Vision')
plt.legend()

plt.figure()
plt.plot(f,PLV_V,label='Ch. Oz')
plt.plot(f,PLV_V2,label='Ch. Cz')
plt.title('PLV Vision')
plt.legend()

plt.figure()
plt.plot(f,phase_1,label='Ch. Oz')
plt.plot(f,phase_2,label='Ch. Cz')
plt.plot(f[f5:f12],GD_line1)
plt.title('Phase')
plt.legend()


#%% Noise floors
# Num_noiseFloors = 20
# Cohnf_V = np.zeros([Coh_V.shape[0],Num_noiseFloors])
# PLVnf_V = np.zeros([Coh_V.shape[0],Num_noiseFloors])

# for nf in range(0,Num_noiseFloors):
#     print('NF:',nf+1,'/',Num_noiseFloors)
#     order_V = np.random.permutation(Vdata.shape[1]-1)
#     Y_V = Vdata[:,order_V]
#     Y_V[:,0:int(np.round(order_V.size/2))] = -Y_V[:,0:int(np.round(order_V.size/2))]

#     PLVn_V, Cohn_V, f = PLV_Coh(mseq,Y_V,TW,fs)

    
#     Cohnf_V[:,nf] = Cohn_V
#     PLVnf_V[:,nf] = PLVn_V

    

# plt.figure()
# plt.plot(f,Cohnf_V,color='grey')
# plt.plot(f,Coh_V,color='k',linewidth=2)
# plt.title('Coh Vision Ch. Oz')

# plt.figure()
# plt.plot(f,PLVnf_V,color='grey')
# plt.plot(f,PLV_V,color='k',linewidth=2)
# plt.title('PLV Vision Ch. Oz')







