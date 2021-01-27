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
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/CrossVisualAuditory_mseq/'
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

epoch_data_1 = mne.Epochs(data_eeg, data_evnt, [1], tmin=-2, tmax=10,reject=None, baseline=(-0.2, 0.)) 
evkd_data_1= epoch_data_1.average();
evkd_data_1.plot(picks = 31, titles = 'AMmseq')

epoch_data_2 = mne.Epochs(data_eeg, data_evnt, [2], tmin=-2, tmax=10,reject=None, baseline=(-0.2, 0.)) 
evkd_data_2= epoch_data_2.average();
evkd_data_2.plot(picks = 31, titles = 'AMmseq')




#%% Plot PSD

#%% Extract part of response when stim is on
t = epoch_data_1.times
t1 = np.where(t>=0)[0][0]
t2 = t1 + mseq.size
channel = 'A16'
Incoh = epoch_data_1.get_data(picks=channel)
Incoh = Incoh.T[:,0,:]
Incoh = Incoh[t1:t2,:]
t = t[t1:t2]

coh = epoch_data_2.get_data(picks=channel)
coh = coh.T[:,0,:]
coh = coh[t1:t2,:]



#%% Remove epochs with large deflections
Peak2Peak = Incoh.max(axis=0) - Incoh.min(axis=0)
Incoh = Incoh[:,Peak2Peak*1e6 < 100.]
 
Peak2Peak = coh.max(axis=0) - coh.min(axis=0)
coh = coh[:,Peak2Peak*1e6 < 100.]

TW = 10
Fres = (1/t[-1]) * TW * 2
fs = epoch_data_1.info['sfreq']

PLV_Incoh, Coh_Incoh, f, phase_Incoh = PLV_Coh(mseq,Incoh,TW,fs)
PLV_coh, Coh_coh, f, phase_Coh = PLV_Coh(mseq,coh,TW,fs)

# fig = plt.figure()
# plt.plot(f,Coh_Incoh,color='r',label='Incoherent')
# plt.plot(f,Coh_coh,color='b',label='Coherent')
# plt.title('Coh')
# plt.legend()

plt.figure()
plt.plot(f,PLV_Incoh,color='r',label='Incoherent')
plt.plot(f,PLV_coh,color='b',label='Coherent')
plt.title('PLV Ch. Cz')
plt.legend()

plt.figure()
plt.plot(f,phase_Incoh,color='r',label='Incoherent')
plt.plot(f,phase_Coh,color='b',label='Coherent')



#%% Noise floors
# Num_noiseFloors = 20
# Cohnf_AM = np.zeros([Coh_AM.shape[0],Num_noiseFloors])


# for nf in range(0,Num_noiseFloors):
#     print('NF:',nf+1,'/',Num_noiseFloors)
#     order_AM = np.random.permutation(AMdata.shape[1]-1)
#     Y_AM = AMdata[:,order_AM]
#     Y_AM[:,0:int(np.round(order_AM.size/2))] = -Y_AM[:,0:int(np.round(order_AM.size/2))]

#     PLVn_AM, Cohn_AM, f = PLV_Coh(mseq,Y_AM,TW,fs)

    
#     Cohnf_AM[:,nf] = Cohn_AM

    

# plt.figure()
# plt.plot(f,Cohnf_AM,color='grey')
# plt.plot(f,Coh_AM,color='k',linewidth=2)







