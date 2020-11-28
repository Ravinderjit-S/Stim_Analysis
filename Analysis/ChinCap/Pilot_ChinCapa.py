# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:28:27 2020

@author: StuffDeveloping
"""


import numpy as np
import matplotlib.pyplot as plt
from EEGpp import EEGconcatenateFolder
import mne
from anlffr import spectral
import os
from scipy import linalg



#data_loc = r'H:\ChinCap\090720\'
data_loc = r'H:\ChinCap\082020\\'
data_loc = '/home/ravinderjit/Documents/ChinCapData/092320/'
stim_type = 'tone_4k_' #'click_'  tone_4k_
pathThing = '/'
nchans = 34
refchans = ['A1', 'A2']
refchans = ['EXG1']
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']
data_eeg,data_evnt = EEGconcatenateFolder(data_loc + stim_type +'80' + pathThing ,nchans,refchans,exclude)
data_eeg.filter(300,3000)
#data_eeg.notch_filter(60)
scalings = dict(eeg=20e-6,stim=1)
#data_eeg.plot(events = data_evnt, scalings=scalings,show_options=True)
data_eeg.plot_psd(picks =[31,32,33])


epochs = mne.Epochs(data_eeg,data_evnt,[255],tmin=-0.005,tmax=0.01)
evoked_80 = epochs.average()

channels = list(range(24))
channels.extend([29,30,31])
channels.remove(21)
channels.remove(20)
channels.remove(0)
evoked_80.plot(picks=channels, titles ='80')


data_eeg,data_evnt = EEGconcatenateFolder(data_loc + stim_type+ '70' + pathThing ,nchans,refchans,exclude)
data_eeg.filter(300,3000)
epochs = mne.Epochs(data_eeg,data_evnt,[255],tmin=-0.005,tmax=0.01)
evoked_70 = epochs.average()
evoked_70.plot(picks=channels, titles = '70')

data_eeg,data_evnt = EEGconcatenateFolder(data_loc + stim_type + '60' + pathThing ,nchans,refchans,exclude)
data_eeg.filter(300,3000)
epochs = mne.Epochs(data_eeg,data_evnt,[255],tmin=-0.005,tmax=0.01)
evoked_60 = epochs.average()
evoked_60.plot(picks=channels, titles = '60')

data_eeg,data_evnt = EEGconcatenateFolder(data_loc + stim_type + '50' + pathThing ,nchans,refchans,exclude)
data_eeg.filter(300,3000)
epochs = mne.Epochs(data_eeg,data_evnt,[255],tmin=-0.005,tmax=0.01)
evoked_50 = epochs.average()
evoked_50.plot(picks=channels, titles = '50')

data_eeg,data_evnt = EEGconcatenateFolder(data_loc + stim_type +'40' + pathThing ,nchans,refchans,exclude)
data_eeg.filter(300,3000)
epochs = mne.Epochs(data_eeg,data_evnt,[255],tmin=-0.005,tmax=0.01)
evoked_40 = epochs.average()
evoked_40.plot(picks=channels, titles = '40')

data_eeg,data_evnt = EEGconcatenateFolder(data_loc + stim_type +'30' + pathThing ,nchans,refchans,exclude)
data_eeg.filter(300,3000)
epochs = mne.Epochs(data_eeg,data_evnt,[255],tmin=-0.005,tmax=0.01)
evoked_30 = epochs.average()
evoked_30.plot(picks=channels, titles = '30')

data_eeg,data_evnt = EEGconcatenateFolder(data_loc + stim_type +'20' + pathThing ,nchans,refchans,exclude)
data_eeg.filter(300,3000)
epochs = mne.Epochs(data_eeg,data_evnt,[255],tmin=-0.005,tmax=0.01)
evoked_20 = epochs.average()
evoked_20.plot(picks=channels, titles = '20')

data_eeg,data_evnt = EEGconcatenateFolder(data_loc + stim_type +'10' + pathThing,nchans,refchans,exclude)
data_eeg.filter(300,3000)
epochs = mne.Epochs(data_eeg,data_evnt,[255],tmin=-0.005,tmax=0.01)
evoked_10 = epochs.average()
evoked_10.plot(picks=channels, titles = '10')

data_eeg,data_evnt = EEGconcatenateFolder(data_loc + stim_type + '0' + pathThing ,nchans,refchans,exclude)
data_eeg.filter(300,3000)
epochs = mne.Epochs(data_eeg,data_evnt,[255],tmin=-0.005,tmax=0.01)
evoked_0 = epochs.average()
evoked_0.plot(picks=channels, titles = '0')


#get data out of structure
fs = evoked_80.info['sfreq']
t = np.arange(-0.005,0.01 + 1/fs, 1/fs)

click_0 = evoked_0.data * 1e6
click_10 = evoked_10.data * 1e6
click_20 = evoked_20.data * 1e6
click_30 = evoked_30.data * 1e6
click_40 = evoked_40.data * 1e6
click_50 = evoked_50.data * 1e6
click_60 = evoked_60.data * 1e6
click_70 = evoked_70.data * 1e6
click_80 = evoked_80.data * 1e6

p2p_0 = np.zeros(32)
p2p_10 = np.zeros(32)
p2p_20 = np.zeros(32)
p2p_30 = np.zeros(32)
p2p_40 = np.zeros(32)
p2p_50 = np.zeros(32)
p2p_60 = np.zeros(32)
p2p_70 = np.zeros(32)
p2p_80 = np.zeros(32)


cmap = plt.get_cmap('hot')
for ch in range(32):
    plt.figure()

    plt.plot(t,click_0[ch,:],label='0',color=cmap(0.9))
    plt.plot(t,click_10[ch,:],label='10',color=cmap(0.8))
    plt.plot(t,click_20[ch,:],label='20',color=cmap(0.7))
    plt.plot(t,click_30[ch,:],label='30',color=cmap(0.6))
    plt.plot(t,click_40[ch,:],label='40',color=cmap(0.5))
    plt.plot(t,click_50[ch,:],label='50',color=cmap(0.4))
    plt.plot(t,click_60[ch,:],label='60',color=cmap(0.3))
    plt.plot(t,click_70[ch,:],label='70',color=cmap(0.2))
    plt.plot(t,click_80[ch,:],label='80',color=cmap(0.1))
    plt.ylim([-1.2,1])
    plt.legend(loc =2)
    plt.title(str(ch))
    
    p2p_0[ch] = click_0[ch,:].max() - click_0[ch,:].min()
    p2p_10[ch] = click_10[ch,:].max() - click_10[ch,:].min()
    p2p_20[ch] = click_20[ch,:].max() - click_20[ch,:].min()
    p2p_30[ch] = click_30[ch,:].max() - click_30[ch,:].min()
    p2p_40[ch] = click_40[ch,:].max() - click_40[ch,:].min()
    p2p_50[ch] = click_50[ch,:].max() - click_50[ch,:].min()
    p2p_60[ch] = click_60[ch,:].max() - click_60[ch,:].min()
    p2p_70[ch] = click_70[ch,:].max() - click_70[ch,:].min()
    p2p_80[ch] = click_80[ch,:].max() - click_80[ch,:].min()


plt.figure()
plt.plot(range(32),p2p_0,label='0',color=cmap(0.9))
plt.plot(range(32),p2p_10,label='10',color=cmap(0.8))
plt.plot(range(32),p2p_20,label='20',color=cmap(0.7))
plt.plot(range(32),p2p_30,label='30',color=cmap(0.6))
plt.plot(range(32),p2p_40,label='40',color=cmap(0.5))
plt.plot(range(32),p2p_50,label='50',color=cmap(0.4))
plt.plot(range(32),p2p_60,label='60',color=cmap(0.3))
plt.plot(range(32),p2p_70,label='70',color=cmap(0.2))
plt.plot(range(32),p2p_80,label='80',color=cmap(0.1))
plt.legend(loc=2)
plt.title('Overall P2p')
plt.xlabel('Channel')
plt.ylabel('P2P')


# x = epochs.get_data()
# x = x[:,0:32,:]
# params = dict()
# params['Fs'] = epochs.info['sfreq']

x_ave = click_60
x_ave = np.delete(x_ave,[1,23,24,25,26,27,32,33],axis=0)
C_td = np.cov(x_ave)
vals, vecs = linalg.eigh(C_td, eigvals_only=False)
y_pc = np.dot(vecs[:, -1], x_ave) / (vecs[:, -1].sum())

plt.figure()
#plt.plot(vecs)
plt.plot(vecs[:,-1] / (vecs[:,-1].sum()),color='k',linewidth=1)


plt.figure()
#plt.plot(t,x_ave[6,:],color='r')
plt.plot(t,x_ave.mean(axis=0),color='k')
plt.plot(t,y_pc,color='r')
plt.legend(['avg_chs','pca'])


#y_cpc, y_pc = spectral.mtcpca_timeDomain(x, params)



















