# -*- coding: utf-8 -*-
"""
Created on Mon Apr 1,2019

@author: StuffDeveloping
Analyze Binding Pilot data
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


nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
    
direct_ = '/media/ravinderjit/Data_Drive/EEGdata/Binding/'


exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved



data_eeg,data_evnt = EEGconcatenateFolder(direct_+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=130)


## blink removal
blinks_eeg = find_blinks(data_eeg, ch_name = ['A1'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
scalings = dict(eeg=40e-6,stim=0.1)

blink_epochs = mne.Epochs(data_eeg,blinks_eeg,998,tmin=-0.25,tmax=0.25,proj=False,
                          baseline=(-0.25,0),reject=dict(eeg=500e-6))

Projs_data = compute_proj_epochs(blink_epochs, n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

#data_eeg.add_proj(Projs_data)   
#data_eeg.plot_projs_topomap()
  
#if Subject == 'Rav':                     
eye_projs = [Projs_data[0],Projs_data[2]]


    
data_eeg.add_proj(eye_projs)
data_eeg.plot_projs_topomap()
data_eeg.plot(events=data_evnt,scalings=scalings,show_options=True,title = 'BindingData')

channels = [31,4,26,25,30]
ylim_vals = [-3.5,3]

epochsAll = mne.Epochs(data_eeg, data_evnt, [1, 2, 3,4,5], tmin=-0.4, tmax=5.0, proj=True,reject=dict(eeg=200e-6)) #, baseline=(-0.2, 0.)) 
evokedAll = epochsAll.average()
evokedAll.plot(picks=channels,titles ='BindingAll_evoked auditory chan')   #ylim = dict(eeg=ylim_vals))   

evokedAll.plot(picks=[13,14,15],titles ='visual channels')

times = np.arange(-0.2,.4,0.05)
evokedAll.plot_topomap(times,ch_type='eeg',time_unit='s')


plt.figure()
t = np.arange(-0.4,5,1./4096)
plt.plot(t,evokedAll.data[31,:],c='b',label='Aud')
plt.plot(t,evokedAll.data[14,:],c='r',label='Vis')
plt.legend()
plt.xlabel('Time (s)')


epochsAll_noProj = mne.Epochs(data_eeg, data_evnt, [1, 2, 3,4,5], tmin=-0.4, tmax=5.0, proj=False,reject=dict(eeg=200e-6)) 
evoked_noProj = epochsAll_noProj.average()
evoked_noProj.plot(picks=channels,titles = 'Binding Proj off')

evk = evokedAll.data[31,:]
evk_np = evoked_noProj.data[31,:]
plt.figure()
t = np.arange(0,evk.size/4096.,1./4096.)
plt.plot(t,evk,c='b')
plt.plot(t,evk_np,c='r')



STI_chan = data_eeg.get_data()[34,:] #subtracting 65280 b/c there is this huge offset
STI_chan = np.mod(STI_chan,256)
t = np.arange(0,STI_chan.size/4096.,1./4096.)
plt.figure()
plt.plot(t,STI_chan,c='b')
plt.plot(data_evnt[:,0]/4096., STI_chan[data_evnt[:,0]]+np.ones(data_evnt[:,0].size)*50.,c='m',marker='*',linestyle='None')


#mod(val,256)

Proj_OnOFF = True

epochs_e1 = mne.Epochs(data_eeg, data_evnt, [1], tmin=-0.4, tmax=1.0, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e1 = epochs_e1.average()

epochs_e2 = mne.Epochs(data_eeg, data_evnt, [2], tmin=-0.4, tmax=1.0, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e2 = epochs_e2.average()

epochs_e3 = mne.Epochs(data_eeg, data_evnt, [3], tmin=-0.4, tmax=1.0, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e3 = epochs_e3.average()

epochs_e4 = mne.Epochs(data_eeg, data_evnt, [4], tmin=-0.4, tmax=1.0, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e4 = epochs_e4.average()

epochs_e5 = mne.Epochs(data_eeg, data_evnt, [5], tmin=-0.4, tmax=1.0, proj=Proj_OnOFF,reject=dict(eeg=200e-6)) 
evoked_e5 = epochs_e5.average()


#plots
evoked_e1.plot(picks=channels,titles = '1')
evoked_e2.plot(picks=channels,titles = '2')
evoked_e3.plot(picks=channels,titles = '3')
evoked_e4.plot(picks=channels,titles = '4')
evoked_e5.plot(picks=channels,titles = '5')


E1epo = epochs_e1.subtract_evoked().get_data()[:,30,:]
E5epo = epochs_e5.subtract_evoked().get_data()[:,30,:]


E1_fft = np.abs(sp.fft(E1epo,axis=1))
E5_fft = np.abs(sp.fft(E5epo,axis=1))

fs = data_eeg.info['sfreq']
N = E1_fft.shape[1]
f =np.arange(0,N)*fs/N
if np.mod(N,2) == 0:
    half_index = N/2;
else:
    half_index = (N-1)/2; 

f = f[:half_index]
E1_fft = E1_fft[:,:half_index].mean(axis=0)
E5_fft = E5_fft[:,:half_index].mean(axis=0)

plt.figure()
plt.plot(f,20*np.log10(E1_fft),c='b')
plt.plot(f,20*np.log10(E5_fft),c='r')



freqs = np.arange(1.,130.,1.)
n_cycles = freqs/10
time_bandwidth = 2.0
vmin = -4
vmax = 4

power_e1 = mne.time_frequency.tfr_multitaper(epochs_e1, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e1.plot_topo(baseline =(-0.4,0),mode= 'zlogratio', title = 'e1', vmin=vmin,vmax=vmax)

power_e2 = mne.time_frequency.tfr_multitaper(epochs_e2, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e2.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e2', vmin=vmin,vmax=vmax)

power_e3 = mne.time_frequency.tfr_multitaper(epochs_e3, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e3.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e3', vmin=vmin,vmax=vmax)

power_e4 = mne.time_frequency.tfr_multitaper(epochs_e4, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e4.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e4', vmin=vmin,vmax=vmax)

power_e5 = mne.time_frequency.tfr_multitaper(epochs_e5, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e5.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e5', vmin=vmin,vmax=vmax)



channels = [30,31]

power_e1_in = mne.time_frequency.tfr_multitaper(epochs_e1.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e1_in.plot_topo(baseline =(-0.4,0),mode= 'zlogratio', title = 'e1_induced', vmin=vmin,vmax=vmax)

power_e2_in = mne.time_frequency.tfr_multitaper(epochs_e2.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e2_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e2_induced', vmin=vmin,vmax=vmax)

power_e3_in = mne.time_frequency.tfr_multitaper(epochs_e3.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e3_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e3_induced', vmin=vmin,vmax=vmax)

power_e4_in = mne.time_frequency.tfr_multitaper(epochs_e4.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e4_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e4_induced', vmin=vmin,vmax=vmax)

power_e5_in = mne.time_frequency.tfr_multitaper(epochs_e5.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False,picks = channels)
power_e5_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e5_induced', vmin=vmin,vmax=vmax)
