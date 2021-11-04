#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:49:37 2021

@author: ravinderjit
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle


nchans = 34;
refchans = ['EXG1','EXG2']

Subjects = ['S211', 'S268', 'S269', 'S270', 'S273', 'S277','S279','S282']

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/Binding/'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Binding'
exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved
   
subject = Subjects[3]
datapath = os.path.join(data_loc,subject + '_Binding')

data_eeg,data_evnt = EEGconcatenateFolder(datapath+'/',nchans,refchans,exclude)
data_eeg.filter(l_freq=1,h_freq=40)

if subject == 'S273':
    data_eeg.info['bads'].append('A1')
    data_eeg.info['bads'].append('A30')
    data_eeg.info['bads'].append('A24')


#%% Remove Blinks

blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                              baseline=(-0.25,0),reject=dict(eeg=500e-6))
Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')

ocular_projs = [Projs[0]]

data_eeg.add_proj(ocular_projs)
data_eeg.plot_projs_topomap()
plt.savefig(os.path.join(fig_loc,'OcularProjs',subject + '_OcularProjs.png'),format='png')
data_eeg.plot(events=blinks,show_options=True)

#%% Add events for AB transitions at t = 1,2,3,4

data_evnt_AB = data_evnt.copy()
fs = data_eeg.info['sfreq']

for cnd in range(2):
    for e in range(4):
        evnt_num = 3 + e + cnd*4
        events_add = data_evnt[data_evnt[:,2] == int(cnd+1),:] + [int(fs*(e+1)),int(0),evnt_num - (cnd+1)]
        data_evnt_AB = np.concatenate((data_evnt_AB,events_add),axis=0)



#%% Plot Data

conds = ['12','20'] #14,18 for S211 from earlier date
reject = dict(eeg=150e-6)
epochs_whole = []
evkd_whole = []

for cnd in range(len(conds)):
    ep_cnd = mne.Epochs(data_eeg,data_evnt,cnd+1,tmin=-0.3,tmax=5.3,reject = reject, baseline = (-0.1,0.))
    epochs_whole.append(ep_cnd)
    evkd_whole.append(ep_cnd.average())
    evkd_whole[cnd].plot(picks=[31],titles=conds[cnd])
    
#%% Extract Different Conditions
    
conds = ['12_0', '20_0', '12_AB1', '12_BA1', '12_AB2', '12_BA2', '20_AB1','20_BA1','20_AB2','20_BA2']
    
epochs = []
evkd = []    
for cnd in range(10):
    ep_cnd = mne.Epochs(data_eeg,data_evnt_AB,cnd+1,tmin=-0.2,tmax=1.1, reject = reject, baseline = (-0.1,0.))
    epochs.append(ep_cnd)
    evkd.append(ep_cnd.average())
    #evkd[cnd].plot(picks=31,titles=conds[cnd])
        
conds.extend(['12AB', '12BA','20AB','20BA'])
ev_combos = [[2,4],[3,5],[6,8],[7,9]]

for it, cnd in enumerate(range(10,14)):
    ep_cnd = mne.Epochs(data_eeg,data_evnt_AB,list(np.array(ev_combos[it])+1),tmin=-0.2,tmax=1.1, reject = reject, baseline = (-0.1,0.))
    epochs.append(ep_cnd)
    evkd.append(ep_cnd.average())
    #evkd[cnd].plot(picks=31,titles=conds[cnd])
    
#%% Plot 1st and second interval

for it, c in enumerate(ev_combos):
    evkds = [evkd[c[0]], evkd[c[1]]]
    mne.viz.plot_compare_evokeds(evkds,picks=31,title = conds[it + 10])


#%% Plot Comparisons 

combos_comp = [[0,1], [10,12], [11,13]] 
comp_labels = ['Onset', 'Incoherent to Coherent', 'Coherent to Incoherent']

for it,c in enumerate(combos_comp):    
    evkds = [evkd[c[0]], evkd[c[1]]]
    mne.viz.plot_compare_evokeds(evkds,picks=31,title=comp_labels[it])


#%% Make Plots outside of MNE

fig, ax = plt.subplots(3,1,sharex=True)

t = epochs[0].times
for cnd in range(len(combos_comp)):
    cz_ep_12 = epochs[combos_comp[cnd][0]].get_data()[:,31,:]
    cz_mean_12 = cz_ep_12.mean(axis=0)
    cz_sem_12 = cz_ep_12.std(axis=0) / np.sqrt(cz_ep_12.shape[0])
    
    cz_ep_20 = epochs[combos_comp[cnd][1]].get_data()[:,31,:]
    cz_mean_20 = cz_ep_20.mean(axis=0)
    cz_sem_20 = cz_ep_20.std(axis=0) / np.sqrt(cz_ep_20.shape[0])
    
    ax[cnd].plot(t,cz_mean_12,label='12')
    ax[cnd].fill_between(t,cz_mean_12 - cz_sem_12, cz_mean_12 + cz_sem_12,alpha=0.5)
    
    ax[cnd].plot(t,cz_mean_20,label='20')
    ax[cnd].fill_between(t,cz_mean_20 - cz_sem_20, cz_mean_20 + cz_sem_20,alpha=0.5)
    
    ax[cnd].set_title(comp_labels[cnd])
    ax[cnd].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
ax[0].legend()
ax[2].set_xlabel('Time(sec)')
ax[2].set_ylabel('Amplitude \$uV')

plt.savefig(os.path.join(fig_loc,subject + '_12vs20.png'),format='png')


#%% Look at 32-channel response 








#%% Compute induced activity

# freqs = np.arange(1.,90.,1.)
# T = 1./5
# n_cycles = freqs*T
# time_bandwidth = 2
# vmin = -.15
# vmax = abs(vmin)
# bline = (-0.1,0)

# channels = np.arange(32)

# tfr_12 = mne.time_frequency.tfr_multitaper(epochs_whole[0].subtract_evoked(),
#                                            freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth,
#                                            return_itc = False,picks = channels,decim=4)#,average=False)

# tfr_20 = mne.time_frequency.tfr_multitaper(epochs_whole[1].subtract_evoked(),
#                                            freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth,
#                                            return_itc = False,picks = channels,decim=4)#,average=False)

# tfr_12.plot_topo(baseline =bline,mode= 'logratio', title = '12', vmin=vmin,vmax=vmax)

# tfr_20.plot_topo(baseline =bline,mode= 'logratio', title = '20', vmin=vmin,vmax=vmax)


#%% Save Epochs









