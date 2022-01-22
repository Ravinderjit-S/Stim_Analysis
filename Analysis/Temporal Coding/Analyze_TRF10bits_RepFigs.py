#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:27:57 2021

@author: ravinderjit

Make Figures for mod-TRF exploring repeatability, frequency response, looking at sources etc.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:15:54 2021

@author: ravinderjit
Investigate repeatability of "ACR"
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
import scipy.io as sio
from scipy.signal import find_peaks

import sys
sys.path.append(os.path.abspath('../ACRanalysis/'))
from ACR_helperFuncs import ACR_sourceHf
from ACR_helperFuncs import Template_tcuts
from ACR_helperFuncs import PCA_tcuts
from ACR_helperFuncs import PCA_tcuts_topomap

fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/')


#%% Load mseq
mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)


#%% Load Template
template_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/Pickles/PCA_passive_template.pickle'

with open(template_loc,'rb') as file:
    [pca_coeffs_cuts,pca_expVar_cuts,t_cuts] = pickle.load(file)
    
bstemTemplate  = pca_coeffs_cuts[0][0,:]
cortexTemplate = pca_coeffs_cuts[2][0,:]

# vmin = bstemTemplate.mean() - 2 * bstemTemplate.std()
# vmax = bstemTemplate.mean() + 2 * bstemTemplate.std()
# plt.figure()
# mne.viz.plot_topomap(cortexTemplate, mne.pick_info(info_obj,np.arange(32)),vmin=vmin,vmax=vmax)
        



#%% Load data collected first

data_loc_old = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
pickle_loc_old = data_loc_old + 'Pickles/'
Subjects = ['S207','S228','S236','S238'] 

A_Tot_trials_old = []
A_Ht_old = []
A_Htnf_old = []
A_info_obj_old = []
A_ch_picks_old = []

A_Ht_old_epochs = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc_old,subject+'_AMmseqbits4.pickle'),'rb') as file:
        [tdat, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_old.append(Tot_trials[3])
    A_Ht_old.append(Ht[3])
    A_Htnf_old.append(Htnf[3])
    
    A_info_obj_old.append(info_obj)
    A_ch_picks_old.append(ch_picks)
    
    with open(os.path.join(pickle_loc_old,subject+'_AMmseqbits4_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        A_Ht_old_epochs.append(Ht_epochs[3])
    
t = tdat[3]


#%% Add first run with 10bit stim
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc = data_loc + 'Pickles/'

Subjects.append('S250')
subject = 'S250'
with open(os.path.join(pickle_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
    [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_old.append(Tot_trials)
    A_Ht_old.append(Ht)
    A_Htnf_old.append(Htnf)
    
    A_info_obj_old.append(info_obj)
    A_ch_picks_old.append(ch_picks)
    
with open(os.path.join(pickle_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
    [Ht_epochs, t_epochs] = pickle.load(file)
    A_Ht_old_epochs.append(Ht_epochs)


print('Done loading 1st visit ...')
#%% Load Second Run of data collection
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc = data_loc + 'Pickles/'

A_Tot_trials = []
A_Ht = []
A_Htnf = []
A_info_obj = []
A_ch_picks = []

A_Ht_epochs = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    if subject == 'S250':
        subject = 'S250_visit2'
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    A_Htnf.append(Htnf)
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
    
    A_Ht_epochs.append(Ht_epochs)
    

print('Done loading 2nd visit ...')



#%% Example CZ
sub = 0
ch_cz = np.where(A_ch_picks[sub]==31)[0][0]
cz = A_Ht[sub][ch_cz,:]

plt.figure()
plt.plot(t*1000,cz, color='k',linewidth = 2)
plt.xlim([-100,500])
#plt.xticks([7.3,29, 47, 94, 201, 500, 1000],labels=['7.3','29','47','94','201','500','1000'])
plt.xlabel('Time (msec)', fontsize=12)
plt.ylabel('Amplitude',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('mod-TRF Ch. Cz',fontsize=14)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
#plt.xscale('log')

#%% Manual Peak Cutoff Reading CZ and FP
    
cuts_tms = []
#S207
cuts_tms.append([.0185, .036, .067, .136 ,.266])

#S228
cuts_tms.append([.016, .043, .069, .125, .238])

#S236
cuts_tms.append([.014, .031, .066, .124, .249])

#238
cuts_tms.append([.0155, .045, .062, .124, .287])

#S250
cuts_tms.append([.016, .049, .123, .220, .334])
      

#%% Look at epochs: 1st visit and 2nd Visit

# fig = plt.figure()
# ax = [None] *5
# ax[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
# ax[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
# ax[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
# ax[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
# ax[4] = plt.subplot2grid((2,6), (1,3), colspan=2)

fs = 4096
fig,ax = plt.subplots(6,2)
fig.set_size_inches(10,10)

t_0 = np.where(t_epochs>=0)[0][0]
colors = ['tab:blue','tab:orange','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']


for sub in np.arange(len(Subjects)):
    #Plot Cz
    
    if sub <5:
        ax[sub,0].axes.xaxis.set_visible(False)
        ax[sub,1].axes.xaxis.set_visible(False)
        
    Ht_mean_old = A_Ht_old_epochs[sub].mean(axis=1) 
    Ht_mean = A_Ht_epochs[sub].mean(axis=1)
    
    Ht_sem_old = A_Ht_old_epochs[sub].std(axis=1) / np.sqrt(A_Ht_old_epochs[sub].shape[1])
    Ht_sem = A_Ht_epochs[sub].std(axis=1) / np.sqrt(A_Ht_epochs[sub].shape[1])
    
    cz_mean_v2 = Ht_mean[-1,:] - Ht_mean[-1,t_0] #Look at cz. Make time 0 start at 0
    
    ax[sub,0].plot(t_epochs, cz_mean_v2,color='grey', label='2nd Visit')
    ax[sub,0].fill_between(t_epochs,cz_mean_v2 -Ht_sem[-1,:],cz_mean_v2 + Ht_sem[-1,:], color='grey',alpha=0.4)
    
    cz_mean_v1 = Ht_mean_old[-1,:] - Ht_mean_old[-1,t_0]
    
    ax[sub,0].plot(t_epochs, cz_mean_v1,color='k',label = '1st Visit')
    #ax[sub,0].fill_between(t_epochs,cz_mean_v1 -Ht_sem_old[-1,:],cz_mean_v1 + Ht_sem_old[-1,:], color='k',alpha=0.5)
    
    t_cuts = cuts_tms[sub]
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t_epochs>=0)[0][0]
        else:
            t_1 = np.where(t_epochs>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t_epochs>=t_cuts[t_c])[0][0]
        
        #ax[sub,0].plot(t_epochs[t_1:t_2], cz_mean_v1[t_1:t_2],color=colors[t_c])
        ax[sub,0].fill_between(t_epochs[t_1:t_2],cz_mean_v1[t_1:t_2] -Ht_sem_old[-1,t_1:t_2],cz_mean_v1[t_1:t_2] + Ht_sem_old[-1,t_1:t_2], color=colors[t_c],alpha=0.8)
    
    
    ax[sub,0].set_xlim([-0.05,0.5])
    ax[sub,0].set_title('Participant ' + str(sub+1),fontweight='bold')
    ax[sub,0].set_xticks([0,0.1,0.2,0.3,0.4])
    ax[sub,0].set_yticks([-.002,0,.002])
    ax[sub,0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    
    t_05 = np.where(t_epochs>=0.5)[0][0]
    cz_mean_v2 = cz_mean_v2[t_0:t_05]
    cz_mean_v2 = cz_mean_v2 - cz_mean_v2.mean()
    Ht_freq_v2 = np.fft.fft(cz_mean_v2[:])
    f = np.fft.fftfreq(Ht_freq_v2.size,d=1/fs)
    
    Ht_freq_v2 = Ht_freq_v2[f>=0]
    f = f[f>=0]
    
    phase_v2 = np.unwrap(np.angle(Ht_freq_v2))
    ax2 = ax[sub,1].twinx()
    
    l1, = ax[sub,1].plot(f,np.abs(Ht_freq_v2),color='k',label='Whole')
    l2, = ax2.plot(f,phase_v2,color='k',linestyle='--',label='Whole Phase')
    ax[sub,1].set_xlim([0,75])
    ax[sub,1].set_yticks([0,0.5])
    ax[sub,1].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    ax2.axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax2.set_ylim([-25,0])
    ax2.set_yticks([-10,0])
    
    
    
    if sub == 0:
        lines = [l1,l2]
        ax[0,1].legend(lines,[l.get_label() for l in lines],fontsize=9)
    
    t_cuts = cuts_tms[sub]
    for t_c in range(len(cuts_tms)):
        if t_c ==0:
            t_1 = np.where(t_epochs>=0)[0][0]
        else:
            t_1 = np.where(t_epochs>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t_epochs>=t_cuts[t_c])[0][0]
        
        t_ep = t_epochs[t_1:t_2]
        
        Ht_freq = np.abs(np.fft.fft(A_Ht_epochs[sub][-1,:,t_1:t_2] - A_Ht_epochs[sub][-1,:,t_1:t_2].mean(axis=-1)[:,np.newaxis]))
        Ht_freq_mean = Ht_freq.mean(axis=0)
        Ht_freq_sem = Ht_freq.std(axis=0) / np.sqrt(Ht_freq.shape[0])     
                   
        f_t = np.fft.fftfreq(Ht_freq_mean.size,d=1/fs)


        ax[sub,1].plot(f_t,Ht_freq_mean,color=colors[t_c])
        ax[sub,1].fill_between(f_t,Ht_freq_mean-Ht_freq_sem,Ht_freq_mean+Ht_freq_sem,color=colors[t_c],alpha=0.5)

    
    
    
    




ax[5,0].plot(t_epochs, cz_mean_v1,color='k') # replace with participant 6 later
ax[5,0].set_xlim([-0.05,0.5])

ax[5,0].set_xlabel('Time (sec)')
ax[5,0].set_ylabel('Amplitude')
ax[5,0].set_yticks([-.002,0,.002,.004])
ax[5,0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[0,0].legend(fontsize=9)

ax[5,1].plot(f,np.abs(Ht_freq_v2),color='k',linestyle='--') # replace with participant 6 later
ax[5,1].set_xlim([0,75])

ax[5,1].set_xlabel('Frequency (Hz)')
ax[5,1].set_ylabel('Magnitude')
ax[5,1].set_yticks([0, 0.5])
ax[5,1].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

plt.savefig(os.path.join(fig_path,'ModTRF_rep.svg'),format='svg')
plt.savefig(os.path.join(fig_path,'ModTRF_rep.eps'),format='eps')


#fig.suptitle('Ch. Cz',fontweight='bold')

#%% Look at epochs with t cuts and in freq domain

colors = ['tab:blue','tab:orange','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']

fs = 4096
fig = plt.figure()
ax = [None] *5
ax[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
ax[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
ax[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
ax[4] = plt.subplot2grid((2,6), (1,3), colspan=2)

fig_f = plt.figure()
axf = [None] *5
axf[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
axf[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
axf[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
axf[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
axf[4] = plt.subplot2grid((2,6), (1,3), colspan=2)

for sub in np.arange(len(Subjects)):
    #Plot Cz
    
    if sub > 0:
        ax[sub].axes.yaxis.set_visible(False)
        axf[sub].axes.yaxis.set_visible(False)
        
    Ht_mean = A_Ht_epochs[sub].mean(axis=1)
    Ht_sem = A_Ht_epochs[sub].std(axis=1) / np.sqrt(A_Ht_epochs[sub].shape[1])
    

    t_cuts = cuts_tms[sub]
    for t_c in range(len(cuts_tms)):
        if t_c ==0:
            t_1 = np.where(t_epochs>=0)[0][0]
        else:
            t_1 = np.where(t_epochs>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t_epochs>=t_cuts[t_c])[0][0]
        
        t_ep = t_epochs[t_1:t_2]
        Ht_mean_tc = Ht_mean[-1,t_1:t_2] #- Ht_mean[-1,t_1:t_2].mean()
        
        Ht_freq = np.abs(np.fft.fft(A_Ht_epochs[sub][-1,:,t_1:t_2] - A_Ht_epochs[sub][-1,:,t_1:t_2].mean(axis=-1)[:,np.newaxis]))
        Ht_freq_mean = Ht_freq.mean(axis=0)
        Ht_freq_sem = Ht_freq.std(axis=0) / np.sqrt(Ht_freq.shape[0])     
                   
        f = np.fft.fftfreq(Ht_freq_mean.size,d=1/fs)

        ax[sub].plot(t_ep, Ht_mean_tc, color=colors[t_c])
        ax[sub].fill_between(t_ep,Ht_mean_tc -Ht_sem[-1,t_1:t_2],Ht_mean_tc + Ht_sem[-1,t_1:t_2], color=colors[t_c],alpha=0.5)
        ax[sub].set_title('Subject ' + str(sub+1))
        ax[sub].set_xticks([0,0.1,0.2,0.3,0.4])
        
        axf[sub].plot(f,Ht_freq_mean,color=colors[t_c])
        axf[sub].fill_between(f,Ht_freq_mean-Ht_freq_sem,Ht_freq_mean+Ht_freq_sem,color=colors[t_c],alpha=0.5)
        axf[sub].set_title('Subject ' + str(sub+1))
        axf[sub].set_xlim([0,75])
        axf[sub].set_xticks([10,25,50])
        


ax[0].set_xlabel('Time (sec)')
ax[0].set_ylabel('Amplitude')
ax[0].set_yticks([-.002,0,.002,.004])
ax[0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
#ax[0].legend(fontsize=9)
#fig.suptitle('Ch. Cz',fontweight='bold')

axf[0].set_xlabel('Frequency (Hz)')
axf[0].set_ylabel('Magnitude')
axf[0].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))




