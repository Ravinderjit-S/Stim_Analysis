#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:37:39 2021

@author: ravinderjit
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

sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr
from mseqHelper import mseqXcorrEpochs


data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc = data_loc + 'Pickles/'

A_Tot_trials = []
A_Ht = []
A_Htnf = []
A_info_obj = []
A_ch_picks = []

Subjects = ['S207','S228','S236','S238','S239','S246']

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
    

print('Done loading mTRF ...')

#%% Load SAM noise measures
pickle_loc_tmtf = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/tmtf/Pickles'

A_Trials_cond = []
A_freqs = []
A_tmtf_mag = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc_tmtf,subject +'_tmtf.pickle'),'rb') as file:
        [freqs, tmtf_mag, Trials_cond] = pickle.load(file)
    
    A_Trials_cond.append(Trials_cond)
    A_freqs.append(freqs)
    A_tmtf_mag.append(tmtf_mag)
    
#%% t cuts
    
cuts_passive = []
cuts_count = []
cuts_sd = []

#S207
cuts_passive.append([.0185, .036, .067, .121 ,.266])
#S228
cuts_passive.append([.016, .043, .069, .125, .253])
#S236
cuts_passive.append([.014, .031, .066, .124, .249])
#238
cuts_passive.append([.0155, .045, .067, .124, .287])
#239
cuts_passive.append([.016, .036, .064, .119, .241])
#246
cuts_passive.append([.016, .030, .062, .130, .258])
#247
#cuts_passive.append([.018, .042, .071, .120, .220, 0.5, 0.75 ])
#250
#cuts_passive.append([.016, .049, .123, .220, .320, 0.5, 0.75])


#%% plot tMTF and mTRF
colors = ['tab:blue','tab:orange','green','red','purple', 'brown', 'pink']
fs = 4096

plt.rcParams.update({'font.size':15})
for sub in range(len(Subjects)):
    ch_num = 31
    ch_ind = np.where(A_ch_picks[sub] == 31)[0][0]
    cz = A_Ht[sub][ch_ind,:]
    
    t_cuts = cuts_passive[sub]
    
    fig,axs = plt.subplots(2,1)
    for tc in range(len(t_cuts)):
        if tc ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[tc-1])[0][0]
            
        t_2 = np.where(t>=t_cuts[tc])[0][0]
        
        [w_tc, Hf_tc] = freqz(b=cz[t_1:t_2] - cz[t_1:t_2].mean(),a=1,worN=np.arange(0,fs/2,2),fs=fs) 
        
        axs[0].plot(t[t_1:t_2]*1000,cz[t_1:t_2],color=colors[tc])
        axs[0].set_xlabel('Time (ms)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title(Subjects[sub] + ' mTRF')
        axs[0].axes.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
        
        axs[1].plot(w_tc,np.abs(Hf_tc), color=colors[tc])
        axs[1].set_xlim([0,75])
        axs[1].set_xlabel('Frequency')
        axs[1].set_ylabel('Magnitude')
        axs[1].set_title('tMTF',y=0.8)
        
    t_1 = np.where(t>=0)[0][0]
    t_2 = np.where(t>=t_cuts[-1])[0][0]
    [w, Hf] = freqz(b=cz[t_1:t_2] - cz[t_1:t_2].mean(),a=1,worN=np.arange(0,fs/2,2),fs=fs) 
    axs[1].plot(w,np.abs(Hf),color='k',linestyle='dashed',alpha=0.5,label='Whole tMTF')
    axs[1].legend()
    
    #fig.suptitle(Subjects[sub])
    
#%% Plot whole tMTF vs SAM noise version and compute error
    
t_1 = np.where(t>=0)[0][0]
t_2 = np.where(t>=0.350)[0][0]
A_tmtf_mag_norm = []
A_Hf_norm = np.zeros((t_2-t_1,len(Subjects)))
    
for sub in range(len(Subjects)):
    ch_num = 31
    ch_ind = np.where(A_ch_picks[sub] == 31)[0][0]
    cz = A_Ht[sub][ch_ind,:]      
    
    freqs = A_freqs[sub]
    tmtf_mag = A_tmtf_mag[sub]
    tmtf_mag_norm = tmtf_mag / np.max(tmtf_mag)
    A_tmtf_mag_norm.append(tmtf_mag_norm)

    [w, Hf] = freqz(b=cz[t_1:t_2] - cz[t_1:t_2].mean(),a=1,worN=t_2-t_1,fs=fs) 
    
    Hf_norm = np.abs(Hf) / np.max(np.abs(Hf))
    
    A_Hf_norm[:,sub] = Hf_norm
    

    fig,axs = plt.subplots()
    axs.plot(w,Hf_norm,color='tab:blue',label='Whole tMTF')
    axs.scatter(freqs,tmtf_mag/np.max(tmtf_mag),color='k')
    axs.set_xlabel('Frequency')  
    axs.set_ylabel('Normalized Magnitude')
    axs.set_xlim([0,75])
    axs.set_title(Subjects[sub])
    
    
    
    
f_blocks = [5,10,15,20,30,40,50,60]  
f_blocks = [5,10,15,20,30,40,50,60] 

A_freqs = np.array(A_freqs)
A_tmtf_mag_norm = np.array(A_tmtf_mag_norm)
tmtf_magAvg = np.zeros(len(f_blocks))
tmtf_magSEM = np.zeros(len(f_blocks))
for fb in range(len(f_blocks)):
    if fb == 0:
        f1 = 0
    else:
        f1 = f_blocks[fb-1]
    
    f2 = f_blocks[fb]
    mask = (A_freqs > f1) & (A_freqs < f2)
    tmtf_magAvg[fb] = A_tmtf_mag_norm[mask].mean()    
    tmtf_magSEM[fb] = A_tmtf_mag_norm[mask].std() / np.sqrt(np.sum(mask))

    
plt.figure()
#plt.title('Average tMTFs')
plt.plot(w,A_Hf_norm.mean(axis=1),label='mod-TRF tMTF')
plt.xlabel('Modulation Freqeuncy')
plt.ylabel('Normalized Magnitude')
plt.xlim([0,75])
sem = A_Hf_norm.std(axis=1)# / np.sqrt(len(Subjects))
plt.fill_between(w,A_Hf_norm.mean(axis=1) - sem, A_Hf_norm.mean(axis=1) + sem, color='tab:blue',alpha=0.5)
plt.legend()
# for sub in range(len(Subjects)):
#     freqs=A_freqs[sub]
#     tmtf_mag = A_tmtf_mag[sub]
#     tmtf_mag_norm = tmtf_mag/np.max(tmtf_mag)
#     plt.scatter(A_freqs[sub],tmtf_mag_norm,color='grey')
# plt.plot(f_blocks,tmtf_magAvg,color='black',linewidth=2,label='Average single freqency EFRs')
# plt.fill_between(f_blocks,tmtf_magAvg-tmtf_magSEM,tmtf_magAvg+tmtf_magSEM,color='k',alpha=0.5)
# plt.legend()


plt.figure()
plt.title('Average tMTFs Gamma Range')
plt.plot(w,A_Hf_norm.mean(axis=1))
plt.xlabel('Freqeuncy')
plt.ylabel('Normalized Magnitude')
plt.xlim([30,75])
sem = A_Hf_norm.std(axis=1) / np.sqrt(len(Subjects))
plt.fill_between(w,A_Hf_norm.mean(axis=1) - sem, A_Hf_norm.mean(axis=1) + sem, color='tab:blue',alpha=0.4)
for sub in range(len(Subjects)):
    freqs=A_freqs[sub]
    tmtf_mag = A_tmtf_mag[sub]    
    tmtf_mag_norm = tmtf_mag/np.max(tmtf_mag)
    plt.scatter(A_freqs[sub],tmtf_mag_norm,color='k')
plt.plot(f_blocks,tmtf_magAvg,color='black',linewidth=2)
 
  
    

        
        
        
        
        
        
        
        
        
        
        

