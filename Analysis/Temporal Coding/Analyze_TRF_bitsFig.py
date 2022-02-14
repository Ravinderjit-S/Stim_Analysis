#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:12:05 2022

@author: ravinderjit

Demonstrate how mod-TRF changes depending on the number of bits used in the m-seq
"""


import os
import pickle
import numpy as np
import scipy as sp
import mne
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
import scipy.io as sio

fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/TemporalCoding/')

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
# pickle_loc = data_loc + 'Pickles_full_wholeHead/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S207','S211','S236','S228']#,'S238'] #leave S238 out for now to make 4 x 4 figure



m_bits = [7,8,9,10]
mseq_locs = ['mseqEEG_150_bits7_4096.mat', 'mseqEEG_150_bits8_4096.mat', 
             'mseqEEG_150_bits9_4096.mat', 'mseqEEG_150_bits10_4096.mat']
mseq = []
for m in mseq_locs:
    file_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/' + m
    Mseq_dat = sio.loadmat(file_loc)
    mseq.append( Mseq_dat['mseqEEG_4096'].astype(float) )


fs = 4096


#%% Load data


A_Ht_epochs = []
A_ch_picks = []
A_Htnf = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject+'_AMmseqbits4_epochs.pickle'),'rb') as file:
        [Ht_epochs,t_epochs] = pickle.load(file)
    
    A_Ht_epochs.append(Ht_epochs)
    
# Load Ch_picks    
for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject+'_AMmseqbits4.pickle'),'rb') as file:
        [tdat, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
    
    A_ch_picks.append(ch_picks)
    A_Htnf.append(Htnf)
    

#%% Plot time domain

colors = [ '#FDBEA0', '#F75D5D', '#BF0202', '#101010']

fig,ax = plt.subplots(4,4,sharex=True)
fig.set_size_inches(10,8)
for sub in range(len(Subjects)):
    sub_nf = A_Htnf[sub]
    for k in range(4):
        cz_nf_k = []
        for nf in range(50):
          cz_nf_k.append(A_Htnf[sub][nf + k*50][A_ch_picks[sub] ==31,:]) #ch cz
        
        cz_nf_k = np.array(cz_nf_k)[:,0,:]
        
        mean_nf = cz_nf_k.mean(axis=0)
        std_nf = cz_nf_k.std(axis=0)
        
        trials_cz = A_Ht_epochs[sub][k][A_ch_picks[sub] ==31,:,:].squeeze()
        trials_sem = trials_cz.std(axis=0) / np.sqrt(trials_cz.shape[0])
        trials_mean = trials_cz.mean(axis=0)
        
        t_0 = np.where(t_epochs[k] >=0)[0][0]
        trials_mean = trials_mean - trials_mean[t_0]
        
        mean_nf = mean_nf - mean_nf[t_0]
        
        
        #ax[k,sub].plot(tdat[k], mean_nf,color='grey')
        #ax[k,sub].fill_between(tdat[k], mean_nf - std_nf, mean_nf + std_nf,color='grey',alpha=0.5)
        ax[k,sub].plot(tdat[k], cz_nf_k.T,color='grey',alpha=0.5)
        
        ax[k,sub].plot(t_epochs[k],trials_mean,color=colors[k])
        ax[k,sub].fill_between(t_epochs[k],trials_mean-trials_sem,trials_mean+trials_sem,alpha=0.5,color=colors[k])
        ax[k,sub].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        
        ax[k,sub].locator_params(axis='y',nbins=3)
        
    ax[0,sub].set_title('S' + str(sub+1),fontweight='bold')
    
ax[0,0].set_xlim(-0.050,0.4)
ax[0,0].set_xticks([0,0.2,0.4])
ax[3,0].set_xlabel('Time (s)')
ax[0,0].set_ylabel('7 Bits',rotation=0,labelpad=15,fontweight='bold')
ax[1,0].set_ylabel('8 Bits',rotation=0,labelpad=15,fontweight='bold')
ax[2,0].set_ylabel('9 Bits',rotation=0,labelpad=15,fontweight='bold')
ax[3,0].set_ylabel('10 Bits',rotation=0,labelpad=15,fontweight='bold')

#plt.savefig(os.path.join(fig_path,'ModTRF_4bits.svg'),format='svg')


fig,ax = plt.subplots(2,2,sharex=True)
ax = np.reshape(ax,[4])
fig.set_size_inches(10,8)
for sub in range(len(Subjects)):
    for k in range(4):
        t_0 = np.where(t_epochs[k] >=0)[0][0]
        t_1 = np.where(t_epochs[k] >=0.010)[0][0]
        
        trials_cz = A_Ht_epochs[sub][k][A_ch_picks[sub] ==31,:,:].squeeze()
        trials_mean = trials_cz.mean(axis=0)
        trials_sem =  trials_cz.std(axis=0) / np.sqrt(trials_cz.shape[0])
        
        trials_mean = trials_mean - trials_mean[t_0]
        norm_pk1 = trials_mean[t_0:t_1].max()
        
        trials_mean /= norm_pk1
        trials_sem /=norm_pk1
        
        ax[sub].plot(t_epochs[k],trials_mean,color=colors[k])
        ax[sub].fill_between(t_epochs[k],trials_mean-trials_sem,trials_mean+trials_sem,alpha=0.5,color=colors[k])
        ax[sub].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax[sub].locator_params(axis='y',nbins=3)
    
    ax[sub].set_title('S' + str(sub+1), fontweight='bold')
        
ax[2].set_xlim(-0.050,0.4) 
ax[2].legend(['7 bit','8 bit', '9bit','10 bit'])     
ax[2].set_xlabel('Time (sec)')
ax[2].set_ylabel('Normalized Amplitude')           

#plt.savefig(os.path.join(fig_path,'ModTRF_compareBits.svg'),format='svg')








