#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:40:41 2021

@author: ravinderjit
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import signal
import mne

import sys
sys.path.append(os.path.abspath('../../mseqAnalysis/'))
from mseqHelper import mseqXcorr

direct_Mseq = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Mseq_4096fs_compensated.mat'
data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32_refAvg/')
fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/')


Mseq_mat = sio.loadmat(direct_Mseq)
Mseq = Mseq_mat['Mseq_sig'].T
Mseq = Mseq.astype(float)
Mseq = signal.decimate(Mseq,2,axis=0)
#fix issues due to filtering from downsampling ... data is sampled at 2048
Mseq[Mseq<0] = -1
Mseq[Mseq>0] = 1

Subject = 'S207'


with open(os.path.join(data_loc, Subject+'_DynBin.pickle'),'rb') as f:
    IAC_epochs, ITD_epochs = pickle.load(f)


#%% Extract epochs when stim is on
t = IAC_epochs.times
fs = IAC_epochs.info['sfreq']
t1 = np.where(t>=0)[0][0]
t2 = t1 + Mseq.size + int(np.round(0.4*fs))
t = t[t1:t2]
t = np.concatenate((-t[-int(np.round(0.4*fs)):0:-1],t[:-1]))
ch_picks = np.arange(32)

IAC_ep = IAC_epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)
#ITD_ep = ITD_epochs.get_data()[:,ch_picks,t1:t2].transpose(1,0,2)

IAC_Ht = mseqXcorr(IAC_ep,Mseq[:,0])
#IAC_Ht = IAC_Ht[:,np.where(t>=0)[0][0]:]

#%% Plot stuff

IAC_evoked = IAC_epochs.average()
fig = mne.viz.plot_evoked_topo(IAC_evoked,legend=False)
plt.savefig(os.path.join(fig_path, 'IAC_evkd_topo.svg') , format='svg')


times_plot = (t>0) & (t <0.5)


IAC_evoked.data = IAC_Ht[:,times_plot]
IAC_evoked.times = t[times_plot]
mne.viz.plot_evoked_topo(IAC_evoked,legend=False)
plt.savefig(os.path.join(fig_path, 'IAC_Ht_topo.svg') , format='svg')


t = np.arange(0,Mseq.size/fs,1/fs)
plt.figure()
plt.plot(t,Mseq)

#%% Get average mcBTRF
data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles_32_refAvg/')

Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']

A_IAC_Ht = []

for sub in range(len(Subjects)):
    Subject = Subjects[sub]
    with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc.pickle'),'rb') as file:     
        [t, IAC_Ht, ITD_Ht, IAC_Htnf, ITD_Htnf,Tot_trials_IAC,Tot_trials_ITD] = pickle.load(file)
        
        print(sub)
        
        # [t,IAC_Ht,ITD_Ht,IAC_Htnf,ITD_Htnf,
        #     pca_space_IAC,pca_coeff_IAC,pca_expVar_IAC,
        #     pca_space_ITD,pca_coeff_ITD,pca_expVar_ITD, 
        #     pca_space_IAC_nf,pca_coeffs_IAC_nf,
        #     pca_expVar_IAC_nf, pca_space_ITD_nf,
        #     pca_coeffs_ITD_nf,pca_expVar_ITD_nf, ica_space_ITD,
        #     ica_coeff_ITD, ica_space_IAC, ica_coeff_IAC, 
        #     ica_space_ITD_nf,ica_f_ITD_nf,ica_coeffs_ITD_nf,
        #     ica_space_IAC_nf,ica_coeffs_IAC_nf]= pickle.load(file)
        
              # [t,f,IAC_Ht,ITD_Ht,IAC_Htnf,ITD_Htnf,IAC_Hf,ITD_Hf,
              #   pca_space_IAC,pca_f_IAC,pca_coeff_IAC,pca_expVar_IAC,
              #   pca_space_ITD,pca_f_ITD,pca_coeff_ITD,pca_expVar_ITD, 
              #   pca_space_IAC_nf,pca_f_IAC_nf,pca_coeffs_IAC_nf,
              #   pca_expVar_IAC_nf, pca_space_ITD_nf,pca_f_ITD_nf,
              #   pca_coeffs_ITD_nf,pca_expVar_ITD_nf, ica_space_ITD,
              #   ica_f_ITD,ica_coeff_ITD, ica_space_IAC, ica_f_IAC, 
              #   ica_coeff_IAC, ica_space_ITD_nf,ica_f_ITD_nf,ica_coeffs_ITD_nf,
              #   ica_space_IAC_nf, ica_f_IAC_nf,ica_coeffs_IAC_nf] = pickle.load(file)

    A_IAC_Ht.append(IAC_Ht)

Anp_Ht_IAC = np.zeros([A_IAC_Ht[0].shape[0],A_IAC_Ht[0].shape[1],len(Subjects)])
for s in range(len(Subjects)):
    Anp_Ht_IAC[:,:,s] = A_IAC_Ht[s]
    
Ht_avg_IAC = Anp_Ht_IAC.mean(axis=2)

IAC_evoked.data = Ht_avg_IAC[:,times_plot]
mne.viz.plot_evoked_topo(IAC_evoked,legend=False)
plt.savefig(os.path.join(fig_path, 'IAC_Htavg_topo.svg') , format='svg')
