#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:21:25 2020

@author: ravinderjit
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as sio
import mne
from anlffr.preproc import find_blinks
from EEGpp import EEGconcatenateFolder
from mne.preprocessing.ssp import compute_proj_epochs
import os
import pickle
from anlffr.helper import biosemi2mne as bs
from mne import concatenate_raws
import importlib.util
import sys
sys.path.append('../mseqAnalysis/mseqHelper.py')
from mseqHelper import mseqXcorr



# from anlffr.spectral import mtspecraw
# from anlffr.spectral import mtplv
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA



nchans = 34;
refchans = ['EXG1','EXG2']

data_eeg = [];
data_evnt = [];
  
mseq_locs = ['mseqEEG_150_bits7_16384_.mat', 'mseqEEG_150_bits10_16384_.mat']
mseq = []
for m in mseq_locs:
    file_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/' + m
    Mseq_dat = sio.loadmat(file_loc)
    mseq.append( Mseq_dat['mseqEEG_16384'].astype(float) )



#data_loc = '/media/ravinderjit/Storage2/EEGdata/'
data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits2/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S211']

num_nfs = 1

for subject in Subjects:
    print('On Subject ...... ' + subject )
    exclude = ['EXG3','EXG4','EXG5','EXG6','EXG7','EXG8']; #don't need these extra external channels that are saved=
    
    datapath =  os.path.join(data_loc, subject)
    EEGfiles = os.listdir(datapath)
    EEGfiles.sort() #This line and next to fix order of files
    EEGfiles.insert(0,EEGfiles.pop(len(EEGfiles)-1))
    
    epdat = []

#%% Go through all files
    file_num = 0
    files_at_time = 3
    while file_num < len(EEGfiles):
        data_eeg = []
        data_evnt = []
        for fn in range(files_at_time):
            if file_num < len(EEGfiles):
                raw_f, events_f = bs.importbdf(datapath+'/'+ EEGfiles[file_num],nchans,refchans,exclude=exclude)
                data_eeg.append(raw_f)
                data_evnt.append(events_f)
                file_num += 1
        data_eeg, data_evnt = concatenate_raws(data_eeg,events_list=data_evnt)
        data_eeg.filter(l_freq=1,h_freq=1000)
        del raw_f, events_f 
        
# Blink Removal
        blinks = find_blinks(data_eeg,ch_name = ['A1'],thresh = 100e-6, l_trans_bandwidth = 0.5, l_freq =1.0)
        blink_epochs = mne.Epochs(data_eeg,blinks,998,tmin=-0.25,tmax=0.25,proj=False,
                                      baseline=(-0.25,0),reject=dict(eeg=500e-6))
        Projs = compute_proj_epochs(blink_epochs,n_grad=0,n_mag=0,n_eeg=8,verbose='DEBUG')
        
        ocular_projs = Projs[0]
        
        
        data_eeg.add_proj(ocular_projs)
        data_eeg.plot_projs_topomap()
        #data_eeg.plot(events=blinks,show_options=True)
        del blinks, blink_epochs,Projs,ocular_projs

# Plot data

        labels = ['7bits','10bits']
        reject = dict(eeg=250e-6)
        epochs = []
        fs = data_eeg.info['sfreq']
        for j in range(len(mseq)):
            epochs.append(mne.Epochs(data_eeg, data_evnt, [j+1], tmin=-0.3, 
                                     tmax=np.ceil(mseq[j].size/fs),reject=reject, baseline=(-0.2, 0.)) )
            #epochs[j].average().plot(picks=[31],titles = labels[j])
        info_obj = epochs[0].info
        print('On Subject ...... ' + subject )
    
# Extract part of response when stim is on
        ch_picks = np.arange(32)
        remove_chs = []
        ch_picks = np.delete(ch_picks,remove_chs)
        
        tdat = []
        for m in range(len(mseq)):
            t = epochs[m].times
            t1 = np.where(t>=0)[0][0]
            t2 = t1 + mseq[m].size
            if len(epdat) <2:
                epdat.append(epochs[m].get_data()[:,ch_picks,t1:t2].transpose(1,0,2))
            else:
                epdat[m] = np.concatenate((epdat[m],epochs[m].get_data()[:,ch_picks,t1:t2].transpose(1,0,2)),axis=1)
            tdat.append(t[t1:t2])
            tdat[m] = np.concatenate([-tdat[m][-1:0:-1],tdat[m]]) 
        
        del epochs
  
#%% Remove epochs with large deflections
    # Reject_Thresh=150e-6
   
    # Tot_trials = np.zeros([len(mseq)])
    # for m in range(len(mseq)):
    #     Peak2Peak = epdat[m].max(axis=2) - epdat[m].min(axis=2)
    #     mask_trials = np.all(Peak2Peak <Reject_Thresh,axis=0)
    #     print('rejected ' + str(epdat[m].shape[1] - sum(mask_trials)) + ' trials due to P2P')
    #     epdat[m] = epdat[m][:,mask_trials,:]
    #     print('Total Trials Left: ' + str(epdat[m].shape[1]))
    #     Tot_trials[m] = epdat[m].shape[1]
        
    # plt.figure()
    # plt.plot(Peak2Peak.T)
    
    

#%% Correlation Analysis
    
    # tend =  0.5#time of Ht to keep
    # tend_ind = round(tend*fs) - 1
    
    Ht = []
    Htnf = []
    Tot_trials = np.zeros([len(mseq)])
    # do cross corr
    for m in range(len(epdat)): 
        print('On mseq # ' + str(m+1))
        Tot_trials[m] = epdat[m].shape[1]
        
        Ht_m = mseqXcorr(epdat[m],mseq[m][0,:])
        Ht.append(Ht_m)
        for nf in range(num_nfs):
            resp = epdat[m]
            inv_inds = np.random.permutation(epdat[m].shape[1])[:round(epdat[m].shape[1]/2)]
            resp[:,inv_inds,:] = -resp[:,inv_inds,:]
            Ht_nf = mseqXcorr(resp,mseq[m][0,:])
                
                
        
        
    #only keep Ht up to tend 
    # for h in range(len(Ht)):
    #     Ht[h] = Ht[h][:,:tend_ind]
    #     tdat[h] = tdat[h][:tend_ind]
    



#%% Plot Ht
    
    if ch_picks.size == 31:
        sbp = [5,3]
        sbp2 = [4,4]
    elif ch_picks.size == 32:
        sbp = [4,4]
        sbp2 = [4,4]
    elif ch_picks.size == 30:
        sbp = [5,3]
        sbp2 = [5,3]
        

    
    for m in range(len(Ht)):
        Ht_1 = Ht[m]
        t = tdat[m]
        fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
        for p1 in range(sbp[0]):
            for p2 in range(sbp[1]):
                axs[p1,p2].plot(t,Ht_1[p1*sbp[1]+p2,:],color='k')
                axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
                # for n in range(m*num_nfs,num_nfs*(m+1)):
                #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
                
        fig.suptitle('Ht ' + labels[m])
        
        
        fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
        for p1 in range(sbp2[0]):
            for p2 in range(sbp2[1]):
                axs[p1,p2].plot(t,Ht_1[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
                axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
                # for n in range(m*num_nfs,num_nfs*(m+1)):
                #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
                
        fig.suptitle('Ht ' + labels[m])    
        
    
        


#%% PCA decomposition of Ht
    # pca_sp = []
    # pca_coeff = []
    # pca_expVar = []
    
    # pca_sp_nf = []
    # pca_coeff_nf = []
    # pca_expVar_nf = []
    
    # n_comp = 4
    
    # for m in range(len(Ht)):
    #     pca = PCA(n_components=n_comp)
    #     pca.fit(Ht[m])
    #     pca_space = pca.fit_transform(Ht[m].T)
        
       
        
    #     pca_sp.append(pca_space)
    #     pca_coeff.append(pca.components_)
    #     pca_expVar.append(pca.explained_variance_ratio_)
        
    # for n in range(len(Htnf)):
    #     pca = PCA(n_components=n_comp)
    #     pca.fit(Htnf[n])
    #     pca_space = pca.fit_transform(Htnf[n].T)
    
    #     pca_sp_nf.append(pca_space)
    #     pca_coeff_nf.append(pca.components_)
    #     pca_expVar_nf.append(pca.explained_variance_ratio_)
        
    # for m in range(len(pca_sp)):
    #     fig,axs = plt.subplots(2,1)
    #     axs[0].plot(tdat[m],pca_sp[m])
    #     # for n in range(m*num_nfs,num_nfs*(m+1)):
    #     #     axs[0].plot(tdat[m],pca_sp_nf[n],color='grey',alpha=0.3)
        
        
    #     axs[1].plot(ch_picks,pca_coeff[m].T)
    #     axs[1].set_xlabel('channel')
    #     # for n in range(m*num_nfs,num_nfs*(m+1)):
    #     #     axs[1].plot(ch_picks,pca_coeff_nf[n].T,color='grey',alpha=0.1)
    #     fig.suptitle('PCA ' + labels[m])    
        
    # p_ind = 0
    # vmin = pca_coeff[p_ind].mean() - 2 * pca_coeff[p_ind].std()
    # vmax = pca_coeff[p_ind].mean() + 2 * pca_coeff[p_ind].std()
    # plt.figure()
    # mne.viz.plot_topomap(pca_coeff[p_ind][0,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.figure()
    # mne.viz.plot_topomap(pca_coeff[p_ind][1,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.figure()
    # mne.viz.plot_topomap(pca_coeff[p_ind][2,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.figure()
    # mne.viz.plot_topomap(pca_coeff[p_ind][3,:], mne.pick_info(epochs[0].info, ch_picks),vmin=vmin,vmax=vmax)
    
    
    
        

#%% ICA decomposition of Ht
    ica_sp = []
    ica_coeff = []
    #ica_expVar = []
    
    ica_sp_nf = []
    ica_coeff_nf = []
    #ica_expVar_nf = []
    
    n_comp = 2
    
    # for m in range(len(Ht)):
    #     ica = FastICA(n_components=n_comp)
    #     ica.fit(Ht[m])
    #     ica_space = ica.fit_transform(Ht[m].T)
    #     ica_sp.append(ica_space)
    #     ica_coeff.append(ica.components_)
        
    # for n in range(len(Htnf)):
        
    #     ica = FastICA(n_components=n_comp)
    #     ica.fit(Ht[m])
    #     ica_space = ica.fit_transform(Htnf[m].T)
        
    #     ica_sp_nf.append(ica_space)
    #     ica_coeff_nf.append(ica.components_)
    #     #ica_expVar_nf.append(ica.explained_variance_ratio_)
        
    
            
        
    # for m in range(len(ica_sp)):
    #     fig,axs = plt.subplots(2,1)
    #     axs[0].plot(tdat[m],ica_sp[m])
    #     for n in range(m*num_nfs,num_nfs*(m+1)):
    #         axs[0].plot(tdat[m],ica_sp_nf[n],color='grey',alpha=0.3)
    
    #     axs[1].plot(ch_picks,ica_coeff[m].T)
    #     axs[1].set_xlabel('channel')
    #     # for n in range(m*num_nfs,num_nfs*(m+1)):
    #     #     axs[3].plot(ch_picks,ica_coeff_nf[n].T,color='grey',alpha=0.1)
    #     fig.suptitle('ICA ' + labels[m])    
        
        
    # p_ind = 3
    # vmin = ica_coeff[p_ind].mean() - 2 * ica_coeff[p_ind].std()
    # vmax = ica_coeff[p_ind].mean() + 2 * ica_coeff[p_ind].std()
    # plt.figure()
    # mne.viz.plot_topomap(ica_coeff[p_ind][0,:], mne.pick_info(epochs[3].info, ch_picks),vmin=vmin,vmax=vmax)
    # plt.figure()
    # mne.viz.plot_topomap(ica_coeff[p_ind][1,:], mne.pick_info(epochs[3].info, ch_picks),vmin=vmin,vmax=vmax)
        

    
    #%% Save Data
    with open(os.path.join(pickle_loc,subject+'_AMmseqbits2_16384.pickle'),'wb') as file:
        pickle.dump([tdat, Tot_trials, Ht, Htnf,
                     info_obj, ch_picks],file)
