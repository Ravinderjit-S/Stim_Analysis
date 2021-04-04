#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 15:40:45 2021

@author: ravinderjit
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


data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
pickle_loc = data_loc + 'Pickles_full/'


Subjects = ['S211','S207','S236','S228','S238'] #S237 data is crazy noisy

m_bits = [7,8,9,10]
mseq_locs = ['mseqEEG_150_bits7_4096.mat', 'mseqEEG_150_bits8_4096.mat', 
             'mseqEEG_150_bits9_4096.mat', 'mseqEEG_150_bits10_4096.mat']
mseq = []
for m in mseq_locs:
    file_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/' + m
    Mseq_dat = sio.loadmat(file_loc)
    mseq.append( Mseq_dat['mseqEEG_4096'].astype(float) )


fs = 4096

A_Tot_trials = []
A_Ht =[]
A_Htnf =[]

A_info_obj = []
A_ch_picks = []


#%% Load data

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject+'_AMmseqbits4.pickle'),'rb') as file:
        # [tdat, Tot_trials, Ht, Htnf, pca_sp, pca_coeff, pca_expVar, 
        #  pca_sp_nf, pca_coeff_nf,pca_expVar_nf,ica_sp,
        #  ica_coeff,ica_sp_nf,ica_coeff_nf, info_obj, ch_picks] = pickle.load(file)
        [tdat, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    A_Htnf.append(Htnf)
    
    
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
    

#%% Plot time domain Ht
    
    
num_nf = len(A_Htnf[0]) / len(m_bits)
    
fig,ax = plt.subplots(4,len(Subjects),sharex=True)
for s in range(len(Subjects)):
    for m in range(len(m_bits)):
        ax[m,s].plot(tdat[m],A_Ht[s][m][-1,:]) #Just Channel A32 (last channel)
    ax[0,s].set_title(Subjects[s])
    

sbp = [4,4]
sbp2 = [4,4]

for m in range(len(Ht)):
    fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
    t = tdat[m]
    for s in range(len(Subjects)):
        Ht_1 = A_Ht[s][m]
        ch_picks_s = A_ch_picks[s]
        for p1 in range(sbp[0]):
            for p2 in range(sbp[1]):
                cur_ch = p1*sbp[1]+p2
                if np.any(cur_ch==ch_picks_s):
                    ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])    
                    # axs[p1,p2].set_xlim([0,0.5])
                    for n in range(int(m*num_nf),int(num_nf*(m+1))):
                        axs[p1,p2].plot(t,A_Htnf[s][n][ch_ind,:],color='grey',alpha=0.3)
                
    fig.suptitle('Ht ' + str(m_bits[m]) + ' bits')
    
for m in range(len(Ht)):    
    fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
    t = tdat[m]    
    for s in range(len(Subjects)):
        Ht_1 = A_Ht[s][m]
        ch_picks_s = A_ch_picks[s]
        for p1 in range(sbp2[0]):
            for p2 in range(sbp2[1]):
                cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
                if np.any(cur_ch==ch_picks_s):
                    ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])   
                    # axs[p1,p2].set_xlim([0,0.5])
                    for n in range(int(m*num_nf),int(num_nf*(m+1))):
                        axs[p1,p2].plot(t,A_Htnf[s][n][ch_ind,:],color='grey',alpha=0.3)
            
    fig.suptitle('Ht ' +  str(m_bits[m]) + ' bits')   

#%% Split Ht into two times and do PCA
    
# t_split1 = .050
# t_split2 = .5

# pca_sp_s1 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]
# pca_sp_s2 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]

# pca_expVar_s1 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]
# pca_expVar_s2 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]

# pca_coeff_s1 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]
# pca_coeff_s2 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]

# ica_sp_s1 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]
# ica_sp_s2 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]

# ica_coeff_s1 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]
# ica_coeff_s2 = [[list() for i in range(len(A_Ht[0]))] for j in range(len(A_Ht))]


# pca = PCA(n_components=2)
# ica = FastICA(n_components=1)
# for s in range(len(Subjects)):
#     for m in range(len(Ht)):
#         t = tdat[m]
#         t_1 = np.where(t>=0)[0][0]
#         t_2 = np.where(t>=t_split1)[0][0]
#         t_3 = np.where(t>=t_split2)[0][0]
        
#         pca_sp_s1[s][m] = pca.fit_transform(A_Ht[s][m][:,t_1:t_2].T)
#         pca_expVar_s1[s][m] = pca.explained_variance_ratio_
#         pca_coeff_s1[s][m] = pca.components_
        
#         pca_sp_s2[s][m] = pca.fit_transform(A_Ht[s][m][:,t_2:t_3].T)
#         pca_expVar_s2[s][m] = pca.explained_variance_ratio_
#         pca_coeff_s2[s][m] = pca.components_
        
#         ica_sp_s1[s][m] = ica.fit_transform(A_Ht[s][m][:,t_1:t_2].T)
#         ica_coeff_s1[s][m] = ica.components_
        
#         ica_sp_s2[s][m] = ica.fit_transform(A_Ht[s][m][:,t_2:t_3].T)
#         ica_coeff_s2[s][m] = ica.components_
        
#%% Get average Ht

perCh = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh[ch,0] += np.sum(A_ch_picks[s] == ch)

Avg_Ht = []
for m in range(len(Ht)):
    Avg_m = np.zeros([32,tdat[m].size])
    for s in range(len(Subjects)):
        Avg_m[A_ch_picks[s],:] += A_Ht[s][m]
    Avg_m = Avg_m / perCh
    Avg_Ht.append(Avg_m)
    
   
for m in range(len(Avg_Ht)):
   Ht_1 = Avg_Ht[m]
   t = tdat[m]
   fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,gridspec_kw=None)
   for p1 in range(sbp[0]):
       for p2 in range(sbp[1]):
           axs[p1,p2].plot(t,Ht_1[p1*sbp[1]+p2,:],color='k')
           axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])    
           axs[p1,p2].set_xlim([0,0.5])
           # for n in range(m*num_nfs,num_nfs*(m+1)):
           #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
           
   fig.suptitle('Ht ' + str(m_bits[m]))
   
   
   fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,gridspec_kw=None)
   for p1 in range(sbp2[0]):
       for p2 in range(sbp2[1]):
           axs[p1,p2].plot(t,Ht_1[p1*sbp2[1]+p2+sbp[0]*sbp[1],:],color='k')
           axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
           axs[p1,p2].set_xlim([0,0.5])
           # for n in range(m*num_nfs,num_nfs*(m+1)):
           #     axs[p1,p2].plot(t,Htnf[n][p1*sbp[1]+p2,:],color='grey',alpha=0.3)
           
   fig.suptitle('Ht ' + str(m_bits[m]))    
   
    
#%% Pearson corr check to find time cutoffs

template_ch = 13

t_cuts = [0]

t_steps = np.arange(t_cuts[-1]+.001,.500,.001)
corr_ch_t = np.zeros([t_steps.size,32])

t = tdat[3]
for t_s in range(len(t_steps)):
    t1 = np.where(t>=t_cuts[-1])[0][0]
    t2 = np.where(t>=t_steps[t_s])[0][0]
    for ch in range(32):
        corr_ch_t[t_s,ch] = np.abs(pearsonr(Avg_Ht[3][template_ch,t1:t2],Avg_Ht[3][ch,t1:t2])[0])    



fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True,sharey=True)
for p1 in range(sbp[0]):
    for p2 in range(sbp[1]):
        axs[p1,p2].plot(t_steps,corr_ch_t[:,p1*sbp[1]+p2],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp[1]+p2])   
        axs[p1,p2].set_ylim([0,1])
        #axs[p1,p2].set_xlim([0,0.5])

        
fig.suptitle('Ht_Pearson ')


fig,axs = plt.subplots(sbp2[0],sbp2[1],sharex=True,sharey=True)
for p1 in range(sbp2[0]):
    for p2 in range(sbp2[1]):
        axs[p1,p2].plot(t_steps,corr_ch_t[:,p1*sbp2[1]+p2+sbp[0]*sbp[1]],color='k')
        axs[p1,p2].set_title(ch_picks[p1*sbp2[1]+p2+sbp[0]*sbp[1]])   
        axs[p1,p2].set_ylim([0,1])
        #axs[p1,p2].set_xlim([0,0.5])

        
fig.suptitle('Ht_pearson ')

t_cuts = [.015,.040,.125,.500]

#%% PCA on t_cuts

pca_sp_cuts = [list() for i in range(len(t_cuts))]
ica_sp_cuts = [list() for i in range(len(t_cuts))]

pca_expVar_cuts = [list() for i in range(len(t_cuts))]
pca_coeff_cuts = [list() for i in range(len(t_cuts))]

ica_coeff_cuts = [list() for i in range(len(t_cuts))]
A_t_ = [list() for i in range(len(t_cuts))]

pca = PCA(n_components=2)
ica = FastICA(n_components=1)
t = tdat[3]
for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    
    pca_sp_cuts[t_c] = pca.fit_transform(Avg_Ht[3][:,t_1:t_2].T)
    pca_expVar_cuts[t_c] = pca.explained_variance_ratio_
    pca_coeff_cuts[t_c] = pca.components_
    
    ica_sp_cuts[t_c] = ica.fit_transform(Avg_Ht[3][:,t_1:t_2].T)
    ica_coeff_cuts[t_c] = ica.components_
    
    if pca_coeff_cuts[t_c][0,31] < 0:  # Expand this too look at mutlitple electrodes
        pca_coeff_cuts[t_c] = -pca_coeff_cuts[t_c]
        pca_sp_cuts[t_c] = -pca_sp_cuts[t_c]
    

for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    
    t_ = t[t_1:t_2]
    
    A_t_[t_c] = t_
    
    plt.figure()
    plt.title(t_cuts[t_c])
    plt.plot(t_,pca_sp_cuts[t_c])
    
plt.figure()
for t_c in range(len(t_cuts)):
    plt.plot(A_t_[t_c],pca_sp_cuts[t_c][:,0])

plt.figure()
labels = ['subcortical','mixed','cortical','higher cortical']
vmin = pca_coeff_cuts[-1][0,:].mean() - 2* pca_coeff_cuts[-1][0,:].std()
vmax = pca_coeff_cuts[-1][0,:].mean() + 2* pca_coeff_cuts[-1][0,:].std()
for t_c in range(len(t_cuts)):
    plt.subplot(2,len(t_cuts),t_c+1)
    #plt.title(str(t_cuts[t_c]) + ' ' + labels[t_c])
    plt.title(' ExpVar ' + str(np.round(pca_expVar_cuts[t_c][0]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts[t_c][0,:], mne.pick_info(A_info_obj[2], A_ch_picks[2]),vmin=vmin,vmax=vmax)
    
    plt.subplot(2,len(t_cuts),len(t_cuts)+t_c+1)
    plt.title(' ExpVar ' + str(np.round(pca_expVar_cuts[t_c][1]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts[t_c][1,:], mne.pick_info(A_info_obj[2], A_ch_picks[2]),vmin=vmin,vmax=vmax)
    
#pca on first 500 ms
t_1 = np.where(t>=0)[0][0]
t_2 = np.where(t>=0.5)[0][0]
pca_sp_whole = pca.fit_transform(Avg_Ht[3][:,t_1:t_2].T)
pca_expVar_whole = pca.explained_variance_ratio_
pca_coeff_whole = pca.components_

plt.figure()
plt.plot(t[t_1:t_2],pca_sp_whole)

plt.figure()
plt.subplot(1,2,1)
mne.viz.plot_topomap(pca_coeff_whole[0,:], mne.pick_info(A_info_obj[2], A_ch_picks[2]),vmin=vmin,vmax=vmax)
plt.subplot(1,2,2)
mne.viz.plot_topomap(pca_coeff_whole[1,:], mne.pick_info(A_info_obj[2], A_ch_picks[2]),vmin=vmin,vmax=vmax)





    


    