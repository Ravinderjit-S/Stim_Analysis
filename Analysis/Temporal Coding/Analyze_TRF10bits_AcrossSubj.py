#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 19:07:52 2021

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


mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)

data_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
pickle_loc = data_loc + 'Pickles/'

Subjects = ['S207', 'S228','S236','S238','S239','S246','S247','S250']

fs = 4096

A_Tot_trials = []
A_Ht = []
A_Htnf = []
A_info_obj = []
A_ch_picks = []

#%% Load Data

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickle_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials.append(Tot_trials)
    A_Ht.append(Ht)
    A_Htnf.append(Htnf)
    A_info_obj.append(info_obj)
    A_ch_picks.append(ch_picks)
    
#%% Get average Ht
perCh = np.zeros([32,1])
for s in range(len(Subjects)):
    for ch in range(32):
        perCh[ch,0] += np.sum(A_ch_picks[s]==ch)

Avg_Ht = np.zeros([32,t.size])
for s in range(len(Subjects)):
    Avg_Ht[A_ch_picks[s],:] += A_Ht[s]

Avg_Ht = Avg_Ht / perCh


#%% Plot time domain Ht

num_nf = len(A_Htnf[0])

sbp = [4,4]
sbp2 = [4,4]

fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for sub in range(len(A_Ht) + 1):
    if sub == len(A_Ht):
        Ht_1 = Avg_Ht
        ch_picks_s = np.arange(32)
    else:
        Ht_1 = A_Ht[sub]
        ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp[1] + p2
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                if sub == len(A_Ht):
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:],color='k',linewidth=2)
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                else:
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                    
plt.legend(Subjects)
                
    
fig,axs = plt.subplots(sbp[0],sbp[1],sharex=True)
for sub in range(len(A_Ht)+1):
    if sub == len(A_Ht):
        Ht_1 = Avg_Ht
        ch_picks_s = np.arange(32)
    else:
        Ht_1 = A_Ht[sub]
        ch_picks_s = A_ch_picks[sub]
    for p1 in range(sbp[0]):
        for p2 in range(sbp[1]):
            cur_ch = p1*sbp2[1]+p2+sbp[0]*sbp[1]
            if np.any(cur_ch==ch_picks_s):
                ch_ind = np.where(cur_ch==ch_picks_s)[0][0]
                if sub == len(A_Ht):
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:],color='k',linewidth=2)
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                else:
                    axs[p1,p2].plot(t,Ht_1[ch_ind,:])
                    axs[p1,p2].set_title(ch_picks[ch_ind])
                    axs[p1,p2].set_xlim([-0.050,0.5])
                
                   
#%% PCA on t_cuts

t_cuts = [.016, .066,.250,0.500]

pca_sp_cuts = []
pca_expVar_cuts = []
pca_coeff_cuts = []
pca_expVar2s = []

t_cutT = []
pca = PCA(n_components=2)

for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
    
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    pca_sp = pca.fit_transform(Avg_Ht[:,t_1:t_2].T)
    pca_expVar = pca.explained_variance_ratio_
    pca_coeff = pca.components_
    
    if pca_coeff[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff = -pca_coeff
       pca_sp = -pca_sp
       
    Avg_demean = Avg_Ht[:,t_1:t_2] - Avg_Ht[:,t_1:t_2].mean(axis=1)[:,np.newaxis]
    H_tc_est = np.matmul(pca_sp[:,0][:,np.newaxis],pca_coeff[0,:][np.newaxis,:])
    pca_expVar2 = explained_variance_score(Avg_demean.T, H_tc_est,multioutput='variance_weighted')
    
    pca_expVar2s.append(pca_expVar2)
    pca_sp_cuts.append(pca_sp)
    pca_expVar_cuts.append(pca_expVar)
    pca_coeff_cuts.append(pca_coeff)
    t_cutT.append(t[t_1:t_2])
    
    plt.figure()
    plt.plot(t[t_1:t_2], Avg_demean[31,:])
    plt.plot(t[t_1:t_2], H_tc_est[:,31])
    
    
    

plt.figure()
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts[t_c][:,0])
plt.plot(t,Avg_Ht[31,:] -Avg_Ht[31,:].mean(),color='k')
plt.xlim([0,0.5])
    
plt.figure()
plt.title('2nd component')
for t_c in range(len(t_cuts)):
    plt.plot(t_cutT[t_c],pca_sp_cuts[t_c][:,1])
    
plt.figure()
plt.plot(t,Avg_Ht[31,:])
plt.plot(t,-Avg_Ht[0,:])
plt.title('Ch. Cz and Fp1')
plt.xlim([0,0.5])
plt.legend(['Cz','- Fp1'])

plt.figure()
labels = ['comp1', 'comp2']
vmin = pca_coeff_cuts[-1][0,:].mean() - 2 * pca_coeff_cuts[-1][0,:].std()
vmax = pca_coeff_cuts[-1][0,:].mean() + 2 * pca_coeff_cuts[-1][0,:].std()
for t_c in range(len(t_cuts)):
    plt.subplot(2,len(t_cuts),t_c+1)
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts[t_c][0]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts[t_c][0,:], mne.pick_info(A_info_obj[1],A_ch_picks[1]),vmin=vmin,vmax=vmax)
    
    plt.subplot(2,len(t_cuts),t_c+1 + len(t_cuts))
    plt.title('ExpVar ' + str(np.round(pca_expVar_cuts[t_c][1]*100)) + '%')
    mne.viz.plot_topomap(pca_coeff_cuts[t_c][1,:], mne.pick_info(A_info_obj[1],A_ch_picks[1]),vmin=vmin,vmax=vmax)
    
  
with open(os.path.join(pickle_loc,'PCA_passive_template.pickle'),'wb') as file:
    pickle.dump([pca_coeff_cuts,pca_expVar_cuts,t_cuts],file)
  
    
#%% Use PCA template on t_splits for individuals

pca_sp_cuts_sub = []
pca_expVar_cuts_sub = []

for sub in range(len(Subjects)):
    pca_sp_cuts_ = []
    pca_expVar_cuts_ = []

    H_t = A_Ht[sub].T
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        H_tc = H_t[t_1:t_2,:]- H_t[t_1:t_2,:].mean(axis=0)[np.newaxis,:]
        
        pca_sp = np.matmul(H_tc,pca_coeff_cuts[t_c][0,A_ch_picks[sub]])
        
        H_tc_est = np.matmul(pca_coeff_cuts[t_c][0,A_ch_picks[sub]][:,np.newaxis],pca_sp[np.newaxis,:])
        pca_expVar = explained_variance_score(H_tc,H_tc_est.T, multioutput='variance_weighted')
        
        pca_sp_cuts_.append(pca_sp)
        pca_expVar_cuts_.append(pca_expVar)
    
    pca_sp_cuts_sub.append(pca_sp_cuts_)
    pca_expVar_cuts_sub.append(pca_expVar_cuts_)
    
    
# for t_c in range(len(t_cuts)):
#     plt.figure()
#     for sub in range(len(Subjects)):
#         plt.plot(t_cutT[t_c],pca_sp_cuts_sub[sub][t_c]/np.max(pca_sp_cuts_sub[sub][0]))


for sub in range(len(Subjects)):
    plt.figure()
    plt.title(Subjects[sub])
    for t_c in range(len(t_cuts)):
        plt.plot(t_cutT[t_c],pca_sp_cuts_sub[sub][t_c])
        
        
for sub in range(len(Subjects)):
    plt.figure()
    plt.title(Subjects[sub] + ' Cz')
    plt.plot(t,A_Ht[sub][np.where(A_ch_picks[sub]==31)[0][0],:])
    plt.xlim([0,0.5])
    
#%% Load data collected earlier
        
data_loc_old = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_bits4/'
pickle_loc_old = data_loc_old + 'Pickles/'
Subjects_old = ['S211','S207','S236','S228','S238'] 

A_Tot_trials_old = []
A_Ht_old = []
A_Htnf_old = []
A_info_obj_old = []
A_ch_picks_old = []

for sub in range(len(Subjects_old)):
    subject = Subjects_old[sub]
    with open(os.path.join(pickle_loc_old,subject+'_AMmseqbits4.pickle'),'rb') as file:
        [tdat, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_old.append(Tot_trials[3])
    A_Ht_old.append(Ht[3])
    A_Htnf_old.append(Htnf[3])
    
    A_info_obj_old.append(info_obj)
    A_ch_picks_old.append(ch_picks)
    
t = tdat[3]
    
#%% Avg Ht on old data

perCh = np.zeros([32,1])
for s in range(len(Subjects_old)):
    for ch in range(32):
        perCh[ch,0] += np.sum(A_ch_picks_old[s]==ch)

Avg_Ht_old = np.zeros([32,t.size])
for s in range(len(Subjects_old)):
    Avg_Ht_old[A_ch_picks_old[s],:] += A_Ht_old[s]

Avg_Ht_old = Avg_Ht_old / perCh

#%% PCA on old data

pca_sp_cuts_old = []
pca_expVar_cuts_old = []
pca_coeff_cuts_old = []

for t_c in range(len(t_cuts)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
    
    t_2 = np.where(t>=t_cuts[t_c])[0][0]
    pca_sp = pca.fit_transform(Avg_Ht_old[:,t_1:t_2].T)
    pca_expVar = pca.explained_variance_ratio_
    pca_coeff = pca.components_
    
    if pca_coeff[0,31] < 0:  # Expand this too look at mutlitple electrodes
       pca_coeff = -pca_coeff
       pca_sp = -pca_sp
       
    pca_sp_cuts_old.append(pca_sp)
    pca_expVar_cuts_old.append(pca_expVar)
    pca_coeff_cuts_old.append(pca_coeff)
     

plt.figure()
plt.plot(pca_coeff_cuts[0][0,:])
plt.plot(pca_coeff_cuts_old[0][0,:])

plt.figure()
plt.plot(pca_coeff_cuts[1][0,:])
plt.plot(pca_coeff_cuts_old[1][0,:])

#%% PCA on old individual data

pca_sp_cuts_sub_old = []
pca_expVar_cuts_sub_old = []

for sub in range(len(Subjects_old)):
    pca_sp_cuts_ = []
    pca_expVar_cuts_ = []
    H_t = A_Ht_old[sub].T
    for t_c in range(len(t_cuts)):
        if t_c ==0:
            t_1 = np.where(t>=0)[0][0]
        else:
            t_1 = np.where(t>=t_cuts[t_c-1])[0][0]
        
        t_2 = np.where(t>=t_cuts[t_c])[0][0]
        
        H_tc = H_t[t_1:t_2,:]- H_t[t_1:t_2,:].mean(axis=0)[np.newaxis,:]
        
        pca_sp = np.matmul(H_tc,pca_coeff_cuts[t_c][0,A_ch_picks_old[sub]])
        
        H_tc_est = np.matmul(pca_coeff_cuts[t_c][0,A_ch_picks_old[sub]][:,np.newaxis],pca_sp[np.newaxis,:])
        pca_expVar = explained_variance_score(H_tc,H_tc_est.T, multioutput='variance_weighted')
        
        pca_sp_cuts_.append(pca_sp)
        pca_expVar_cuts_.append(pca_expVar)
    
    pca_sp_cuts_sub_old.append(pca_sp_cuts_)
    pca_expVar_cuts_sub_old.append(pca_expVar_cuts_)
    
    
for t_c in range(len(t_cuts)):
    plt.figure()
    for sub in range(len(Subjects_old)):
        plt.plot(t_cutT[t_c],pca_sp_cuts_sub_old[sub][t_c]/np.max(pca_sp_cuts_sub_old[sub][0]))
    

#%% Compare old and new

for sub in range(1,len(Subjects_old)):
    index_new = Subjects.index(Subjects_old[sub])
    
    plt.figure()
    plt.title(Subjects_old[sub])
    for t_c in range(len(t_cuts)):
        l1 = plt.plot(t_cutT[t_c],pca_sp_cuts_sub_old[sub][t_c]/np.max(pca_sp_cuts_sub_old[sub][t_c]),color='tab:blue')
        plt.plot(t_cutT[t_c],pca_sp_cuts_sub_old[sub][t_c]/np.max(pca_sp_cuts_sub_old[sub][t_c]),color='tab:blue')
        
        l2 = plt.plot(t_cutT[t_c],pca_sp_cuts_sub[index_new][t_c]/np.max(pca_sp_cuts_sub[index_new][t_c]),color='tab:orange')
        plt.plot(t_cutT[t_c],pca_sp_cuts_sub[index_new][t_c]/np.max(pca_sp_cuts_sub[index_new][t_c]),color='tab:orange')
    plt.legend(handles = (l1[0],l2[0]),labels =('old','new'))
    
    
#%%Plot Cz and frontal electrodes
    
t_1 = np.where(t>=0)[0][0]
t_2 = np.where(t>=0.010)[0][0]
for sub in range(len(Subjects)):
    plt.figure()
    plt.title(Subjects[sub])
    ch31 = A_Ht[sub][np.where(A_ch_picks[0]==31)[0][0],:] 
    ch1 = -A_Ht[sub][np.where(A_ch_picks[0]==0)[0][0],:]
    ch31 = ch31 - ch31.mean()
    ch1 = ch1 - ch1.mean()
    ch31 = ch31 / np.abs(ch31[t_1:t_2].max())
    ch1 = ch1 / np.abs(ch1[t_1:t_2].max())
    
    plt.plot(t,ch31,color='tab:blue')
    plt.plot(t,ch1,color='tab:orange')
    plt.xlim([0,0.5])
    for t_c in range(len(t_cuts)):
        plt.plot(t_cutT[t_c],pca_sp_cuts_sub[sub][t_c] / np.max(pca_sp_cuts_sub[sub][0]) ,color='k')

#%% Plot freq responses
        
t1 = np.where(t>=0)[0][0]
t2 = np.where(t>=0.500)[0][0]
Cz_avg = Avg_Ht[31,t1:t2] - Avg_Ht[31,t1:t2].mean()
[w,h] = freqz(b=Cz_avg,a=1,worN=np.arange(0,fs/2,2),fs=fs)
phase_resp = np.unwrap(np.angle(h))

plt.figure()
plt.plot(w,np.abs(h))
plt.xlim([0,100])
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.figure()
plt.plot(w,phase_resp)
plt.xlabel('Frequency')
plt.ylabel('Phase (Radians)')
plt.xlim([0,100])

#%% Seperate Peaks in time domain

t_Cz = t[t1:t2]
[peaks,properties] = find_peaks(Cz_avg)
[peaks_neg,properties] = find_peaks(-Cz_avg)

plt.figure()
plt.plot(t_Cz,Cz_avg)
plt.scatter(t_Cz[peaks],Cz_avg[peaks],marker='x',color='r')
plt.scatter(t_Cz[peaks_neg],Cz_avg[peaks_neg],marker='x',color='b')

tpks = []
pks = []
pks_Hf = []
pks_w = []
pks_phase = []
pks_phaseLine = []
pks_phaseLineW = []
pks_gd = []

f1 = [40, 10, 8, 4]
f2 = [80, 25, 22, 10]

peak_locs = peaks[0:4]
peaks_neg = peaks_neg[0:3]
peaks_neg = np.append(peaks_neg,np.where(t_Cz>=.250)[0][0])

for pk in range(len(peaks_neg)):
    if pk ==0:
        t_1 = 0
    else:
        t_1 = peaks_neg[pk-1]
        
    t_2 = peaks_neg[pk]

    tpks.append(t_Cz[t_1:t_2])
    pks.append(Cz_avg[t_1:t_2])
    
    [w, p_Hf] = freqz(b= Cz_avg[t_1:t_2] - Cz_avg[t_1:t_2].mean() ,a=1,worN=np.arange(0,fs/2,2),fs=fs)
    
    f_ind1 = np.where(w>=f1[pk])[0][0]
    f_ind2 = np.where(w>=f2[pk]+2)[0][0]
    
    phase_pkresp = np.unwrap(np.angle(p_Hf))
    coeffs= np.polyfit(w[f_ind1:f_ind2],phase_pkresp[f_ind1:f_ind2],deg=1)
    pks_phaseLine.append(coeffs[0] * w[f_ind1:f_ind2] +coeffs[1])
    pks_phaseLineW.append(w[f_ind1:f_ind2])
    pks_gd.append(-coeffs[0] / (2*np.pi))
    
    
    pks_w.append(w)
    pks_Hf.append(p_Hf)
    pks_phase.append(phase_pkresp)


plt.figure()
for pk in range(len(tpks)):
    plt.plot(tpks[pk],pks[pk])
plt.xlabel('Time (sec)')

plt.figure()
for pk in range(len(tpks)):
    plt.plot(pks_w[pk],np.abs(pks_Hf[pk]))
plt.xlim([0,150])
plt.xlabel('Frequency')
plt.legend([1,2,3,4])

plt.figure()
for pk in range(len(tpks)):
    plt.plot(pks_w[pk], pks_phase[pk])
    plt.plot(pks_phaseLineW[pk], pks_phaseLine[pk],color='k')
plt.xlim([0,150])
plt.xlabel('Frequency')
plt.ylabel('Phase')
plt.ylim([-20,10])
plt.legend([1,2,3,4])


#%% Seperate peaks on sACR

sTRF = np.array([])      
t_s =  np.array([])   
for t_c in range(len(t_cuts)):
    sTRF = np.append(sTRF,pca_sp_cuts[t_c][:,0])
    t_s = np.append(t_s,t_cutT[t_c])

plt.figure()
plt.plot(t_s,sTRF)

[tpks, pks, pks_Hf, pks_w, pks_phase, 
pks_phaseLine, pks_phaseLineW, pks_gd] = ACR_sourceHf(peaks_neg,sTRF,t_s,fs,f1,f2)


plt.figure()
for pk in range(len(tpks)):
    plt.plot(tpks[pk],pks[pk])
plt.xlabel('Time (sec)')

plt.figure()
for pk in range(len(tpks)):
    plt.plot(pks_w[pk],np.abs(pks_Hf[pk]))
plt.xlim([0,150])
plt.xlabel('Frequency')
plt.legend([1,2,3,4])

plt.figure()
for pk in range(len(tpks)):
    plt.plot(pks_w[pk], pks_phase[pk])
    plt.plot(pks_phaseLineW[pk], pks_phaseLine[pk],color='k')
plt.xlim([0,150])
plt.xlabel('Frequency')
plt.ylabel('Phase')
plt.ylim([-20,10])
plt.legend([1,2,3,4])



#%% Seperate peaks for individuals

peaks_sub = []
peaks_neg_sub = []

Cz_sub = []
for sub in range(len(Subjects)):
    plt.figure()
    plt.title(Subjects[sub])
    Ch_ind = np.where(A_ch_picks[sub] == 31)[0][0]
    Cz_avg = A_Ht[sub][Ch_ind,t1:t2] - A_Ht[sub][Ch_ind,t1:t2].mean()
    Cz_sub.append(Cz_avg)
    [peaks,properties] = find_peaks(Cz_avg)
    [peaks_neg,properties] = find_peaks(-Cz_avg)
    peaks_sub.append(peaks)
    peaks_neg_sub.append(peaks_neg)
    plt.plot(t[t1:t2],Cz_avg)
    plt.scatter(t[t1:t2][peaks],Cz_avg[peaks],marker='x',color='r')
    plt.scatter(t[t1:t2][peaks_neg],Cz_avg[peaks_neg],marker='x',color='b')
    plt.xlim([0,0.5])
    

split_locs_sub = []
split_locs_sub.append([80,144,280,559,1024]) #S207
split_locs_sub.append([66,280,521,1024]) #S228
split_locs_sub.append([58,271,500,994])  #S236
split_locs_sub.append([67,270,511,1068]) #S238
split_locs_sub.append([65,262,487,987])  #S239
split_locs_sub.append([69,251,536,1054]) #S246
split_locs_sub.append([75,285,495,816])  #S247
split_locs_sub.append([79,505,943,1360]) #S250

peak_mag_locs_neg = []
peak_mag_locs_pos = []


peak_mag_locs_neg.append( np.concatenate((np.array([0]), peaks_neg_sub[0][0:4]))   )

peak_mag_locs_pos.append([30, 117, 192, 375, 815]) #S207
peak_mag_locs_pos.append([30, 152, 235, 373, 773]) #S228
peak_mag_locs_pos.append([26, 102, 165, 361  ]) #S236 2&3 very merged

Cz_sub[0] = (Cz_sub[0] - np.min(Cz_sub[0][:1024]))
Cz_sub[0] = Cz_sub[0] / np.max(Cz_sub[0][:1024])



plt.figure()
plt.plot(t[t1:t2],Cz_sub[0])

 
  




#%% Analyze peaks in individuals 

A_tpks = []
A_pks = []
A_pksHf = []
A_pks_w = []
A_pks_phase = []
A_pks_phaseLine = []
A_pks_phaseLineW = []
A_pksGD = []

for sub in range(len(Subjects)):
    Ch_ind = np.where(A_ch_picks[sub] == 31)[0][0]
    Cz_avg = A_Ht[sub][Ch_ind,t1:t2] - A_Ht[sub][Ch_ind,t1:t2].mean()
    [tpks, pks, pks_Hf, pks_w, pks_phase, 
    pks_phaseLine, pks_phaseLineW, pks_gd] = ACR_sourceHf(split_locs_sub[sub],Cz_avg,t[t1:t2],fs,f1,f2)
    
    A_tpks.append(tpks)
    A_pks.append(pks)
    A_pksHf.append(pks_Hf)
    A_pks_w.append(pks_w)
    A_pks_phase.append(pks_phase)
    A_pks_phaseLine.append(pks_phaseLine)
    A_pks_phaseLineW.append(pks_phaseLineW)
    A_pksGD.append(pks_gd)
    
    fig,axs = plt.subplots(2,1)
    fig.suptitle = Subjects[sub]
    for pk in range(len(tpks)):
        axs[0].plot(tpks[pk],pks[pk])
        axs[0].set_xlabel('Time (sec)')
    axs[0].set_title(Subjects[sub])
  
    for pk in range(len(tpks)):
        axs[1].plot(pks_w[pk],np.abs(pks_Hf[pk]))
        axs[1].set_xlim([0,150])
        axs[1].set_xlabel('Frequency')
        
    [w,h] = freqz(b=Cz_avg[:split_locs_sub[sub][-1]] - Cz_avg[:split_locs_sub[sub][-1]].mean()  ,a=1,worN=np.arange(0,fs/2,2),fs=fs)
    axs[1].plot(w,np.abs(h),color='black',alpha=0.5, linestyle='dashed')
    
#%% Compare Cz via SAM vs Cz mseq 

pickle_loc_tmtf = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/tmtf/Pickles'

Subjects_tmtf = ['S207','S228','S236','S238','S239','S246']
A_Trials_cond = []
A_freqs = []
A_tmtf_mag = []

for sub in range(len(Subjects_tmtf)):
    subject = Subjects_tmtf[sub]
    with open(os.path.join(pickle_loc_tmtf,subject +'_tmtf.pickle'),'rb') as file:
        [freqs, tmtf_mag, Trials_cond] = pickle.load(file)
    
    A_Trials_cond.append(Trials_cond)
    A_freqs.append(freqs)
    A_tmtf_mag.append(tmtf_mag)
    
    sub_ind = Subjects.index(subject)
    Ch_ind = np.where(A_ch_picks[sub] == 31)[0][0]
    Cz_avg = A_Ht[sub][Ch_ind,t1:t2] - A_Ht[sub][Ch_ind,t1:t2].mean()
    [w,h] = freqz(b=Cz_avg[:split_locs_sub[sub][-1]] - Cz_avg[:split_locs_sub[sub][-1]].mean()  ,a=1,worN=np.arange(0,fs/2,3),fs=fs)
    plt.figure()
    plt.title(subject)
    plt.plot(w,np.abs(h)/np.max(np.abs(h)))
    plt.scatter(freqs,tmtf_mag/np.max(tmtf_mag),color='tab:orange')
    plt.xlim([0,70])
    plt.xlabel('Frequency')
    plt.ylabel('Normalized Magnitude')
    







