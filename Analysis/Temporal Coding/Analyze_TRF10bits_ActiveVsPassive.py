#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:15:54 2021

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
import ACR_helperFuncs

sys.path.append(os.path.abspath('../mseqAnalysis/'))
from mseqHelper import mseqXcorr

#%% Load mseq
mseq_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/mseqEEG_150_bits10_4096.mat'
Mseq_dat = sio.loadmat(mseq_loc)
mseq = Mseq_dat['mseqEEG_4096'].astype(float)
        
#%% Subjects

Subjects = ['S207', 'S228','S236','S238','S239','S246','S247','S250']
Subjects_sd = ['S207', 'S228', 'S236', 'S238', 'S239', 'S250'] #Leaving out S211 for now


dataPassive_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/'
picklePassive_loc = dataPassive_loc + 'Pickles/'

dataCount_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active/'
pickleCount_loc = dataCount_loc + 'Pickles/'

pickle_loc_sd = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_active_harder/Pickles/'


#%% Load Passive Data

A_Tot_trials_pass = []
A_Ht_pass = []
A_Htnf_pass = []
A_info_obj_pass = []
A_ch_picks_pass = []

A_Ht_epochs_pass = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    if subject == 'S250':
        subject = 'S250_visit2'
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials_pass.append(Tot_trials)
    A_Ht_pass.append(Ht)
    A_Htnf_pass.append(Htnf)
    A_info_obj_pass.append(info_obj)
    A_ch_picks_pass.append(ch_picks)
    
    with open(os.path.join(picklePassive_loc,subject +'_AMmseq10bits_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
    
    A_Ht_epochs_pass.append(Ht_epochs)

print('Done loading passive ...')

#%% Load Counting Data

A_Tot_trials_count = []
A_Ht_count = []
A_Htnf_count = []
A_info_obj_count = []
A_ch_picks_count = []

A_Ht_epochs_count = []

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active.pickle'),'rb') as file:
        [t, Tot_trials, Ht, Htnf, info_obj, ch_picks] = pickle.load(file)
        
    A_Tot_trials_count.append(Tot_trials)
    A_Ht_count.append(Ht)
    A_Htnf_count.append(Htnf)
    A_info_obj_count.append(info_obj)
    A_ch_picks_count.append(ch_picks)
    
    with open(os.path.join(pickleCount_loc,subject +'_AMmseq10bits_Active_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    A_Ht_epochs_count.append(Ht_epochs)
    
print('Done Loading Counting data')

#%% Load Shift Detect Data

Subjects_sd = ['S207', 'S228', 'S236', 'S238', 'S239', 'S250'] #Leaving out S211 for now

A_Tot_trials_sd = []
A_Ht_sd = []
A_info_obj_sd = []
A_ch_picks_sd = []

A_Ht_epochs_sd = []

for sub in range(len(Subjects_sd)):
    subject = Subjects_sd[sub]
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder.pickle'),'rb') as file:
        [t, Tot_trials, Ht, info_obj, ch_picks] = pickle.load(file)
    
    A_Tot_trials_sd.append(Tot_trials)
    A_Ht_sd.append(Ht)
    A_info_obj_sd.append(info_obj)
    A_ch_picks_sd.append(ch_picks)
    
    with open(os.path.join(pickle_loc_sd,subject +'_AMmseq10bit_Active_harder_epochs.pickle'),'rb') as file:
        [Ht_epochs, t_epochs] = pickle.load(file)
        
    A_Ht_epochs_sd.append(Ht_epochs)
    
print('Done loading shift detect data')

#%%#%% Load Template
    
template_loc = '/media/ravinderjit/Data_Drive/Data/EEGdata/TemporalCoding/AMmseq_10bits/Pickles/PCA_passive_template.pickle'

with open(template_loc,'rb') as file:
    [pca_coeffs_cuts,pca_expVar_cuts,t_cuts] = pickle.load(file)
    
bstemTemplate  = pca_coeffs_cuts[0][0,:]
cortexTemplate = pca_coeffs_cuts[2][0,:]

vmin = bstemTemplate.mean() - 2 * bstemTemplate.std()
vmax = bstemTemplate.mean() + 2 * bstemTemplate.std()

plt.figure() 
plt.subplot(1,2,1)
mne.viz.plot_topomap(bstemTemplate, mne.pick_info(A_info_obj_sd[2],np.arange(32)),vmin=vmin,vmax=vmax)
plt.title('Brainstem')

plt.subplot(1,2,2)
mne.viz.plot_topomap(cortexTemplate,  mne.pick_info(A_info_obj_sd[2],np.arange(32)),vmin=vmin,vmax=vmax)
plt.title('Cortex')

#%% Plot Ch. Cz



for sub in range(len(Subjects)):
    plt.figure()
    subject = Subjects[sub]
    
    ch =  31 #Ch. Cz
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_pass[sub][ch_pass_ind,:]
    cz_count = A_Ht_count[sub][ch_count_ind,:]
    
    # cz_pass = -cz_pass
    # cz_count = -cz_count
   
    [troughs_cz1_pass,properties] = find_peaks(-cz_pass)
    [troughs_cz1_count,properties] = find_peaks(-cz_count)

    plt.plot(t,cz_pass, label='Passive', color='k',linewidth=2)
    #plt.scatter(t[troughs_cz1_pass], cz_pass[troughs_cz1_pass], color ='tab:blue')
    
    plt.plot(t,cz_count, label='Count', color='tab:blue', linestyle='dashed')
    #plt.scatter(t[troughs_cz1_count], cz_count[troughs_cz1_count], color ='tab:orange')
    
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_sd[sub_sd][ch_sd_ind,:]
        # cz_sd = -cz_sd
        
        [troughs_cz1_sd,properties] = find_peaks(-cz_sd)
        
        plt.plot(t,cz_sd, label='Shift Detect', color='tab:orange', linestyle ='dotted')
       # plt.scatter(t[troughs_cz1_sd], cz_sd[troughs_cz1_sd], color ='green')
    
        
    
    plt.title(Subjects[sub] + ' Ch. ' + str(ch))
    plt.legend()
    plt.xlim([-0.050,0.5])
    
#%% Plot Ch. Cz with epochs (so can have confidence intervals)
    
fig,ax = plt.subplots(nrows=2,ncols=4,sharex=True)
ax = np.reshape(ax,8)

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    
    ch = 31
    
    if sub !=4:
        ax[sub].axes.yaxis.set_visible(False)
        
    if sub < len(Subjects)/2:
        ax[sub].axes.xaxis.set_visible(False)
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_epochs_pass[sub][ch_pass_ind,:,:]
    cz_count = A_Ht_epochs_count[sub][ch_count_ind,:,:]
    
    cz_pass_sem = cz_pass.std(axis=0) / np.sqrt(cz_pass.shape[0])
    cz_count_sem = cz_count.std(axis=0) / np.sqrt(cz_count.shape[0])
    
    cz_pass = cz_pass.mean(axis=0)
    cz_count = cz_count.mean(axis=0)
    
    ax[sub].plot(t_epochs,cz_pass, label='Passive', color='k',linewidth=2)
    ax[sub].fill_between(t_epochs,cz_pass-cz_pass_sem, cz_pass+cz_pass_sem,color='k',alpha=0.5)
    
    ax[sub].plot(t_epochs,cz_count, label='Count', color='tab:blue')
    ax[sub].fill_between(t_epochs,cz_count-cz_count_sem, cz_count+cz_count_sem,color='tab:blue',alpha=0.5)
    
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_epochs_sd[sub_sd][ch_sd_ind,:,:]
        
        cz_sd_sem = cz_sd.std(axis=0) / np.sqrt(cz_sd.shape[0])
        cz_sd = cz_sd.mean(axis=0)
        
        ax[sub].plot(t_epochs,cz_sd, label='Shift Detect', color='tab:orange')
        ax[sub].fill_between(t_epochs,cz_sd - cz_sd_sem, cz_sd + cz_sd_sem, color='tab:orange',alpha=0.5)
    
    ax[sub].set_title('S' + str(sub+1))
    ax[sub].set_xlim([-0.050,0.1]) 
    ax[sub].set_xticks([0,0.050,0.1])
    #ax[sub].set_xticks([0,0.1,0.2,0.4])
    
ax[1].legend(fontsize=9)
ax[4].axes.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[4].set_xlabel('Time (sec)')
ax[4].set_ylabel('Amplitude')
    
#%% Manual Peak Cutoff Reading CZ and FP
    
cuts_passive = []
cuts_count = []
cuts_sd = []

#S207
cuts_passive.append([.0185, .036, .067, .136 ,.266, 0.5, 0.75])
cuts_count.append([.018, .034, .068,  .130, .266, 0.5,0.75])
cuts_sd.append([.014, .032, .068, .121, .270, .449, 0.75])

#S228
cuts_passive.append([.016, .043, .069, .125, .253, 0.5, 0.75])
cuts_count.append([.014, .040, .070, .127, .242, 0.5, 0.75 ])
cuts_sd.append([.013, .040, .067, .121, .220, 0.5, 0.75]) #0.377

#S236
cuts_passive.append([.014, .031, .066, .124, .249, 0.5, 0.75])
cuts_count.append([.014, .029, .067, .125, .260, 0.5, 0.75 ])
cuts_sd.append([.013, .035, .068, .124, .236, 0.5, 0.75]) 

#238
cuts_passive.append([.0155, .045, .067, .124, .287, 0.5, 0.75])
cuts_count.append([.015, .048, .071, .120, .273, 0.5, 0.75 ])
cuts_sd.append([.014, .045, .068, .115, .340, 0.5 , .75]) #.032, .182

#239
cuts_passive.append([.016, .036, .064, .119, .241, 0.5, 0.75])
cuts_count.append([.014, .036, .068, .112, .240, 0.5, 0.75 ])
cuts_sd.append([.012, .033, .063, .118, .241, 0.5, 0.75  ])

#246
cuts_passive.append([.016, .030, .062, .130, .258, 0.5, 0.75 ])
cuts_count.append([.013, .033, .061, .132, .250, 0.5, 0.75 ])

#247
cuts_passive.append([.018, .042, .071, .120, .220, 0.5, 0.75 ])
cuts_count.append([.017, .042, .070, .120, .220, 0.5, 0.75  ])

#250
cuts_passive.append([.016, .049, .123, .220, .320, 0.5, 0.75])
cuts_count.append([.0145, .049, .123, .220, .320, 0.5, 0.75 ])
cuts_sd.append([.0135, .047, .124, .240, .320, 0.5, 0.75 ])

#%% Plot panel of Subjects Ch. Cz with t_cuts
colors = ['tab:blue','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
axs = []

plt.figure()
for sub in range(len(Subjects)):
    subject = Subjects[sub]
    ch =  31 #Ch. Cz
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_pass[sub][ch_pass_ind,:]
    cz_count = A_Ht_count[sub][ch_count_ind,:]
    
    if sub > 0:
        ax = plt.subplot(2,4,sub+1,sharex=axs[-1])
    else:
        ax = plt.subplot(2,4,sub+1)
            

    for tc in range(len(cuts_passive[sub]) - 2):
        if tc ==0:
            t1_pass = np.where(t>=0)[0][0]
            t1_count =  np.where(t>=0)[0][0]
        else:
            t1_pass = np.where(t>=cuts_passive[sub][tc-1])[0][0]
            t1_count = np.where(t>=cuts_count[sub][tc-1])[0][0]
            
        t2_pass = np.where(t>=cuts_passive[sub][tc])[0][0]
        t2_count = np.where(t>=cuts_count[sub][tc])[0][0]
        
        peaks_pass, props = find_peaks(cz_pass[t1_pass:t2_pass])
        peaks_count, props = find_peaks(cz_count[t1_count:t2_count])
        
        
        ax.plot(t[t1_pass:t2_pass],cz_pass[t1_pass:t2_pass] -cz_pass[t1_pass:t2_pass].mean() , label='Passive', color='k',linewidth=2)
        ax.scatter(t[t1_pass:t2_pass][peaks_pass],(cz_pass[t1_pass:t2_pass] -cz_pass[t1_pass:t2_pass].mean())[peaks_pass], color='k', marker='x')
        
        ax.plot(t[t1_count:t2_count],cz_count[t1_count:t2_count] - cz_count[t1_count:t2_count].mean() , label='Easy', color='grey', linestyle='dashed')
        ax.scatter(t[t1_count:t2_count][peaks_count],(cz_count[t1_count:t2_count] -cz_count[t1_count:t2_count].mean())[peaks_count], color='k', marker='x')
        
    
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_sd[sub_sd][ch_sd_ind,:]
        
        for tc in range(len(cuts_sd[sub_sd]) - 2):
            if tc ==0:
                t1_sd =  np.where(t>=0)[0][0]
            else:
                t1_sd = np.where(t>=cuts_sd[sub_sd][tc-1])[0][0]
                
            t2_sd = np.where(t>=cuts_sd[sub_sd][tc])[0][0]
        
            ax.plot(t[t1_sd:t2_sd], cz_sd[t1_sd:t2_sd] - cz_sd[t1_sd:t2_sd].mean(), label='Hard', color=colors[tc], linestyle ='dashed')
        
    ax.set_title(Subjects[sub] + ' Ch. Cz', fontweight='bold')
    ax.set_yticks([-.002,.002])
    
    ax.axes.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
    ax.set_xticks([0,0.2,0.35])
    plt.xlim([-0.050,0.350])
    # ax.set_xticks([0,0.005,0.010,.020])
    # plt.xlim([-0.002,0.020])
    axs.append(ax)
    
axs[4].set_xlabel('Time (s)',fontweight='bold')
axs[4].set_ylabel('Amplitude',fontweight='bold')
axs[5].legend(['Passive', 'Easy'])

#%% Plot freq spectrum

colors = ['tab:blue','tab:green','tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
fs =4096
axs = []
plt.figure()
for sub in range(len(Subjects)):
    subject = Subjects[sub]
    ch =  31 #Ch. Cz
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_pass[sub][ch_pass_ind,:]
    cz_count = A_Ht_count[sub][ch_count_ind,:]
    
    if sub > 0:
        ax = plt.subplot(2,4,sub+1,sharex=axs[-1])
    else:
        ax = plt.subplot(2,4,sub+1)
            

    for tc in range(len(cuts_passive[sub]) - 2):
        if tc ==0:
            t1_pass = np.where(t>=0)[0][0]
            t1_count =  np.where(t>=0)[0][0]
        else:
            t1_pass = np.where(t>=cuts_passive[sub][tc-1])[0][0]
            t1_count = np.where(t>=cuts_count[sub][tc-1])[0][0]
            
        t2_pass = np.where(t>=cuts_passive[sub][tc])[0][0]
        t2_count = np.where(t>=cuts_count[sub][tc])[0][0]
        
        cz_pass_tc = cz_pass[t1_pass:t2_pass] - cz_pass[t1_pass:t2_pass].mean()
        cz_count_tc = cz_count[t1_count:t2_count] - cz_count[t1_count:t2_count].mean()
        
        cz_pass_tc_f = np.fft.fft(cz_pass_tc)
        cz_pass_f = np.fft.fftfreq(cz_pass_tc_f.size,d=1/fs)
        
        cz_count_tc_f = np.fft.fft(cz_count_tc)
        cz_count_f = np.fft.fftfreq(cz_count_tc_f.size,d=1/fs)
        
        
        ax.plot(cz_pass_f,np.abs(cz_pass_tc_f) , label='Passive', color=colors[tc])
        ax.plot(cz_count_f,np.abs(cz_count_tc_f), label='Active',color=colors[tc],linestyle='--')

        
    
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_sd[sub_sd][ch_sd_ind,:]
        
        for tc in range(len(cuts_sd[sub_sd]) - 2):
            if tc ==0:
                t1_sd =  np.where(t>=0)[0][0]
            else:
                t1_sd = np.where(t>=cuts_sd[sub_sd][tc-1])[0][0]
                
            t2_sd = np.where(t>=cuts_sd[sub_sd][tc])[0][0]
            
            cz_sd_tc =  cz_sd[t1_sd:t2_sd] - cz_sd[t1_sd:t2_sd].mean()
            
            cz_sd_tc_f = np.fft.fft(cz_sd_tc)
            cz_sd_f = np.fft.fftfreq(cz_sd_tc.size,d=1/fs)
        
            ax.plot(cz_sd_f,np.abs(cz_sd_tc_f), label='Hard', color=colors[tc], linestyle =':', linewidth=2)
        
    ax.set_title(Subjects[sub] + ' Ch. Cz', fontweight='bold')
    #ax.set_yticks([-.002,.002])
    
    ax.axes.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
    #ax.set_xticks([0,0.2,0.35])
   # plt.xlim([-0.050,0.350])
    # ax.set_xticks([0,0.005,0.010,.020])
    plt.xlim([0,150])
    axs.append(ax)
    
axs[4].set_xlabel('Time (s)',fontweight='bold')
axs[4].set_ylabel('Amplitude',fontweight='bold')


#%% Get responses from templates
colors = ['tab:blue','tab:orange','green','red','purple', 'brown', 'pink']

# bs_sub_sp_pass = [[0]*3]*6

# bs_sub_expVar_pass = np.zeros([len(cuts_passive[0]),len(Subjects)])
# ctx_sub_expVar_pass = np.zeros([len(cuts_passive[0]),len(Subjects)])

# bs_sub_expVar_count = np.zeros([len(cuts_count[0]), len(Subjects)])
# ctx_sub_expVar_count = np.zeros([len(cuts_count[0]), len(Subjects)])

# bs_sub_expVar_sd = np.zeros([len(cuts_sd[0]), len(Subjects_sd)])
# ctx_sub_expVar_sd = np.zeros([len(cuts_sd[0]), len(Subjects_sd)])



# for sub in range(len(Subjects)):
#     t_tc_pass, bstem_tc_sp_pass, bstem_tc_expVar_pass = Template_tcuts(A_Ht_pass[sub], t, cuts_passive[sub], A_ch_picks_pass[sub], bstemTemplate)
#     t_tc_count, bstem_tc_sp_count, bstem_tc_expVar_count = Template_tcuts(A_Ht_count[sub], t, cuts_count[sub], A_ch_picks_count[sub], bstemTemplate)

    
#     t_tc_pass, ctx_tc_sp_pass, ctx_tc_expVar_pass = Template_tcuts(A_Ht_pass[sub], t, cuts_passive[sub], A_ch_picks_pass[sub], cortexTemplate)
#     t_tc_count,ctx_tc_sp_count, ctx_tc_expVar_count = Template_tcuts(A_Ht_count[sub], t, cuts_count[sub], A_ch_picks_count[sub], cortexTemplate)
    
#     bs_sub_expVar_pass[:,sub] = bstem_tc_expVar_pass
#     bs_sub_expVar_count[:,sub] = bstem_tc_expVar_count
#     ctx_sub_expVar_pass[:,sub] = ctx_tc_expVar_pass
#     ctx_sub_expVar_count[:,sub] = ctx_tc_expVar_count
    
#     if Subjects[sub] in Subjects_sd:
#         sub_sd = Subjects_sd.index(Subjects[sub])
#         t_tc_sd, bstem_tc_sp_sd, bstem_tc_expVar_sd = Template_tcuts(A_Ht_sd[sub_sd], t, cuts_sd[sub_sd], A_ch_picks_sd[sub_sd], bstemTemplate)
#         t_tc_sd, ctx_tc_sp_sd, ctx_tc_expVar_sd = Template_tcuts(A_Ht_sd[sub_sd], t, cuts_sd[sub_sd], A_ch_picks_sd[sub_sd], cortexTemplate)
        
#         bs_sub_expVar_sd[:,sub_sd] = bstem_tc_expVar_sd
#         ctx_sub_expVar_sd[:,sub_sd] = ctx_tc_expVar_sd
        
    
    
#     fig,ax=plt.subplots(2,1)
#     for tc in range(len(t_tc_pass)):
#         # if tc % 2 == 0:
#         #     y_tloc = np.min(bstem_tc_sp_pass[tc]) - 0.1 * np.min(bstem_tc_sp_pass[tc])
#         # else:
#         #     y_tloc = np.max(bstem_tc_sp_pass[tc]) + 0.1 * np.max(bstem_tc_sp_pass[tc])
        
#         ax[0].plot(t_tc_pass[tc],bstem_tc_sp_pass[tc], color=colors[tc])
#         ax[0].plot(t_tc_count[tc],bstem_tc_sp_count[tc], color=colors[tc], linestyle = 'dashed')
        
        
#         #ax[0].text(t_tc_pass[tc][0], y_tloc, str(np.round(bstem_tc_expVar_pass[tc]*100)) + '%', color = colors[tc])
        
#         ax[1].plot(t_tc_pass[tc],ctx_tc_sp_pass[tc], color=colors[tc])
#         ax[1].plot(t_tc_count[tc],ctx_tc_sp_count[tc], color=colors[tc], linestyle='dashed')
        
#         if Subjects[sub] in Subjects_sd:
#             ax[0].plot(t_tc_sd[tc],bstem_tc_sp_sd[tc], color=colors[tc], linestyle = 'dotted')
#             ax[1].plot(t_tc_sd[tc],ctx_tc_sp_sd[tc], color=colors[tc], linestyle = 'dotted')
        
#         #ax[1].text(t_tc_pass[tc][0], np.max(ctx_tc_sp_pass[tc]), str(np.round(ctx_tc_expVar_pass[tc]*100)) + '%', color = colors[tc])
        
    
#     ax[0].set_title(Subjects[sub] + ' Brainstem')
#     ax[1].set_title('Cortex')
    
#%% Compare Exp Variance

# mean_pass_bs = bs_sub_expVar_pass.mean(axis=1)
# sem_pass_bs = bs_sub_expVar_pass.std(axis=1) / np.sqrt(bs_sub_expVar_pass.shape[1])

# mean_pass_ctx = ctx_sub_expVar_pass.mean(axis=1)
# sem_pass_ctx = ctx_sub_expVar_pass.std(axis=1) / np.sqrt(ctx_sub_expVar_pass.shape[1])

# x = np.arange(0,7) + 1

# mean_pass_bs = mean_pass_bs[0:5]
# sem_pass_bs = sem_pass_bs[0:5]
# mean_pass_ctx = mean_pass_ctx[0:5]
# sem_pass_ctx = sem_pass_ctx[0:5]
# x = x[0:5]

# plt.figure()
# plt.bar(x,mean_pass_bs,width=0.25,yerr=sem_pass_bs, label='Brainstem')
# plt.bar(x+0.25,mean_pass_ctx,width=0.25,yerr=sem_pass_ctx, label='Cortex')
# plt.ylabel('Explained Variance')
# plt.title('Passive')
# plt.legend()

# # count
# mean_count_bs = bs_sub_expVar_count.mean(axis=1)
# sem_count_bs = bs_sub_expVar_count.std(axis=1) / np.sqrt(bs_sub_expVar_count.shape[1])

# mean_count_ctx = ctx_sub_expVar_count.mean(axis=1)
# sem_count_ctx = ctx_sub_expVar_count.std(axis=1) / np.sqrt(ctx_sub_expVar_count.shape[1])

# x = np.arange(0,7) + 1

# mean_count_bs = mean_count_bs[0:5]
# sem_count_bs = sem_count_bs[0:5]
# mean_count_ctx = mean_count_ctx[0:5]
# sem_count_ctx = sem_count_ctx[0:5]
# x = x[0:5]

# plt.figure()
# plt.bar(x,mean_count_bs,width=0.25,yerr=sem_count_bs, label='Brainstem')
# plt.bar(x+0.25,mean_count_ctx,width=0.25,yerr=sem_count_ctx, label='Cortex')
# plt.ylabel('Explained Variance')
# plt.title('Counting Task')
# plt.legend()

# #SD
# mean_sd_bs = bs_sub_expVar_sd.mean(axis=1)
# sem_sd_bs = bs_sub_expVar_sd.std(axis=1) / np.sqrt(bs_sub_expVar_sd.shape[1])

# mean_sd_ctx = ctx_sub_expVar_sd.mean(axis=1)
# sem_sd_ctx = ctx_sub_expVar_sd.std(axis=1) / np.sqrt(ctx_sub_expVar_sd.shape[1])

# x = np.arange(0,7) + 1

# mean_sd_bs = mean_sd_bs[0:5]
# sem_sd_bs = sem_sd_bs[0:5]
# mean_sd_ctx = mean_sd_ctx[0:5]
# sem_sd_ctx = sem_sd_ctx[0:5]
# x = x[0:5]


# plt.figure()
# plt.bar(x,mean_sd_bs,width=0.25,yerr=sem_sd_bs, label='Brainstem')
# plt.bar(x+0.25,mean_sd_ctx,width=0.25,yerr=sem_sd_ctx, label='Cortex')
# plt.ylabel('Explained Variance')
# plt.title('Shift Detect Task')
# plt.legend()


#%% Compute Latency, Peak Mag, Peak area from channel Cz

latency_pass = np.zeros((5,len(Subjects))) #5 sources
p2p_pass = np.zeros((5,len(Subjects)))
pkArea_pass = np.zeros((5,len(Subjects)))

latency_count = np.zeros((5,len(Subjects))) #5 sources
p2p_count = np.zeros((5,len(Subjects)))
pkArea_count = np.zeros((5,len(Subjects)))

latency_sd = np.zeros((5,len(Subjects_sd))) #5 sources
p2p_sd = np.zeros((5,len(Subjects_sd)))
pkArea_sd = np.zeros((5,len(Subjects_sd)))

fs = 4096

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    ch =  31 #Ch. Cz
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_pass[sub][ch_pass_ind,:]
    cz_count = A_Ht_count[sub][ch_count_ind,:]

    for tc in range(len(cuts_passive[sub]) - 2):
        if tc ==0:
            t1_pass = np.where(t>=0)[0][0]
            t1_count =  np.where(t>=0)[0][0]
        else:
            t1_pass = np.where(t>=cuts_passive[sub][tc-1])[0][0]
            t1_count = np.where(t>=cuts_count[sub][tc-1])[0][0]
            
        t2_pass = np.where(t>=cuts_passive[sub][tc])[0][0]
        t2_count = np.where(t>=cuts_count[sub][tc])[0][0]
        
        cz_pass_tc = cz_pass[t1_pass:t2_pass] -cz_pass[t1_pass:t2_pass].mean()
        cz_count_tc = cz_count[t1_count:t2_count] - cz_count[t1_count:t2_count].mean()
        
        peaks_pass, props = find_peaks(cz_pass_tc)
        peaks_count, props = find_peaks(cz_count_tc)
        
        if (peaks_pass.size == 0):
            latency_pass[tc,sub] = np.nan

        else:
            which_pk = np.argmax(cz_pass_tc[peaks_pass])
            latency_pass[tc,sub] = t[t1_pass:t2_pass][peaks_pass[which_pk]]
            
            
        if(peaks_count.size ==0):
            latency_count[tc,sub] = np.nan
        else:
            which_pk = np.argmax(cz_count_tc[peaks_count])
            latency_count[tc,sub] = t[t1_count:t2_count][peaks_count[which_pk]]
            
            
        p2p_pass[tc,sub] = cz_pass_tc.max() - cz_pass_tc.min()
        pkArea_pass[tc,sub] = np.sum(np.abs(cz_pass_tc)) / fs
        
        
        p2p_count[tc,sub] = cz_count_tc.max() - cz_count_tc.min()
        pkArea_count[tc,sub] = np.sum(np.abs(cz_count_tc)) / fs
        
        
   
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_sd[sub_sd][ch_sd_ind,:]
        
        for tc in range(len(cuts_sd[sub_sd]) - 2):
            if tc ==0:
                t1_sd =  np.where(t>=0)[0][0]
            else:
                t1_sd = np.where(t>=cuts_sd[sub_sd][tc-1])[0][0]
                
            t2_sd = np.where(t>=cuts_sd[sub_sd][tc])[0][0]
            
            cz_sd_tc = cz_sd[t1_sd:t2_sd] - cz_sd[t1_sd:t2_sd].mean()
            
            peaks_sd,pros = find_peaks(cz_sd_tc)
            
            if(peaks_sd.size==0):
                latency_sd[tc,sub_sd] = np.nan
            else:
                which_pk = np.argmax(cz_sd_tc[peaks_sd])
                latency_sd[tc,sub_sd] = t[t1_sd:t2_sd][peaks_sd[which_pk]]
                
            p2p_sd[tc,sub_sd] = cz_sd_tc.max() - cz_sd_tc.min()
            pkArea_sd[tc,sub_sd] = np.sum(np.abs(cz_sd_tc)) / fs


#Fix mislabels
latency_pass[1,1] = np.nan #S228, this one is not real peak

latency_sd[2,2] = np.nan #S236, not much of a peak

latency_pass[2,3] = np.nan #S238 does not have a peak for this source
latency_count[2,3] = np.nan
latency_sd[2,3] = np.nan


Sub_sdInds = np.zeros([len(Subjects_sd)],dtype=int)
for s_sd in range(len(Subjects_sd)):
    Sub_sdInds[s_sd] = Subjects.index(Subjects_sd[s_sd])


latency_count_change = latency_count - latency_pass
latency_sd_change = latency_sd - latency_pass[:,Sub_sdInds]

p2p_norm = p2p_pass[0,:]
p2p_pass = p2p_pass / p2p_norm
p2p_count = p2p_count / p2p_norm
p2p_sd = p2p_sd / p2p_norm[Sub_sdInds]

p2p_count_change =  p2p_count - p2p_pass
p2p_sd_change = p2p_sd - p2p_pass[:,Sub_sdInds]


#%% Compute features through frequency domain

latency_pass_f = np.zeros((5,len(Subjects))) #5 sources
pkWidth_pass_f = np.zeros((5,len(Subjects)))

latency_count_f = np.zeros((5,len(Subjects))) #5 sources
pkWidth_count_f = np.zeros((5,len(Subjects)))

latency_sd_f = np.zeros((5,len(Subjects_sd))) #5 sources
pkWidth_sd_f = np.zeros((5,len(Subjects_sd)))

fs = 4096

for sub in range(len(Subjects)):
    subject = Subjects[sub]
    ch =  31 #Ch. Cz
    
    ch_pass_ind = np.where(A_ch_picks_pass[sub] == ch)[0][0]
    ch_count_ind = np.where(A_ch_picks_count[sub] == ch)[0][0]
    
    cz_pass = A_Ht_pass[sub][ch_pass_ind,:]
    cz_count = A_Ht_count[sub][ch_count_ind,:]

    for tc in range(len(cuts_passive[sub]) - 2):
        if tc ==0:
            t1_pass = np.where(t>=0)[0][0]
            t1_count =  np.where(t>=0)[0][0]
        else:
            t1_pass = np.where(t>=cuts_passive[sub][tc-1])[0][0]
            t1_count = np.where(t>=cuts_count[sub][tc-1])[0][0]
            
        t2_pass = np.where(t>=cuts_passive[sub][tc])[0][0]
        t2_count = np.where(t>=cuts_count[sub][tc])[0][0]
        
        cz_pass_tc = cz_pass[t1_pass:t2_pass] -cz_pass[t1_pass:t2_pass].mean()
        cz_count_tc = cz_count[t1_count:t2_count] - cz_count[t1_count:t2_count].mean()
        
        cz_pass_tc_f = np.fft.fft(cz_pass_tc)
        cz_count_tc_f = np.fft.fft(cz_count_tc)
        f = np.ff

   
    if (subject in Subjects_sd):
        sub_sd = Subjects_sd.index(subject)
        ch_sd_ind = np.where(A_ch_picks_sd[sub_sd] == ch)[0][0]
        cz_sd = A_Ht_sd[sub_sd][ch_sd_ind,:]
        
        for tc in range(len(cuts_sd[sub_sd]) - 2):
            if tc ==0:
                t1_sd =  np.where(t>=0)[0][0]
            else:
                t1_sd = np.where(t>=cuts_sd[sub_sd][tc-1])[0][0]
                
            t2_sd = np.where(t>=cuts_sd[sub_sd][tc])[0][0]
            
            cz_sd_tc = cz_sd[t1_sd:t2_sd] - cz_sd[t1_sd:t2_sd].mean()
            

#%% Non Parametric statistics





#%% Average Passive vs Average Active

t_cuts_pass = [.016, .028, .066, .123, .250 ]
t_cuts_sd = [.013, .035, .066, .123, .250 ]

colors = ['tab:blue','green','red','purple', 'brown', 'pink']


cz_pass_subs = np.zeros([A_Ht_pass[0].shape[1], len(Subjects)])
for sub in range(len(Subjects)):
    cz_pass_subs[:,sub] = A_Ht_pass[sub][-1,:]
    
cz_sd_subs = np.zeros([A_Ht_sd[0].shape[1],len(Subjects)])
for sub in range(len(Subjects_sd)):
    cz_sd_subs[:,sub] = A_Ht_sd[sub][-1,:]
    
    
cz_pass_sem = cz_pass_subs.std(axis=1) / np.sqrt(cz_pass_subs.shape[1])
cz_pass_mean = cz_pass_subs.mean(axis=1)

cz_sd_sem = cz_sd_subs.std(axis=1) / np.sqrt(cz_sd_subs.shape[1])
cz_sd_mean = cz_sd_subs.mean(axis=1)

plt.figure()
plt.plot(t,cz_pass_mean,label='Passive',color='k')

for t_c in range(len(t_cuts_pass)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts_pass[t_c-1])[0][0]
    t_2 = np.where(t>=t_cuts_pass[t_c])[0][0]
        
    plt.fill_between(t[t_1:t_2],cz_pass_mean[t_1:t_2] - cz_pass_sem[t_1:t_2], cz_pass_mean[t_1:t_2] + cz_pass_sem[t_1:t_2],alpha=0.5,color=colors[t_c])

plt.plot(t,cz_sd_mean,color='tab:orange', label='Shift Detection')

for t_c in range(len(t_cuts_sd)):
    if t_c ==0:
        t_1 = np.where(t>=0)[0][0]
    else:
        t_1 = np.where(t>=t_cuts_sd[t_c-1])[0][0]
    t_2 = np.where(t>=t_cuts_sd[t_c])[0][0]
        
    plt.fill_between(t[t_1:t_2],cz_sd_mean[t_1:t_2] - cz_sd_sem[t_1:t_2], cz_sd_mean[t_1:t_2] + cz_sd_sem[t_1:t_2],alpha=0.5,color=colors[t_c])

#plt.fill_between(t,cz_sd_mean-cz_sd_sem,cz_sd_mean+cz_sd_sem,alpha=0.5,color='tab:orange')
    
plt.xlim((-0.050,0.3))
plt.xticks([0,0.05,0.1,0.15,0.2])
plt.yticks([-.002,0,.002])
plt.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.legend()

#%% PCA topomaps for Passive and Active

Ht_pass_avg = ACR_helperFuncs.Average_Subjects(A_Ht_pass,A_ch_picks_pass,32)
Ht_sd_avg = ACR_helperFuncs.Average_Subjects(A_Ht_sd,A_ch_picks_sd,32)


pca_sp_cuts_pass, pca_expVar_cuts_pass, pca_coeff_cuts_pass, t_cuts_pass = \
    ACR_helperFuncs.PCA_tcuts(Ht_pass_avg,t,t_cuts_pass,np.arange(32),np.arange(32))

pca_sp_cuts_sd, pca_expVar_cuts_sd, pca_coeff_cuts_sd, t_cuts_sd = \
    ACR_helperFuncs.PCA_tcuts(Ht_sd_avg,t,t_cuts_sd,np.arange(32),np.arange(32))


ACR_helperFuncs.PCA_tcuts_topomap(pca_coeff_cuts_pass, t_cuts_pass,
                                  pca_expVar_cuts_pass, 
                                  mne.pick_info(info_obj,np.arange(32)), 
                                  np.arange(32), 'Passive')

          
ACR_helperFuncs.PCA_tcuts_topomap(pca_coeff_cuts_sd, t_cuts_sd,
                                  pca_expVar_cuts_sd, 
                                  mne.pick_info(info_obj,np.arange(32)), 
                                  np.arange(32), 'Shift Detection')
          
          
          
  
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
