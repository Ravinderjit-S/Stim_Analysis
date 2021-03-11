#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:02:23 2020

@author: ravinderjit
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import norm
import os
# from scipy.io import loadmat 
# import matplotlib.colors as mcolors
# import pickle

def GaussCDF(x,sigma,mu):
    return 0.5 *(1+erf((x-mu)/(sigma * np.sqrt(2)))) * (1-1/3)  + 1/3  # 1/3 for 3 AFC

# StimData_2_10 = loadmat('/home/ravinderjit/Documents/OnlineStim_WavFiles/CMR_frozenMod/Mod_' + str(Mod[0]) +'_'+str(Mod[1]) +'/Stim_Data.mat')
# correct = StimData_2_10['correct'].squeeze()
# SNRdB_exp = StimData_2_10['SNRdB_exp'].squeeze()
# SNRdB = SNRdB_exp[:,0]
# coh = SNRdB_exp[:,1]

Results_fname = []
Results_fname.append('CMR3AFC_2_10_frozen_Rav_results.json')
Results_fname.append('CMR3AFC_16_24f_Rav_results.json')
Results_fname.append('CMR3AFC_36_44_frozen_Rav_results.json')
Results_fname.append('CMR3AFC_131_139f_Rav_results.json')

fig_path = '/home/ravinderjit/Documents/Figures/CMR/'

ASNR =[]
ACoh =[]
Aresps = []
Acorr = []
Asubjects = []


for R_fname in Results_fname:
    with open(R_fname) as f:
        results = json.load(f)
    
    if (R_fname == 'CMR3AFC_2_10_frozen_Rav_results.json') or (R_fname == 'CMR3AFC_16_24f_Rav_results.json'):
        trials = 105
    else:
        trials = 98
        
    trialnum = np.zeros((trials,len(results)))
    SNR_ = np.zeros((trials,len(results)))
    coh_ = np.zeros((trials,len(results)))
    resps = np.zeros((trials,len(results)))
    corr_check = np.zeros((trials,len(results)))
    subjects = []
    
    for k in range(0,len(results)):
        subjects.append(results[k][0]['subject'])
        subj = results[k]
        cur_t = 0
        for trial in subj:
            if not 'annot' in trial:
                continue
            if len(trial['annot']) == 0:
                continue
            
            trialnum[cur_t,k] = trial['trialnum']
            annot = trial['annot']
            SNR_[cur_t,k] = int(annot[annot.find(':')+1:annot.find('}')])
            coh_[cur_t,k] = trial['cond']
            resps[cur_t,k] = int(trial['button_pressed'])
            corr_check[cur_t,k] = trial['correct']
            cur_t+=1
    resps = resps+1
    coh_ = coh_ -1
    ASNR.append(SNR_)
    ACoh.append(coh_)
    Aresps.append(resps)
    Acorr.append(corr_check)
    Asubjects.append(subjects)
        
        

Mod_labels = ['2-10 Hz', '16-24 Hz' ,'36-44 Hz', '131-139 Hz']
CMR = np.zeros(4)
CMR_SE = np.zeros(4)

A_SNRs_1 = []
A_SNRs_2 = []

A_SNRs_1_acc =[]
A_SNRs_2_acc =[]

for task in range(4):
    if task ==0 or task ==1:
        SNRs_1 = [4,-2,-8,-14,-20,-26,-32,-38]
    else:
        SNRs_1 = [6,0,-6,-12,-18,-24,-30]
    
    SNRs_2 = [6,0,-6,-12,-18,-24,-30]
    
    SNR_t = ASNR[task]
    Coh_t = ACoh[task]
    resp_t = Aresps[task]
    corr_t = Acorr[task]
    subjects_t = Asubjects[task]
    
    if task == 2: #remove demo subject
        SNR_t = SNR_t[:,1:]
        Coh_t = Coh_t[:,1:]
        resp_t = resp_t[:,1:]
        corr_t = corr_t[:,1:]
        subjects_t = subjects_t[1:]
    

    SNRs_1_acc = np.zeros((len(SNRs_1),len(subjects_t)))
    for j in range(0,len(SNRs_1)):
        mask = (SNR_t==SNRs_1[j]) & (Coh_t==1)
        SNRs_1_acc[j,:] = corr_t[mask[:,0],:].sum(axis=0) / corr_t[mask[:,0]].shape[0]
    
        
    SNRs_2_acc = np.zeros((len(SNRs_2),len(subjects_t)))
    for j in range(0,len(SNRs_2)):
        mask = (SNR_t==SNRs_2[j]) & (Coh_t==0)
        SNRs_2_acc[j,:] = corr_t[mask[:,0],:].sum(axis=0)/corr_t[mask[:,0]].shape[0]
        
        
  
    # Jacknife, i.e. leave one out  
    CMR_JN = np.zeros(SNRs_1_acc.shape[1])
    for jn in range(SNRs_1_acc.shape[1]):
        SNRs_1_acc_JN = np.delete(SNRs_1_acc,jn,axis=1)
        SNRs_2_acc_JN = np.delete(SNRs_2_acc,jn,axis=1)
        param_1, pcov = curve_fit(GaussCDF,SNRs_1,SNRs_1_acc_JN.mean(axis=1),[0.5,-15])
        param_2, pcov = curve_fit(GaussCDF,SNRs_2,SNRs_2_acc_JN.mean(axis=1),[0.5,-15])
        x1 = np.arange(SNRs_1[-1],SNRs_1[0],.1)
        x2 = np.arange(SNRs_2[-1],SNRs_2[0],.1)
        psycho1 = GaussCDF(x1,param_1[0],param_1[1])
        psycho2 =GaussCDF(x2,param_2[0],param_2[1])
        CMR_JN[jn] = x2[np.where(psycho2>=0.75)[0][0]] -  x1[np.where(psycho1>=0.75)[0][0]]
    
        
        
    param_1, pcov = curve_fit(GaussCDF,SNRs_1,SNRs_1_acc.mean(axis=1),[0.5,-15])
    param_2, pcov = curve_fit(GaussCDF,SNRs_2,SNRs_2_acc.mean(axis=1),[0.5,-15])
    
    x1 = np.arange(SNRs_1[-1],SNRs_1[0],.1)
    x2 = np.arange(SNRs_2[-1],SNRs_2[0],.1)
    psycho1 = GaussCDF(x1,param_1[0],param_1[1])
    psycho2 =GaussCDF(x2,param_2[0],param_2[1])
    
    CMR[task] =  x2[np.where(psycho2>=0.75)[0][0]] -  x1[np.where(psycho1>=0.75)[0][0]]
    CMR_SE[task] = np.sqrt(np.var(CMR_JN) * (SNRs_1_acc.shape[1]-1))
    
        
    Mean_SNR = SNRs_1_acc
    plt.figure()
    plt.plot(SNRs_1, SNRs_1_acc,color='blue',label='Coh')
    plt.plot(SNRs_2, SNRs_2_acc,color='red',label='Incoh')
    plt.legend()
    plt.title(Mod_labels[task])

    fig, ax = plt.subplots(figsize=(7,5)) #figsize is in inches
    fontsize = 17
    ax.errorbar(SNRs_1, SNRs_1_acc.mean(axis=1),SNRs_1_acc.std(axis=1) / np.sqrt(SNRs_1_acc.shape[1]) ,label='Coh',linewidth=2)
    ax.errorbar(SNRs_2, SNRs_2_acc.mean(axis=1),SNRs_2_acc.std(axis=1) /np.sqrt(SNRs_2_acc.shape[1]) ,label='Incoh',linewidth=2)
    #plt.plot(x1,psycho1,color='b')
    #plt.plot(x2,psycho2,color='r')
    # plt.plot(x1,psycho1)
    # plt.plot(x2,psycho2)
    if task==0 or task ==1:
        plt.xlim([-40, 10])
        plt.xticks([-40, -30, -20, -10, 0],fontsize=fontsize)
    else:
        plt.xlim([-32, 10])
        plt.xticks([-30, -20, -10, 0],fontsize=fontsize)
    plt.ylim([.2,1.02])
    plt.yticks([.25, .5, .75, 1],fontsize=fontsize)
    plt.xlabel('SNR',fontsize=fontsize,fontweight='bold')
    plt.ylabel('Accuracy',fontsize=fontsize,fontweight='bold')
    plt.title(Mod_labels[task],fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    fig.savefig(os.path.join(fig_path, 'CMRrandMod_' + Mod_labels[task][0:4] +'.png'),format='png')
    
    A_SNRs_1.append(SNRs_1)
    A_SNRs_2.append(SNRs_2)
    A_SNRs_1_acc.append(SNRs_1_acc)
    A_SNRs_2_acc.append(SNRs_2_acc)
    
fig,ax = plt.subplots(figsize=(16,12))
fontsize=36
ax.errorbar(range(CMR.size),CMR,CMR_SE,color='k',linewidth=4,marker='.',markersize=60)
#plt.title('CMR',fontsize=fontsize)
plt.xticks(ticks = [0,1,2,3],labels=Mod_labels,fontsize=fontsize)
plt.yticks(ticks =[0, 4, 8, 12],fontsize=fontsize)
plt.ylabel('CMR (dB)',fontsize=fontsize)
plt.xlabel('Noise Modulation',fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig(os.path.join(fig_path, 'CMRrandMod_summary'  +'.png'),format='png')

fig,ax = plt.subplots(figsize=(12,10))
fontsize=35
ptcolors = ['tab:blue', 'tab:orange','tab:green','tab:red']
for p in range(len(A_SNRs_1)):
    ax.errorbar(A_SNRs_1[p], A_SNRs_1_acc[p].mean(axis=1), A_SNRs_1_acc[p].std(axis=1) / np.sqrt(A_SNRs_1_acc[p].shape[1]),
                color=ptcolors[p],label=Mod_labels[p],linewidth=2)
    ax.errorbar(A_SNRs_2[p], A_SNRs_2_acc[p].mean(axis=1), A_SNRs_2_acc[p].std(axis=1) / np.sqrt(A_SNRs_2_acc[p].shape[1]),
            color=ptcolors[p],linestyle='dashed')
plt.legend()
    








