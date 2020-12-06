#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:02:23 2020

@author: ravinderjit
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import norm
# from scipy.io import loadmat 
# import matplotlib.colors as mcolors
# import pickle

def GaussCDF(x,sigma,mu):
    return 0.5 *(1+erf((x-mu)/(sigma * np.sqrt(2))))

# StimData_2_10 = loadmat('/home/ravinderjit/Documents/OnlineStim_WavFiles/CMR_frozenMod/Mod_' + str(Mod[0]) +'_'+str(Mod[1]) +'/Stim_Data.mat')
# correct = StimData_2_10['correct'].squeeze()
# SNRdB_exp = StimData_2_10['SNRdB_exp'].squeeze()
# SNRdB = SNRdB_exp[:,0]
# coh = SNRdB_exp[:,1]

Results_fname = []
Results_fname.append('CMR3AFC_2_10_frozen_Rav_results.json')
Results_fname.append('CMR3AFC_36_44_frozen_Rav_results.json')
Results_fname.append('CMR3AFC_131_139f_Rav_results.json')


ASNR =[]
ACoh =[]
Aresps = []
Acorr = []
Asubjects = []

for R_fname in Results_fname:
    with open(R_fname) as f:
        results = json.load(f)
    
    if R_fname == 'CMR3AFC_2_10_frozen_Rav_results.json':
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
        
        

Mod_labels = ['2-10 Hz', '36-44 Hz', '131-139 Hz']
CMR = np.zeros(3)
for task in range(3):
    if task ==0:
        SNRs_1 = [4,-2,-8,-14,-20,-26,-32,-38]
    else:
        SNRs_1 = [6,0,-6,-12,-18,-24,-30]
    
    SNRs_2 = [6,0,-6,-12,-18,-24,-30]
    
    SNR_t = ASNR[task]
    Coh_t = ACoh[task]
    resp_t = Aresps[task]
    corr_t = Acorr[task]
    subjects_t = Asubjects[task]
    
    if task == 1: #remove demo subject
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
        
    param_1, pcov = curve_fit(GaussCDF,SNRs_1,SNRs_1_acc.mean(axis=1),[1,-10])
    param_2, pcov = curve_fit(GaussCDF,SNRs_2,SNRs_2_acc.mean(axis=1),[1,-10])
    
    x1 = np.arange(SNRs_1[-1],SNRs_1[0],1)
    x2 = np.arange(SNRs_2[-1],SNRs_2[0],1)
    psycho1 = GaussCDF(x1,param_1[0],param_1[1])
    psycho2 =GaussCDF(x2,param_2[0],param_2[1])
    
    CMR[task] =  norm.ppf(0.75, param_2[1],param_2[0])- norm.ppf(0.75,param_1[1],param_1[0])
    
        
    Mean_SNR = SNRs_1_acc
    plt.figure()
    plt.plot(SNRs_1, SNRs_1_acc,color='blue',label='Coh')
    plt.plot(SNRs_2, SNRs_2_acc,color='red',label='Incoh')
    plt.legend()
    plt.title(Mod_labels[task])

    fig, ax = plt.subplots(figsize=(4,3))
    fontsize = 8
    ax.errorbar(SNRs_1, SNRs_1_acc.mean(axis=1),SNRs_1_acc.std(axis=1) / np.sqrt(SNRs_1_acc.shape[1]) ,label='Coh',linewidth=2)
    ax.errorbar(SNRs_2, SNRs_2_acc.mean(axis=1),SNRs_2_acc.std(axis=1) /np.sqrt(SNRs_2_acc.shape[1]) ,label='Incoh',linewidth=2)
    # plt.plot(x1,psycho1)
    # plt.plot(x2,psycho2)
    if task==0:
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
    
    











