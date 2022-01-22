#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 21:38:51 2020

@author: ravinderjit
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
import psignifit as ps


#%% Load Data

Results_fname = 'MRT_MTB_Rav_results.json'
with open(Results_fname) as f:
    results = json.load(f)
    
    
SNRs = np.arange(8,-11,-3)
subjects = []
accuracy = np.zeros((len(SNRs),len(results)))

pilots = []

#%% Extract Data from Json file

sub_ind = 0
for k in range(0,len(results)):
    if results[k][0]['subject'][:5] == 'PILOT':
        pilots.append(k)
        continue
    
    subjects.append(results[k][0]['subject'])
    
    subj = results[k]
    cur_t = 0
    cond_correct = np.zeros(SNRs.size)
    cond_trials = np.zeros(SNRs.size) 
    for trial in subj:
        if not 'annot' in trial:
            continue
        if len(trial['annot']) == 0:
            continue
    
        cond_t = trial['cond']
        cond_trials[cond_t-1] += 1
        if trial['correct']:
            cond_correct[cond_t-1] += 1
        
        cur_t +=1 
        
    if ~np.all(cond_trials==20):
        print('Something is Fishy!!!!')
    
    accuracy[:,k] = cond_correct / cond_trials
        

accuracy = np.delete(accuracy,pilots,axis=1)

#%% Fit psychometric curves with psignifit

options = dict({
    'sigmoidName': 'norm',
    'expType': 'nAFC',
    'expN': 6
    })


result_ps = []
plt.figure()
for sub in range(len(subjects)):
    data_sub = np.concatenate((SNRs[:,np.newaxis], accuracy[:,sub][:,np.newaxis] *cond_trials[:,np.newaxis] , cond_trials[:,np.newaxis] ),axis=1)
    result_sub = ps.psignifit(data_sub,options)
    
    result_ps.append(result_sub)
    ps.psigniplot.plotPsych(result_sub)

#%% Plot data

plt.figure()
plt.plot(SNRs,accuracy)










