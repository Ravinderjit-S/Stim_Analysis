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
from scipy.io import savemat 
import psignifit as ps
import os


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
thresh_70 = np.zeros(len(subjects))
lapse = np.zeros(len(subjects))

psCurves = []

plt.figure()
for sub in range(len(subjects)):
    data_sub = np.concatenate((SNRs[:,np.newaxis], accuracy[:,sub][:,np.newaxis] *cond_trials[:,np.newaxis] , cond_trials[:,np.newaxis] ),axis=1)
    result_sub = ps.psignifit(data_sub,options)
    thresh_70[sub] = ps.getThreshold(result_sub,0.70)[0]
    lapse[sub] = result_sub['Fit'][2]
    
    result_ps.append(result_sub)
    ps.psigniplot.plotPsych(result_sub)
    
    
    #%% Store curves to look at a verage
    x_vals  = np.linspace(-13, 13, num=1000)
    
    fit = result_sub['Fit']
    data = result_sub['data']
    options = result_sub['options']

    fitValues = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x_vals,     fit[0], fit[1]) + fit[3]
    
    psCurves.append(fitValues)
    
#%% Clean subject names
#Some subjects had to put modify subject name due to techincal difficulty. fixing here

for sub in range(len(subjects)):
    thisSub = list(subjects[sub])
    
    if thisSub[0] == 's':
        thisSub[0] = 'S'
    
    if len(thisSub) > 4:
        thisSub = thisSub[:4]
        
    subjects[sub] = "".join(thisSub)
    

#%% Plot average curves

psCurves = np.array(psCurves)
ps_mean = psCurves.mean(axis=0)
ps_sem = psCurves.std(axis=0) / np.sqrt(psCurves.shape[0])

fig = plt.figure()
fig.set_size_inches(8,8)
plt.rcParams.update({'font.size': 15})
plt.plot(x_vals,ps_mean, color = 'Black',linewidth=2)
plt.fill_between(x_vals, ps_mean - ps_sem, ps_mean + ps_sem, color='black',alpha =  0.5)
plt.xlim([-13,13])
#plt.xticks([-60,-40,-20])
plt.yticks([0.2,0.6,1])
plt.ylim([0.1, 1.03])
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy')

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/'
plt.savefig(os.path.join(fig_loc,'MRT_psCurve.svg'),format='svg')

#%% Make Box Plot

fig = plt.figure()
fig.set_size_inches(7,8)
plt.rcParams.update({'font.size': 15})
whisker =plt.boxplot(thresh_70)
#whisker['medians'][0].linewidth = 4
plt.xticks([])
plt.yticks([-5,-3, -1])
plt.ylabel('SNR (dB)')

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/'
plt.savefig(os.path.join(fig_loc,'MRT_box.svg'),format='svg')




#%% Save data

savemat('MTB_MRT.mat',{'Subjects':subjects,'thresholds': thresh_70, 'lapse':lapse})






