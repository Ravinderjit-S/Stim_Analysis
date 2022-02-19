#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 12:25:40 2022

@author: ravinderjit
"""

import pandas as pd
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt

data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'
fodl_file = 'Gap4KHz_results.csv'
fig_loc = '/home/ravinderjit/Documents/Figures/AQ/'

#%% Load Data
data_gap4k =  pd.read_csv(os.path.join(data_loc,fodl_file))

#%% Get Gap4Khz data

Subjects_gap4k= data_gap4k['subj'].to_numpy()
age_gap4k = data_gap4k['age']

Subjects_gap4k, ind = np.unique(Subjects_gap4k,return_index=True)
age_gap4k = age_gap4k[ind].to_numpy()

acc_gap4k = np.zeros([8,len(Subjects_gap4k)])

for s in range(len(Subjects_gap4k)):
    acc_gap4k[:,s] = data_gap4k['score'][data_gap4k['subj'] == Subjects_gap4k[s]]

gap = data_gap4k['gap'][0:8]

#%% Load AQ prolific data
    
AQ = sio.loadmat(data_loc + 'AQscores_Prolific.mat',squeeze_me=True)

aq_subj = AQ['Subjects']
aq_scores = AQ['Scores']
aq_ages = AQ['age']

subjs, ind1,ind2 = np.intersect1d(Subjects_gap4k,aq_subj,return_indices=True)

subjs_age = age_gap4k[ind1]
subjs_acc = acc_gap4k[:,ind1]

aq_scores = aq_scores[:,ind2]
aq_ages = aq_ages[ind2]

#%% Compare quartiles and median

full_score = aq_scores.sum(axis=0)

low_score = np.percentile(full_score,25)
med_score = np.percentile(full_score,50)
high_score = np.percentile(full_score,75)

low_mask = full_score < low_score
med_mask = (full_score >= low_score) & (full_score <= high_score)
high_mask = full_score > high_score

acc_low = np.mean(subjs_acc[:,low_mask],axis=1)
acc_med = np.mean(subjs_acc[:,med_mask],axis=1)
acc_high = np.mean(subjs_acc[:,high_mask],axis=1)

sem_low = np.std(subjs_acc[:,low_mask],axis=1) / np.sqrt(np.sum(low_mask))
sem_med = np.std(subjs_acc[:,med_mask],axis=1) / np.sqrt(np.sum(med_mask))
sem_high = np.std(subjs_acc[:,high_mask],axis=1) / np.sqrt(np.sum(high_mask))

plt.figure()
plt.errorbar(gap, acc_low, sem_low,label='Bottom Quartile')
plt.errorbar(gap, acc_med, sem_med,label='Middle 50')
plt.errorbar(gap, acc_high, sem_high,label='Top Quartile')
plt.ylabel('Accuracy')
plt.xlabel('gap')
plt.legend()

fsz = 15
fig = plt.figure()
fig.set_size_inches(9,9)
plt.errorbar(gap, acc_low, sem_low,label='Bottom Quartile (n=' + str(np.sum(low_mask)) + ')',linewidth=2)
plt.errorbar(gap, acc_med, sem_med,label='Middle 50 (n=' + str(np.sum(med_mask)) + ')', linestyle='dashed')
plt.errorbar(gap, acc_high, sem_high,label='Top Quartile (n=' + str(np.sum(high_mask)) + ')',linewidth=2)
plt.ylabel('Accuracy',fontsize=fsz)
plt.xlabel('Gap (ms)',fontsize=fsz)
plt.xticks([0,15,30],fontsize=fsz)
plt.yticks([50, 75, 100], fontsize=fsz)
plt.legend(fontsize=fsz)
plt.savefig(fig_loc + 'AQvsGap4k.svg',format='svg')

age_mean = [aq_ages[low_mask].mean(), aq_ages[med_mask].mean(), aq_ages[high_mask].mean()]
age_sem = [aq_ages[low_mask].std() / np.sqrt(np.sum(low_mask)), 
           aq_ages[med_mask].std() / np.sqrt(np.sum(med_mask)),
           aq_ages[high_mask].std() / np.sqrt(np.sum(high_mask))]
           
fig = plt.figure()
fsz = 10
fig.set_size_inches(4,4)
plt.bar([0, 0.3, 0.6], age_mean , yerr = age_sem, width = 0.2, color=['Tab:blue', 'Tab:orange', 'Tab:green'])
plt.xticks([0, 0.3, 0.6],labels=['Bottom Quartile', 'Middle 50', 'Top Quartile'],fontsize=fsz)
plt.ylabel('Age',fontsize=fsz)
plt.yticks([25, 30, 35],fontsize=fsz)
plt.ylim([22, 38])
plt.savefig(fig_loc + 'AQvsGap4k_age.svg',format='svg')


