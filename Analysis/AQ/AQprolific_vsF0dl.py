#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 00:57:25 2022

@author: ravinderjit
"""


import pandas as pd
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

fig_loc = '/home/ravinderjit/Documents/Figures/AQ/'
data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'
fodl_file = 'F0DLs_results.csv'


#%% Load Data
data_fodl =  pd.read_csv(os.path.join(data_loc,fodl_file))

#%% Get F0dl data

Subjects_fodl = data_fodl['subj'].to_numpy()
age_fodl = data_fodl['age']

Subjects_fodl, ind = np.unique(Subjects_fodl,return_index=True)
age_fodl = age_fodl[ind].to_numpy()

acc_fodl = np.zeros([10,len(Subjects_fodl)])

for s in range(len(Subjects_fodl)):
    acc_fodl[:,s] = data_fodl['score'][data_fodl['subj'] == Subjects_fodl[s]]

df = data_fodl['dfHz'][0:10]

#%% Load AQ prolific data
    
AQ = sio.loadmat(data_loc + 'AQscores_Prolific.mat',squeeze_me=True)

aq_subj = AQ['Subjects']
aq_scores = AQ['Scores']
aq_ages = AQ['age']

subjs, ind1,ind2 = np.intersect1d(Subjects_fodl,aq_subj,return_indices=True)

subjs_age = age_fodl[ind1]
subjs_acc = acc_fodl[:,ind1]

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
plt.errorbar(df, acc_low, sem_low,label='Bottom Quartile (n=' + str(np.sum(low_mask)) + ')',linewidth=2)
plt.errorbar(df, acc_med, sem_med,label='Middle 50 (n=' + str(np.sum(med_mask)) + ')', linestyle='dashed' )
plt.errorbar(df, acc_high, sem_high,label='Top Quartile (n=' + str(np.sum(high_mask)) + ')',linewidth=2)
plt.ylabel('Accuracy')
plt.xlabel('dfHz')
plt.xticks(df)
plt.xscale('log')
plt.legend()

fsz = 15
fig = plt.figure()
fig.set_size_inches(9,9)
plt.errorbar(df, acc_low, sem_low,label='Bottom Quartile (n=' + str(np.sum(low_mask)) + ')',linewidth=2)
plt.errorbar(df, acc_med, sem_med,label='Middle 50 (n=' + str(np.sum(med_mask)) + ')', linestyle='dashed')
plt.errorbar(df, acc_high, sem_high,label='Top Quartile (n=' + str(np.sum(high_mask)) + ')',linewidth=2)
plt.ylabel('Accuracy',fontsize=fsz)
plt.xlabel('F0 Difference (%)',fontsize=fsz)
plt.xscale('log')
plt.xticks([0.1,0.5,1,3], labels = ['0.1', '0.5', '1', '3'],fontsize=fsz)
plt.yticks([33, 66, 100],fontsize=fsz)
plt.legend(fontsize=fsz)
plt.savefig(fig_loc + 'AQvsF0dl.svg',format='svg')

plt.figure()
plt.plot(df, subjs_acc[:,low_mask],color='tab:blue')
plt.plot(df, subjs_acc[:,high_mask],color='tab:green')

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

plt.savefig(fig_loc + 'AQvsF0dl_age.svg',format='svg')



