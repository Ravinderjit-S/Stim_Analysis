#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:32:35 2022

@author: ravinderjit
"""


import pandas as pd
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'
fig_loc = '/home/ravinderjit/Documents/Figures/AQ/'
mrt_file = 'mrt_hari_results.csv'

#%% Get mrt data
data_mrt =  pd.read_csv(os.path.join(data_loc,mrt_file))

Subjects_mrt= data_mrt['subj'].to_numpy()

Subjects_mrt, ind = np.unique(Subjects_mrt,return_index=True)

acc_mrt = np.zeros([4,len(Subjects_mrt)])

for s in range(len(Subjects_mrt)):
    acc_mrt[:,s] = data_mrt['score'][data_mrt['subj'] == Subjects_mrt[s]]
    
    

#%% Get AQ data
    
AQ = sio.loadmat(data_loc + 'AQscores_Prolific.mat',squeeze_me=True)

aq_subj = AQ['Subjects']
aq_scores = AQ['Scores']
aq_ages = AQ['age']

subjs, ind1,ind2 = np.intersect1d(Subjects_mrt,aq_subj,return_indices=True)

subjs_acc = acc_mrt[:,ind1]

aq_scores = aq_scores[:,ind2]
aq_ages = aq_ages[ind2]

#%% Look at AQ vs MRT

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

snrs = [10,5,0,-5]

fsz = 15
fig = plt.figure()
fig.set_size_inches(9,9)
plt.errorbar(snrs, acc_low, sem_low,label='Bottom Quartile (n=' + str(np.sum(low_mask)) + ', AQ < ' + str(int(low_score)) + ')',linewidth=2)
plt.errorbar(snrs, acc_med, sem_med,label='Middle 50 (n=' + str(np.sum(med_mask)) + ')', linestyle='dashed')
plt.errorbar(snrs, acc_high, sem_high,label='Top Quartile (n=' + str(np.sum(high_mask)) + ', AQ >' + str(int(high_score)) + ')',linewidth=2)
plt.ylabel('Accuracy',fontsize=fsz)
plt.xlabel('SNR',fontsize=fsz)
plt.xticks([-5,0,5,10],fontsize=fsz)
plt.yticks([50, 75, 100],fontsize=fsz)
plt.legend(fontsize=fsz)
plt.savefig(fig_loc + 'AQvsMRT.svg',format='svg')

fig = plt.figure()
fsz = 15
fig.set_size_inches(9,9)
plt.bar([0, 0.3, 0.6], [acc_low[2], acc_med[2], acc_high[2]] , 
        yerr = [sem_low[2], sem_med[2], sem_high[2]], width = 0.2, 
        edgecolor=['Tab:blue', 'Tab:orange', 'Tab:green'], linewidth= 5 ,color='k',alpha=0.5)
plt.xticks([0, 0.3, 0.6],labels=['Bottom Quartile', 'Middle 50', 'Top Quartile'],fontsize=fsz)
plt.ylabel('Accuracy at 0 dB SNR',fontsize=fsz)
plt.yticks([70, 85, 100],fontsize=fsz)
plt.ylim([70, 100])
plt.savefig(fig_loc + 'AQvsMRT_0db.svg',format='svg')

ttest_ind(subjs_acc[:,low_mask],subjs_acc[:,high_mask],axis=1)

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
plt.savefig(fig_loc + 'AQvsMRT_age.svg',format='svg')



