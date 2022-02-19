#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 21:51:21 2022

@author: ravinderjit
"""


import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

fig_loc = '/home/ravinderjit/Documents/Figures/AQ/'

data_loc_sp = '/home/ravinderjit/Documents/Data/AQ_prolific/From_Agu/'
data_loc_aq = '/home/ravinderjit/Documents/Data/AQ_prolific/'

#%% Load Data

ITD = sio.loadmat(data_loc_sp + 'data_ITD.mat',squeeze_me=True)
ILD = sio.loadmat(data_loc_sp + 'data_ILD.mat',squeeze_me=True)

ITD_subs = ITD['subjNames'].tolist()
ILD_subs = ILD['subjNames'].tolist()

if ITD_subs[179] == '5ed40dab8680192984d2aa98-2ndTry':
    ITD_subs[179] = '5ed40dab8680192984d2aa98'  
    
#ITD_subs has white space in subj names
ITD_subs = [x.strip(' ') for x in ITD_subs]  

itd_vals = np.array([2, 4, 8, 16, 32, 64, 128])
ild_vals = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])

#%% Get ITD and ILD data

data_itd = ITD['data'][:,:,1] / ITD['data'][:,:,2]  
data_ild = ILD['data'][:,:,1] / ILD['data'][:,:,2]  


#%% Load AQ prolific data
    
AQ = sio.loadmat(data_loc_aq + 'AQscores_Prolific.mat',squeeze_me=True)

aq_subj = AQ['Subjects']
aq_scores = AQ['Scores']
aq_ages = AQ['age']

subjs_ITD, ind1,ind2 = np.intersect1d(ITD_subs,aq_subj,return_indices=True)

itd_acc = data_itd[ind1,:]
aq_scores_ITD = aq_scores[:,ind2]


subjs, ind1,ind2 = np.intersect1d(ILD_subs,aq_subj,return_indices=True)

ild_acc = data_ild[ind1,:]
aq_scores_ILD = aq_scores[:,ind2]
aq_ages = aq_ages[ind2]

#%% AQ vs ITD

full_score = aq_scores_ITD.sum(axis=0)

low_score = np.percentile(full_score,25)
med_score = np.percentile(full_score,50)
high_score = np.percentile(full_score,75)

low_mask = full_score < low_score
med_mask = (full_score >= low_score) & (full_score <= high_score)
high_mask = full_score > high_score

acc_low = np.mean(itd_acc[low_mask,:],axis=0)
acc_med = np.mean(itd_acc[med_mask,:],axis=0)
acc_high = np.mean(itd_acc[high_mask,:],axis=0)

sem_low = np.std(itd_acc[low_mask,:],axis=0) / np.sqrt(np.sum(low_mask))
sem_med = np.std(itd_acc[med_mask,:],axis=0) / np.sqrt(np.sum(med_mask))
sem_high = np.std(itd_acc[high_mask,:],axis=0) / np.sqrt(np.sum(high_mask))


fsz = 15
fig = plt.figure()
fig.set_size_inches(9,9)
plt.errorbar(itd_vals, acc_low, sem_low,label='Bottom Quartile (n=' + str(np.sum(low_mask)) + ')',linewidth=2)
plt.errorbar(itd_vals, acc_med, sem_med,label='Middle 50 (n=' + str(np.sum(med_mask)) + ')', linestyle='dashed')
plt.errorbar(itd_vals, acc_high, sem_high,label='Top Quartile (n=' + str(np.sum(high_mask)) + ')',linewidth=2)
plt.ylabel('Accuracy',fontsize=fsz)
plt.xlabel('ITD (\u03BCs)',fontsize=fsz)
plt.xticks([0,50,100],fontsize=fsz)
plt.yticks([.50, .75, 1], labels=[50, 75, 100],fontsize=fsz)
plt.legend(fontsize=fsz)
plt.savefig(fig_loc + 'AQvsITD.svg',format='svg')


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
plt.savefig(fig_loc + 'AQvsITD_age.svg',format='svg')


#%% AQ vs ILD

full_score = aq_scores_ILD.sum(axis=0)

low_score = np.percentile(full_score,25)
med_score = np.percentile(full_score,50)
high_score = np.percentile(full_score,75)

low_mask = full_score < low_score
med_mask = (full_score >= low_score) & (full_score <= high_score)
high_mask = full_score > high_score

acc_low = np.mean(ild_acc[low_mask,:],axis=0)
acc_med = np.mean(ild_acc[med_mask,:],axis=0)
acc_high = np.mean(ild_acc[high_mask,:],axis=0)

sem_low = np.std(ild_acc[low_mask,:],axis=0) / np.sqrt(np.sum(low_mask))
sem_med = np.std(ild_acc[med_mask,:],axis=0) / np.sqrt(np.sum(med_mask))
sem_high = np.std(ild_acc[high_mask,:],axis=0) / np.sqrt(np.sum(high_mask))

fsz = 15
fig = plt.figure()
fig.set_size_inches(9,9)
plt.errorbar(ild_vals, acc_low, sem_low,label='Bottom Quartile (n=' + str(np.sum(low_mask)) + ')',linewidth=2)
plt.errorbar(ild_vals, acc_med, sem_med,label='Middle 50 (n=' + str(np.sum(med_mask)) + ')', linestyle='dashed')
plt.errorbar(ild_vals, acc_high, sem_high,label='Top Quartile (n=' + str(np.sum(high_mask)) + ')',linewidth=2)
plt.ylabel('Accuracy',fontsize=fsz)
plt.xlabel('ILD (dB)',fontsize=fsz)
plt.xticks([0,1,2,3],fontsize=fsz)
plt.yticks([.50, .75, 1], labels=[50, 75, 100],fontsize=fsz)
plt.legend(fontsize=fsz)
plt.savefig(fig_loc + 'AQvsILD.svg',format='svg')

ild_stats = ttest_ind(ild_acc[low_mask,:],ild_acc[high_mask,:],axis=0)

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
plt.savefig(fig_loc + 'AQvsILD_age.svg',format='svg')







