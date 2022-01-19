#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:37:51 2022

@author: ravinderjit
"""


import pandas as pd
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt

data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'
data_file = 'cmr_hari_results.csv'

data = pd.read_csv(os.path.join(data_loc,data_file))

Subjects = data['subj'].to_numpy()
age = data['age'].to_numpy()

Subjects, ind = np.unique(Subjects,return_index=True)
age = age[ind]

acc = np.zeros([12,len(Subjects)])

for s in range(len(Subjects)):
    acc[:,s] = data['score'][data['subj'] == Subjects[s]][6:]

snrs = data['snr'][data['subj'] == Subjects[0]][6:]
conds = data['Condition'][data['subj'] == Subjects[0]][6:]

#%% Load AQ prolific data
# Get subjects who we have CMR and AQ on ...

AQ = sio.loadmat(data_loc + 'AQscores_Prolific.mat')

aq_subj = AQ['Subjects']
aq_scores = AQ['Scores']

subjs, ind1,ind2 = np.intersect1d(Subjects,aq_subj,return_indices=True)

subjs_age = age[ind1]
subjs_acc = acc[:,ind1]

aq_scores = aq_scores[:,ind2]

#%% Compare above and below median

full_score = aq_scores.sum(axis=0)

med = np.median(full_score)

acc_low = np.mean(subjs_acc[:,full_score <= med],axis=1)
acc_high = np.mean(subjs_acc[:,full_score > med],axis=1)

sem_low = np.std(subjs_acc[:,full_score <= med],axis=1) / np.sqrt(np.sum(full_score <= med))
sem_high = np.std(subjs_acc[:,full_score > med],axis=1) / np.sqrt(np.sum(full_score > med))

plt.figure()
plt.errorbar(snrs[0:6],acc_low[0:6],sem_low[0:6])
plt.errorbar(snrs[6:],acc_low[6:],sem_low[0:6])
plt.title('AQ scores below median')


plt.figure()
plt.errorbar(snrs[0:6],acc_high[0:6],sem_high[0:6])
plt.errorbar(snrs[6:],acc_high[6:],sem_high[6:])
plt.title('AQ scores above median')


plt.figure()
plt.errorbar(snrs[0:6],acc_low[0:6],sem_low[0:6],color='tab:blue')
plt.errorbar(snrs[6:],acc_low[6:],sem_low[0:6],color='tab:blue', linestyle = '--')

plt.errorbar(snrs[0:6],acc_high[0:6],sem_high[0:6],color='tab:orange')
plt.errorbar(snrs[6:],acc_high[6:],sem_high[6:],color='tab:orange',linestyle='--')

plt.title('Top vs Bottom half')

#%% Compare bottom and top 25 percentile


low_score = np.percentile(full_score,25)
high_score = np.percentile(full_score,75)

acc_low = np.median(subjs_acc[:,full_score <= low_score],axis=1)
acc_high = np.median(subjs_acc[:,full_score >= high_score],axis=1)

sem_low = np.std(subjs_acc[:,full_score <= low_score],axis=1) / np.sqrt(np.sum(full_score <= low_score))
sem_high = np.std(subjs_acc[:,full_score >= high_score],axis=1) / np.sqrt(np.sum(full_score >= high_score))

plt.figure()
plt.errorbar(snrs[0:6],acc_low[0:6],sem_low[0:6])
plt.errorbar(snrs[6:],acc_low[6:],sem_low[0:6])
plt.title('AQ scores below median')


plt.figure()
plt.errorbar(snrs[0:6],acc_high[0:6],sem_high[0:6])
plt.errorbar(snrs[6:],acc_high[6:],sem_high[6:])
plt.title('AQ scores above median')


plt.figure()
plt.errorbar(snrs[0:6],acc_low[0:6],sem_low[0:6],color='tab:blue')
plt.errorbar(snrs[6:],acc_low[6:],sem_low[0:6],color='tab:blue', linestyle = '--')

plt.errorbar(snrs[0:6],acc_high[0:6],sem_high[0:6],color='tab:orange')
plt.errorbar(snrs[6:],acc_high[6:],sem_high[6:],color='tab:orange',linestyle='--')

plt.title('Top and bottom quartile')

