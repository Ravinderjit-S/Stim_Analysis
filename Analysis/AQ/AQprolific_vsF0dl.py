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

data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'
fodl_file = 'F0DLs_results.csv'

#%% Load Data
data_fodl =  pd.read_csv(os.path.join(data_loc,fodl_file))

#%% \
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
    
AQ = sio.loadmat(data_loc + 'AQscores_Prolific.mat')

aq_subj = AQ['Subjects']
aq_scores = AQ['Scores']

subjs, ind1,ind2 = np.intersect1d(Subjects_fodl,aq_subj,return_indices=True)

subjs_age = age_fodl[ind1]
subjs_acc = acc_fodl[:,ind1]

aq_scores = aq_scores[:,ind2]

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
plt.errorbar(df, acc_low, sem_low,label='Bottom Quartile')
#plt.errorbar(df, acc_med, sem_med,label='Middle 50')
plt.errorbar(df, acc_high, sem_high,label='Top Quartile')
plt.ylabel('Accuracy')
plt.xlabel('dfHz')
plt.legend()




