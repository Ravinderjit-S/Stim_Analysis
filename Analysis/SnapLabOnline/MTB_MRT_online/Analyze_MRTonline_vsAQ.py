#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 19:03:22 2022

@author: ravinderjit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
import pandas as pd

def SortSubjects(Subjects,Subjects2):
    #Find indices of Subjects in Subjects 2
    index_sub = []
    del_inds = []
    for s in range(len(Subjects2)):
        if Subjects2[s] in Subjects:
            index_sub.append(Subjects.index(Subjects2[s]))
        else:
            del_inds.append(s)
    
    return index_sub, del_inds

#%% load data

data_loc = '/media/ravinderjit/Data_Drive/Data/AQ/'

aq_score1 = loadmat(data_loc + 'AQscores_Prolific.mat',squeeze_me=True)
aq_score2 = loadmat(data_loc + 'AQscores_rep_Prolific.mat',squeeze_me=True)
aq_autScrn = loadmat(data_loc + 'AQscores_AutScrn_Prolific.mat',squeeze_me=True)

mrt_data = loadmat(data_loc + 'MRT_AQ.mat',squeeze_me=True)

#%% Put all data into a dataframe

subjects = list(mrt_data['Subjects'])
aut_ind = mrt_data['aut_ind']

asd = np.zeros(len(subjects))
asd[aut_ind] = 1

data = pd.DataFrame(data={'subjects': subjects, 'mrt_70': mrt_data['thresholds'],
                          'lapse': mrt_data['lapse'],'gender': mrt_data['gender'],
                          'age': mrt_data['age'], 'race': mrt_data['race'],
                          'neuro':mrt_data['neuro'],
                          'ASD': asd})



#Get aq scores
aq_full = np.empty(len(subjects))
aq_full[:] = np.nan

#first run.
aq_score1_subjects = aq_score1['Subjects'].tolist()
index_sub_aq, del_inds_aq = SortSubjects(subjects,aq_score1_subjects)
aq_scores1_full = aq_score1['Scores'].sum(axis=0)
aq_scores1_full = np.delete(aq_scores1_full,del_inds_aq)
aq_full[index_sub_aq] = aq_scores1_full

#aut scrn
aq_autScrn_subjects = aq_autScrn['Subjects_aut'].tolist()
index_sub_aut, del_inds_aut = SortSubjects(subjects,aq_autScrn_subjects)
aq_scoresAut_full = aq_autScrn['Scores_aut'].sum(axis=0)
aq_scoresAut_full = np.delete(aq_scoresAut_full,del_inds_aut)
aq_full[index_sub_aut] = aq_scoresAut_full


data['aq'] = aq_full

#x = x[~numpy.isnan(x)]

#%% Plot some stuff

plt.figure()
plt.scatter(data['age'][asd==0],data['mrt_70'][asd==0],label='NeuroTyp')
plt.scatter(data['age'][asd==1],data['mrt_70'][asd==1],label='ASD')
plt.legend()
plt.xlabel('Age')
plt.ylabel('MRT 70% Threshold')

plt.figure()
plt.scatter(data['age'][aq_full<=24],data['mrt_70'][aq_full<=24],label='Below Meidan AQ')
plt.scatter(data['age'][aq_full>31],data['mrt_70'][aq_full>31],label='Above Median AQ')
plt.legend()
plt.xlabel('Age')
plt.ylabel('MRT 70% Threshold')

plt.figure()
plt.scatter(data['aq'][asd==0],data['mrt_70'][asd==0],label='NeuroTyp')
plt.scatter(data['aq'][asd==1],data['mrt_70'][asd==1],label='ASD')
plt.legend()
plt.xlabel('AQ')
plt.ylabel('MRT 70% Threshold')

plt.figure()
plt.scatter(data['age'][asd==0],data['lapse'][asd==0],label='NeuroTyp')
plt.scatter(data['age'][asd==1],data['lapse'][asd==1],label='ASD')
plt.legend()
plt.xlabel('Age')
plt.ylabel('lapse')

#%% AQ score dist

plt.figure()
plt.boxplot([data['aq'][asd==0], data['aq'][asd==1 & ~np.isnan(data['aq']).to_numpy()]], labels= ['No Aut', 'Aut'])






