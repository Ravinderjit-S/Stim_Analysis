#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 12:04:14 2022

@author: ravinderjit

Analyze collected AQ data from Prolific
Have a initial and repeat measure in neurotypicals as well as one measure from
the Autism screener on Prolific - 05/21
"""

import pandas as pd
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt


#%% Load data

data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'
data_loc = '/media/ravinderjit/Data_Drive/Data/AQ/'


AQ = sio.loadmat(data_loc + 'AQscores_Prolific.mat',squeeze_me=True)
AQ_rep = sio.loadmat(data_loc + 'AQscores_rep_Prolific.mat', squeeze_me=True)
AQ_aut = sio.loadmat(data_loc + 'AQscores_AutScrn_Prolific.mat',squeeze_me=True)

#%% Look at general stats

aq_subj = AQ['Subjects']
aq_scores = AQ['Scores']

aqrep_subj = AQ_rep['Subjects_rep']
aqrep_scores = AQ_rep['Scores_rep']

aqaut_subj = AQ_aut['Subjects_aut']
aqaut_scores = AQ_aut['Scores_aut']


#%% For test repeat, filter out people who didn't do twice
sub_inds = []
for sub in aqrep_subj:
    sub_inds.append(list(aq_subj).index(sub))


aq_scores = aq_scores[:,sub_inds]


#%% Plot after filtering out people who didn't do twice

plt.figure()
plt.boxplot([aq_scores.sum(axis=0), aqrep_scores.sum(axis=0), aqaut_scores.sum(axis=0)], labels= ['AQ 1', 'AQ 2', 'AQ aut'])


#%% Look at test repeat

plt.figure()
plt.scatter(aq_scores.sum(axis=0),aqrep_scores.sum(axis=0))

score_diff = aq_scores.sum(axis=0) - aqrep_scores.sum(axis=0)






