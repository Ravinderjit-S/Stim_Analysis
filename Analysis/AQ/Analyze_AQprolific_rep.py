#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 12:04:14 2022

@author: ravinderjit
"""

import pandas as pd
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt


#%% Load data

data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'


AQ = sio.loadmat(data_loc + 'AQscores_Prolific.mat',squeeze_me=True)
AQ_rep = sio.loadmat(data_loc + 'AQscores_rep_Prolific.mat', squeeze_me=True)

aq_subj = AQ['Subjects']
aq_scores = AQ['Scores']

aqrep_subj = AQ_rep['Subjects_rep']
aqrep_scores = AQ_rep['Scores_rep']

sub_inds = []
for sub in aqrep_subj:
    sub_inds.append(list(aq_subj).index(sub))


aq_scores = aq_scores[:,sub_inds]

aq_tot = aq_scores.sum(axis=0)
aq_tot_rep = aqrep_scores.sum(axis=0)






