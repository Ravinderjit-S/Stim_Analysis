#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/12/2022

@author: ravinderjit
"""
import pandas as pd
import numpy as np
import os
import scipy.io as sio

data_loc = '/home/ravinderjit/Documents/Data/AQ_prolific/'
data_file = 'AQ_Prolific_Feb1122.csv'

data = pd.read_csv(os.path.join(data_loc,data_file))

answers = ['Definitely Agree', 'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree']


Subjects = data['1a'][2:].to_numpy()

survey = np.zeros((50,len(Subjects)))
for q in range(50):
    question = 'Q' + str( int(np.floor(q/10)+2)) #2-6
    trial = str(int(np.mod(q,10) + 1)) #1-10
    index = question + '_' + trial
    for sub in range(len(Subjects)):
        survey[q,sub] = answers.index(data[index][sub+2])
        
   
SS = np.array([1,11,13,15,22,36,44,45,47,48],ndmin=2) - 1 #Social Skills questions
AS = np.array([2,4,10,16,25,32,34,37,43,46],ndmin=2) - 1 #Attention Switching questions
AD = np.array([5,6,9,12,19,23,28,29,30,49],ndmin=2) - 1 #Attention to Detail questions
Comm = np.array([7,17,18,26,27,31,33,35,38,39],ndmin=2) -1 #Communication questions
Imag = np.array([3,8,14,20,21,24,40,41,42,50],ndmin=2) - 1 #Imagination questions

Qtypes = np.concatenate((SS,AS,AD,Comm,Imag),axis=0).T


Agree = np.array([1,2,4,5,6,7,9,12,13,16,18,19,20,21,22,23,26,33,35,39,41,42,43,45,46]) #1 point if agree
Disagree = np.delete(np.arange(50)+1,Agree-1) #1 point if disagree

point_q = np.zeros((50,len(Subjects)))

point_q[Agree-1,:] = (survey[Agree-1,:] == 0) | (survey[Agree-1,:] == 1) 
point_q[Disagree-1,:] = (survey[Disagree-1,:] ==2) | (survey[Disagree-1,:] == 3)

Scores = np.zeros([5,len(Subjects)])
for qtype in range(Qtypes.shape[1]):
    Scores[qtype,:] = np.sum(point_q[Qtypes[:,qtype]],axis=0)


sio.savemat(data_loc + 'AQscores_Prolific.mat',{'Scores':Scores, 'Subjects':Subjects})

