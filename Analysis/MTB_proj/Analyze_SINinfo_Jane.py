#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 19:17:46 2022

@author: ravinderjit
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os


def getReversals(x):
    nReversals = 0
    downList = []
    upList = []
    revList = []
    for k in range(2,x.size):
        if( (x[k-1] > x[k]) & (x[k-1] > x[k-2]) ):
            nReversals += 1
            revList.append(k-1)
            downList.append(k-1)
            
        if( (x[k-1] < x[k]) & (x[k-1] < x[k-2]) ):
            nReversals += 1
            revList.append(k-1)
            upList.append(k-1)
            
    return upList, downList, revList

def getThresh(track):
    [up, down, revs] = getReversals(track)
    threshold = np.mean(track[up[-4:]])*0.25 + 0.75 * np.mean(track[down[-4:]])
    return threshold
                
            
Subjects = [ 'S069', 'S072','S078','S088', 'S104', 'S105', 'S207','S211',
            'S259', 'S260', 'S268', 'S269', 'S270','S271', 'S272', 'S273',
            'S274', 'S277','S279', 'S280', 'S282', 'S284', 'S285', 'S288',
            'S290' ,'S281','S291', 'S303', 'S305', 'S308', 'S309', 'S310',
            'S312', 'S339', 'S340', 'S341', 'S344']#, 'S345']




age = [49, 55, 47, 52, 51, 61, 25, 28, 20, 33, 19, 19, 21, 21, 20, 18,
       19, 20, 20, 20, 19, 26, 19, 30, 21, 21, 66, 28, 27, 59, 33, 70,
       37, 71, 39, 35, 60]#, 61]


data_loc_jane = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/SIN_Info_JANE/'
sub_thresh= []
for sub in Subjects:
    path_sub = data_loc_jane + sub
    files = os.listdir(path_sub)
    
    thresholds = []
    for f in files:
        fdata = sio.loadmat(os.path.join(path_sub,f),squeeze_me=True)
        track = fdata['presentedSNRs']
        thresholds.append(getThresh(track))
    
    thresholds = np.sort(thresholds)[:2]
    sub_thresh.append(np.mean(thresholds[-2:]))

sub_thresh = np.array(sub_thresh)


#%% Box Plot
fig = plt.figure()
fig.set_size_inches(7,8)
plt.rcParams.update({'font.size': 15})
whisker =plt.boxplot(sub_thresh)
#whisker['medians'][0].linewidth = 4
plt.xticks([])
plt.yticks([-45,-35, -25, ])
plt.ylabel('TMR (dB)')

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/'
plt.savefig(os.path.join(fig_loc,'MST_box.svg'),format='svg')

#%% Young vs Old

young = np.array(age) < 35
old = np.array(age) >=35

youngThresh = sub_thresh[young]
oldThresh = sub_thresh[old]

fig = plt.figure()
fig.set_size_inches(7,8)
plt.rcParams.update({'font.size': 15})
whisker =plt.boxplot([youngThresh, oldThresh],labels=['Young','Old'])
#whisker['medians'][0].linewidth = 4
#plt.xticks(['Young', 'Old'])
plt.yticks([-45,-35, -25])
plt.ylabel('CMR (dB)')


#%% Save data

sio.savemat(data_loc_jane + 'SINinfo_Jane.mat',{'thresholds':sub_thresh, 'Subjects':Subjects})



