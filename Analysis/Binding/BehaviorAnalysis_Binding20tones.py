#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 13:13:36 2021

@author: ravinderjit
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from matplotlib import rcParams

data_loc = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/MTB_beh/'
fig_loc = '/media/ravinderjit/Data_Drive/Data/Figures/MTB/'

# Subjects = ['S069', 'S072', 'S078', 'S088', 'S104', 'S105', 'S207',
#             'S211', 'S246', 'S259', 'S260', 'S268', 'S269', 'S270',
#             'S271', 'S272', 'S273', 'S274', 'S277', 'S279', 'S280',
#             'S281', 'S282', 'S284', 'S285', 'S288', 'S290', 'S291', 
#             'S303', 'S305', 'S308', 'S309', 'S310', 'S312', 'S337', 
#             'S339', 'S340', 'S341', 'S342', 'S344', 'S345', 'S347',
#             'S352', 'S355', 'S358']

Subjects = [ 'S069', 'S072','S078','S088', 'S104', 'S105', 'S207','S211',
            'S259', 'S260', 'S268', 'S269', 'S270','S271', 'S272', 'S273',
            'S274', 'S277','S279', 'S280', 'S282', 'S284', 'S285', 'S288',
            'S290' ,'S281','S291', 'S303', 'S305', 'S308', 'S309', 'S310',
            'S312', 'S339', 'S340', 'S341', 'S344', 'S345', 'S337', 'S352',
            'S355', 'S358']


age = [49, 55, 47, 52, 51, 61, 25, 28, 20, 33, 19, 19, 21, 21, 20, 18,
       19, 20, 20, 20, 19, 26, 19, 30, 21, 21, 66, 28, 27, 59, 33, 70,
       37, 71, 39, 35, 60, 61, 66, 35, 49, 56]
            

acc = np.zeros((7,len(Subjects)))
for sub in range(len(Subjects)):
    subject = Subjects[sub]
    dat_subpath = os.path.join(data_loc,subject + '_BindingBehavior20tones.mat')
    data = sio.loadmat(dat_subpath)
    
    CorrSet = data['CorrSet'][0]
    Corr_inds = data['Corr_inds'][0]
    correctList = data['correctList'][0]
    respList = data['respList'][0]
    
    ntrials = np.sum(CorrSet==1) #20
    

    for cond in np.unique(CorrSet):
        mask = CorrSet==cond 
        num_right = np.sum(correctList[mask] == respList[mask])
        acc[cond-1,sub] = num_right/ntrials
        
        
  
# 0 = 19:20
# 1 = 17-20
# 2 = 15-20
# 3 = 13:20
# 4 = 1, 8, 14, 20
# 5 = 1, 4, 8, 12, 16, 20
# 6 = 1, 4, 6, 9, 12, 15, 17, 20


mean_acc = acc.mean(axis=1)    
se_acc = acc.std(axis=1) / np.sqrt(acc.shape[1])    
        
fig = plt.figure()
fig.set_size_inches(15,8)
plt.bar(0,mean_acc[0], width = 0.3, yerr= se_acc[0], color='k')
plt.bar(np.array([0,0.7, 1.4]) + 0.4,mean_acc[1:4],  width = 0.3, yerr= se_acc[1:4],color='k', label ='Consecutive')
plt.bar(np.array([0.3,1,1.7]) + 0.4,mean_acc[4:7],  width = 0.3, yerr = se_acc[4:7],color='grey', label='Interrupted')
plt.axhline(y=0.33,xmin=0,xmax=2,linestyle='dashed',color='tab:orange',linewidth=3,label='Chance')
plt.ylim([0,1.1])
plt.xticks(np.array([-0.4,0,0.3,0.7,1,1.4,1.7]) + 0.4,labels=['1.5 ERBs', '1.5 ERBs', '9.5 ERBs', '1.5 ERBs','5.7 ERBs', '1.5 ERBs', '4 ERBs'],fontsize=16)
plt.yticks([0,0.5,1])
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=16)
plt.savefig(fig_loc + 'BindingAccuracy.svg',format='svg')

plt.figure()
plt.plot([1,2,3,4],acc[:4,:])

plt.figure()
plt.plot([1,2,3],acc[4:,:])

plt.figure()
plt.plot(np.arange(7)+1,acc)
plt.plot(np.arange(7)+1,acc.mean(axis=1),color='k',linewidth=3)

plt.figure()
young = np.array(age) <35
old = np.array(age)  >=35
plt.plot(np.arange(7)+1,acc[:,young],color='b')
plt.plot(np.arange(7)+1,acc[:,old],color='r')
plt.plot(np.arange(7)+1,acc.mean(axis=1),color='k',linewidth=3)

#%% Age Plot

young = np.array(age) <32
old = np.array(age)  >=40

mean_acc_y = acc[:,young].mean(axis=1)    
se_acc_y = acc[:,young].std(axis=1) / np.sqrt(acc[:,young].shape[1])    

mean_acc_o = acc[:,old].mean(axis=1)    
se_acc_o = acc[:,old].std(axis=1) / np.sqrt(acc[:,old].shape[1])    
        
fig = plt.figure()
fig.set_size_inches(15,8)
plt.bar(0,mean_acc_y[0], width = 0.3, yerr= se_acc[0], color='k')


plt.bar(np.array([0,0.7, 1.4]) + 0.4,mean_acc_y[1:4],  width = 0.15, yerr= se_acc_y[1:4],color='k', label ='Consecutive')
plt.bar(np.array([0,0.7, 1.4]) + 0.55,mean_acc_o[1:4],  width = 0.15, yerr= se_acc_o[1:4],color='red', label ='Consecutive')


plt.bar(np.array([0.3,1,1.7]) + 0.4,mean_acc_y[4:7],  width = 0.15, yerr = se_acc_y[4:7],color='grey', label='Interrupted')
plt.bar(np.array([0.3,1,1.7]) + 0.55,mean_acc_o[4:7],  width = 0.15, yerr = se_acc_o[4:7],color='red', label='Interrupted')

plt.axhline(y=0.33,xmin=0,xmax=2,linestyle='dashed',color='tab:orange',linewidth=3,label='Chance')
plt.ylim([0,1.1])
plt.xticks(np.array([-0.4,0,0.3,0.7,1,1.4,1.7]) + 0.4,labels=['1.5 ERBs', '1.5 ERBs', '9.5 ERBs', '1.5 ERBs','5.7 ERBs', '1.5 ERBs', '4 ERBs'],fontsize=16)
plt.yticks([0,0.5,1])
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=16)
#plt.savefig(fig_loc + 'BindingAccuracy.svg',format='svg')




#%% Save data

sio.savemat(data_loc + 'BindingBeh.mat',{'acc':acc, 'Subjects':Subjects})

