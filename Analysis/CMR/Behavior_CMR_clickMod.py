#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:34:12 2021

@author: ravinderjit
"""

import numpy as np
import scipy.io as sio
from numpy import matlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import psignifit as ps


def GaussCDF(x,sigma,mu,lapseRate):
    return 0.5 *(1+erf((x-mu)/(sigma * np.sqrt(2)))) * (1-1/3-lapseRate)  + 1/3  # 1/3 for 3 AFC

data_loc = '/media/ravinderjit/Data_Drive/Data/BehaviorData/CMR/'

Subjects = ['S211','SVM']
subject = Subjects[1]
data = sio.loadmat(data_loc + subject + '_CMRrandModClicky.mat',squeeze_me=True)
ntrials = 20

SNRs_0 = np.concatenate((np.array([-10]),np.arange(-25,-55,-5)))
SNRs_1 = np.concatenate((np.array([-15]),np.arange(-35,-65,-5)))

snrList_unique = np.concatenate((SNRs_0,SNRs_1))
snrList = matlib.repmat(snrList_unique,1,20).squeeze()

cohList = data['cohList']
correctList = data['correctList']
respList = data['respList']

acc_0 = np.zeros([SNRs_0.size])
acc_1 = np.zeros([SNRs_1.size])

for ii in range(acc_0.size):
    mask = (cohList == 0) & (snrList == SNRs_0[ii])
    acc_0[ii] = np.sum(correctList[mask] == respList[mask]) / ntrials
    
for ii in range(acc_1.size):
    mask = (cohList ==1) & (snrList == SNRs_1[ii])
    acc_1[ii] = np.sum(correctList[mask] == respList[mask]) / ntrials
    
    
param_0, pcov_0 = curve_fit(GaussCDF,SNRs_0,acc_0,[2,-40,1-acc_0[0]])
x_0 = np.arange(SNRs_0[-1],SNRs_0[0],.1)
psy_curve_0 = GaussCDF(x_0,param_0[0],param_0[1],param_0[2])

param_1, pcov_1 = curve_fit(GaussCDF,SNRs_1,acc_1,[2,-50,1-acc_1[0]])
x_1 = np.arange(SNRs_1[-1],SNRs_1[0],.1)
psy_curve_1 = GaussCDF(x_1,param_1[0],param_1[1],param_1[2])
    
plt.figure()
plt.plot(SNRs_0, acc_0)     
plt.plot(SNRs_1, acc_1)
plt.plot(x_0,psy_curve_0,'--')   
plt.plot(x_1,psy_curve_1,'--')


#%% Do fit with psignifit

data_0ps = np.concatenate((SNRs_0[:,np.newaxis], acc_0[:,np.newaxis] * ntrials, 20*np.ones([7,1])),axis=1)
data_1ps = np.concatenate((SNRs_1[:,np.newaxis], acc_1[:,np.newaxis] * ntrials, 20*np.ones([7,1])),axis=1)

options = dict({
    'sigmoidName': 'norm',
    'expType': '3AFC'
    })

result_0ps = ps.psignifit(data_0ps, options)
result_1ps = ps.psignifit(data_1ps, options)

plt.figure()
ps.psigniplot.plotPsych(result_0ps)
ps.psigniplot.plotPsych(result_1ps)
plt.plot(x_0,psy_curve_0,'--')   
plt.plot(x_1,psy_curve_1,'--')




