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


data_loc = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/CMR_beh/'

Subjects = [ 'S072', 'S078', 'S088', 'S207', 'S211','S246', 'S259', 'S260',
            'S268', 'S269', 'S270', 'S271', 'S272', 'S273', 'S274', 'S277',
            'S279', 'S280', 'S281', 'S282', 'S284', 'S285', 'S288', 'S290',
            'S291', 'S303', 'S305', 'S308', 'S309', 'S310'] 


CMR = np.zeros((len(Subjects)))

for sub in range(len(Subjects)):

    subject = Subjects[sub]
    data = sio.loadmat(data_loc + subject + '_CMRrandModClicky.mat',squeeze_me=True)
    ntrials = 20
    
    #SNRs_0 = np.concatenate((np.array([-10]),np.arange(-25,-55,-5)))
    #SNRs_1 = np.concatenate((np.array([-20]),np.arange(-37,-61,-3)))
    
    SNRs_0 = data['SNR_0']
    SNRs_1 = data['SNR_1']
    
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

    #%% Do fit with psignifit
    
    data_0ps = np.concatenate((SNRs_0[:,np.newaxis], acc_0[:,np.newaxis] * ntrials, 20*np.ones([7,1])),axis=1)
    data_1ps = np.concatenate((SNRs_1[:,np.newaxis], acc_1[:,np.newaxis] * ntrials, 20*np.ones([acc_1.size,1])),axis=1)
    
    options = dict({
        'sigmoidName': 'norm',
        'expType': '3AFC'
        })
    
    result_0ps = ps.psignifit(data_0ps, options)
    result_1ps = ps.psignifit(data_1ps, options)
    
    plt.figure()
    ps.psigniplot.plotPsych(result_0ps)
    ps.psigniplot.plotPsych(result_1ps, dataColor='tab:orange',lineColor='tab:orange')
    plt.title(Subjects[sub])
    
    percentCorr = 0.75
    CMR[sub] = ps.getThreshold(result_0ps,percentCorr)[0] - ps.getThreshold(result_1ps,percentCorr)[0]
    
sio.savemat(data_loc + 'CMRclickMod.mat',{'CMR':CMR, 'Subjects':Subjects})

