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
import os

data_loc = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/CMR_beh/'

# Subjects = [ 'S072', 'S078', 'S088', 'S207', 'S211','S246', 'S259', 'S260',
#             'S268', 'S269', 'S270', 'S271', 'S272', 'S273', 'S274', 'S277',
#             'S279', 'S280', 'S281', 'S282', 'S284', 'S285', 'S288', 'S290',
#             'S291', 'S303', 'S305', 'S308', 'S309', 'S310'] 

Subjects = [ 'S069', 'S072','S078','S088', 'S104', 'S105', 'S207','S211',
            'S259', 'S260', 'S268', 'S269', 'S270','S271', 'S272', 'S273',
            'S274', 'S277','S279', 'S280', 'S282', 'S284', 'S285', 'S288',
            'S290' ,'S281','S291', 'S303', 'S305', 'S308', 'S309', 'S310',
            'S312', 'S339', 'S340', 'S341', 'S344', 'S345']



age = [49, 55, 47, 52, 51, 61, 25, 28, 20, 33, 19, 19, 21, 21, 20, 18,
       19, 20, 20, 20, 19, 26, 19, 30, 21, 21, 66, 28, 27, 59, 33, 70,
       37, 71, 39, 35, 60, 61]


CMR = np.zeros((len(Subjects)))
lapse = np.zeros((len(Subjects)))

psCurve_0 = []
psCurve_1 = []

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
    
    # plt.figure()
    # ps.psigniplot.plotPsych(result_0ps)
    # ps.psigniplot.plotPsych(result_1ps, dataColor='tab:orange',lineColor='tab:orange')
    # plt.title(Subjects[sub])
    
    percentCorr = 0.70
    CMR[sub] = ps.getThreshold(result_0ps,percentCorr)[0] - ps.getThreshold(result_1ps,percentCorr)[0]
    lapse[sub] = (result_0ps['Fit'][2] + result_1ps['Fit'][2]) / 2
    
    #%% Pull out stuff to look at average 
    x_vals  = np.linspace(-60, 0, num=1000)
    
    fit = result_0ps['Fit']
    data = result_0ps['data']
    options = result_0ps['options']

    fitValues_0 = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x_vals,     fit[0], fit[1]) + fit[3]
    

    fit = result_1ps['Fit']
    data = result_1ps['data']
    options = result_1ps['options']
    
    fitValues_1 = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x_vals,     fit[0], fit[1]) + fit[3]
    
    psCurve_0.append(fitValues_0)
    psCurve_1.append(fitValues_1)
    
    
#%% Plot Average Response

psCurve_0 = np.array(psCurve_0)
psCurve_1 = np.array(psCurve_1)

incoh_mean = psCurve_0.mean(axis=0)
incoh_sem = psCurve_0.std(axis=0) / np.sqrt(psCurve_0.shape[0])

coh_mean = psCurve_1.mean(axis=0)
coh_sem = psCurve_1.std(axis=0) / np.sqrt(psCurve_1.shape[0])

fig = plt.figure()
fig.set_size_inches(8,8)
plt.rcParams.update({'font.size': 15})
plt.plot(x_vals,incoh_mean, color = 'Grey', label='Incoherent',linewidth=2)
plt.fill_between(x_vals, incoh_mean - incoh_sem, incoh_mean + incoh_sem, color='grey',alpha =  0.5)
plt.plot(x_vals, coh_mean, color = 'Black', label = 'Coherent',linewidth=2)
plt.fill_between(x_vals, coh_mean - coh_sem, coh_mean + coh_sem, color='Black',alpha = 0.5)
plt.legend(loc=2)
plt.xlim([-60,-10])
plt.xticks([-60,-40,-20])
plt.yticks([0.33,0.66,1])
plt.ylim([0.3, 1.03])
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy')

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/'
plt.savefig(os.path.join(fig_loc,'CMR_psCurves.svg'),format='svg')

#%% Make Box Plot

fig = plt.figure()
fig.set_size_inches(7,8)
plt.rcParams.update({'font.size': 15})
whisker =plt.boxplot(CMR)
#whisker['medians'][0].linewidth = 4
plt.xticks([])
plt.yticks([4, 10, 16])
plt.ylabel('CMR (dB)')

fig_loc =  '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/'
plt.savefig(os.path.join(fig_loc,'CMR_box.svg'),format='svg')

#%% young vs old

young = np.array(age) < 35
old = np.array(age) >=35

youngCMR = CMR[young]
oldCMR = CMR[old]

fig = plt.figure()
fig.set_size_inches(7,8)
plt.rcParams.update({'font.size': 15})
whisker =plt.boxplot([youngCMR, oldCMR],labels=['Young','Old'])
#whisker['medians'][0].linewidth = 4
#plt.xticks(['Young', 'Old'])
plt.yticks([4, 10, 16])
plt.ylabel('CMR (dB)')


   
#%% Save Data    
sio.savemat(data_loc + 'CMRclickMod.mat',{'CMR':CMR, 'lapse':lapse, 'Subjects':Subjects})











