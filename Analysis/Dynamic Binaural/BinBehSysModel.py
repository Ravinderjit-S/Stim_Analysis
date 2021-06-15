#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:13:57 2021

@author: ravinderjit
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


beh_dataPath = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/'
beh_IACsq = loadmat(os.path.abspath(beh_dataPath+'IACsquareTone_Processed.mat'))

Window_t = beh_IACsq['WindowSizes'][0]
f_beh = Window_t[1:]**-1
SNRs_beh = beh_IACsq['AcrossSubjectsSNR']
AcrossSubjectSEM = beh_IACsq['AcrossSubjectsSEM']

plt.figure()
plt.errorbar(Window_t,SNRs_beh,AcrossSubjectSEM)


def assymetricGaussFit(x,A,tau):
    Out = A*(1 - np.exp(-x/tau))
    return Out


fs = 1000 #arbitrary ... just can't be too small

stDevNeg = 0.1
stDevPos = 0.05
tneg = np.arange(-stDevNeg*4,0,1/fs)
tpos = np.arange(0,stDevPos*4+.01,1/fs)

simpGaussNeg = np.exp(-0.5*(tneg/stDevNeg)**2)
simpGaussPos = np.exp(-0.5*(tpos/stDevPos)**2)

simpGauss = np.concatenate((simpGaussNeg,simpGaussPos))
simpGauss = simpGauss / np.sum(simpGauss)
t = np.concatenate((tneg,tpos))

IAC_0block = np.zeros([int(round(fs*0.4))])
wind_j = Window_t[9]
IAC_1block = np.ones([int(round(fs*wind_j))])

IACnoise = np.concatenate((IAC_0block,IAC_1block,IAC_0block))

tn = np.arange(0,IACnoise.size/fs,1/fs)
plt.figure()
plt.plot(tn, IACnoise)

plt.figure()
plt.plot(t,simpGauss)

xcorr_integration = np.correlate(IACnoise,simpGauss,mode='full')
t_xcorr = np.arange(-(simpGauss.size-1)/fs,(xcorr_integration.size-simpGauss.size+1)/fs,1/fs)

t_overlapStart = np.where(t_xcorr+simpGauss.size/fs > (IACnoise.size/2+ 0.01*fs)/fs)[0][0]
t_overlapEnd = np.where(t_xcorr-simpGauss.size/fs < (IACnoise.size/2 - 0.01*fs)/fs)[0][-1]



maxIAC = np.max(xcorr_integration[t_overlapStart:t_overlapEnd])

plt.figure()
plt.plot(t_xcorr, xcorr_integration)
plt.plot(tn,IACnoise+0.1)



# popt, pcov = sp.optimize.curve_fit(expFit, Window_t,SNRs_beh[:,0],bounds=([0,-np.inf],[np.inf,np.inf]),p0=[10,0.1])
# A = popt[0]
# tau = popt[1]
# fit_t = np.arange(Window_t[0],Window_t[-1],1/fs)
# exp_fit = expFit(fit_t,A,tau)


Tnu = 10**(-beh_IACsq['AcrossSubjectsSNR'][0]/10)
Tno = 10**(-beh_IACsq['AcrossSubjectsSNR'][9]/10)

rho = np.arange(0,1.01,.01)
eq = -10*np.log10(rho + (1-rho)*(Tnu/Tno))

plt.figure()
plt.plot(rho,eq)

plt.figure()
plt.errorbar(f_beh/2,SNRs_beh[1:,0],AcrossSubjectSEM[1:,0])

