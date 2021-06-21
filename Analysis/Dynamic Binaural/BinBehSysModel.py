#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:13:57 2021

@author: ravinderjit
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.io import loadmat
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.signal import freqz


beh_dataPath = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/'
beh_IACsq = loadmat(os.path.abspath(beh_dataPath+'IACsquareTone_Processed.mat'))

Window_t = beh_IACsq['WindowSizes'][0]
f_beh = Window_t[1:]**-1
SNRs_beh = beh_IACsq['AcrossSubjectsSNR']
AcrossSubjectSEM = beh_IACsq['AcrossSubjectsSEM']

plt.figure()
plt.errorbar(Window_t,SNRs_beh,AcrossSubjectSEM)


Tnu = 10**(-beh_IACsq['AcrossSubjectsSNR'][0]/10)
Tno = 10**(-beh_IACsq['AcrossSubjectsSNR'][9]/10) 


def IACcurveFit(winds,stDevNeg,stDevPos):
    global Tnu
    global Tno
    out = np.zeros(winds.size)
    for j in range(0,winds.size):
        wind_j = winds[j]
        fs = 1000 #arbitrary ... just can't be too small
        tneg = np.arange(-stDevNeg*4,0,1/fs)
        tpos = np.arange(0,stDevPos*4+.01,1/fs)
        
        simpGaussNeg = np.exp(-0.5*(tneg/stDevNeg)**2)
        simpGaussPos = np.exp(-0.5*(tpos/stDevPos)**2)
        
        simpGauss = np.concatenate((simpGaussNeg,simpGaussPos))
        simpGauss = simpGauss / np.sum(simpGauss)
        
        IAC_0block = np.zeros([int(np.round(fs*0.4))])
        IAC_1block = np.ones([int(np.round(fs*wind_j))])
        IACnoise = np.concatenate((IAC_0block,IAC_1block,IAC_0block))
        
        xcorr_integration = np.correlate(IACnoise,simpGauss,mode='full')
        t_xcorr = np.arange(-(simpGauss.size-1)/fs,(xcorr_integration.size-simpGauss.size+1)/fs,1/fs)
        
        #where does window overlap with tone
        t_overlapStart = np.where(t_xcorr+simpGauss.size/fs > (IACnoise.size/2+ 0.01*fs)/fs)[0][0]
        t_overlapEnd = np.where(t_xcorr-simpGauss.size/fs < (IACnoise.size/2 - 0.01*fs)/fs)[0][-1]
        
        maxIAC = np.max(xcorr_integration[t_overlapStart:t_overlapEnd])
        
        rho = np.arange(0,1.01,.01)
        eq = -10*np.log10((1-rho) + (rho)*(Tno/Tnu))
        
        if(maxIAC >=1):
            ind_rho = -1
        elif(maxIAC <= 0):
            ind_rho = 0
        else:
            ind_rho = np.where(rho>=maxIAC)[0][0]
        
        out[j] = eq[ind_rho]
        
    return out
    
    


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



rho = np.arange(0,1.01,.01)
#eq = -10*np.log10(rho + (1-rho)*(Tnu/Tno))
eq = -10*np.log10((1-rho) + (rho)*(Tno/Tnu))

plt.figure()
plt.plot(rho,eq)

plt.figure()
plt.errorbar(f_beh/2,SNRs_beh[1:,0],AcrossSubjectSEM[1:,0])


BMLD = SNRs_beh[:,0] - SNRs_beh[0]
popt, pcov = curve_fit(IACcurveFit, Window_t, BMLD, bounds=(0.001,1), p0=[.4,.8]) #doesn't work 

BMLDest = IACcurveFit(Window_t, popt[0], popt[1])
plt.figure()
plt.plot(Window_t,BMLD)
plt.plot(Window_t,BMLDest)


#%% Optimization by exploring space ... direct approach

# Explore = np.arange(.001,0.500,.001)

# errors = 100*np.ones([Explore.size,Explore.size])

# for s1 in range(Explore.size):
#     print('iter: ' + str(s1) + ' curMin: ' + str(np.min(errors)))
#     for s2 in range(Explore.size):
#         err = mean_squared_error(BMLD, IACcurveFit(Window_t,Explore[s1],Explore[s2]))
#         errors[s1,s2] = err

# fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
# X,Y = np.meshgrid(Explore,Explore)
# ax.plot_surface(X,Y,errors)

# min_locs = np.where(errors<=np.min(errors))
# Explore[min_locs[0]]
# symGaussMin_loc = np.where(min_locs[0] == min_locs[1]) #where is the symmetric gaussian if there is one 

#%% Analyze minimum

#symGaussMin_loc is at index 60 ... Explore[60] or 61 ms
BMLDest = IACcurveFit(Window_t,.061,.061)

plt.figure()
plt.plot(Window_t,BMLD)
plt.plot(Window_t,BMLDest)

stDevNeg = .061
stDevPos = .061
tneg = np.arange(-stDevNeg*4,0,1/fs)
tpos = np.arange(0,stDevPos*4+.01,1/fs)

simpGaussNeg = np.exp(-0.5*(tneg/stDevNeg)**2)
simpGaussPos = np.exp(-0.5*(tpos/stDevPos)**2)

simpGauss = np.concatenate((simpGaussNeg,simpGaussPos))
simpGauss = simpGauss / np.sum(simpGauss)

plt.figure()
plt.plot(simpGauss)

w, h_behFit = freqz(simpGauss,a=1,worN=2000,fs=fs)

plt.figure()
plt.plot(w,20*np.log10(np.abs(h_behFit)))

plt.figure()
plt.plot(w,np.abs(h_behFit))






