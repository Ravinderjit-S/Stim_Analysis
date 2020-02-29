#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:57:32 2020

@author: ravinderjit
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pickle
import numpy as np
import scipy as sp


def Zscore(X,noise):
    #X and noise is shape frequency x instances
    Z = (X.mean(axis=1) - noise.mean(axis=1)) / noise.std(axis=1)
    Z_Zsem = (X.mean(axis=1) + X.std(axis=1)/np.sqrt(X.shape[1]) - noise.mean(axis=1)) / noise.std(axis=1)
    Z_sem = Z_Zsem - Z
    return Z, Z_sem


#TW = 20 for files without TW label
data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles/SystemFuncs')

Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']

All_CohIAC = np.zeros([65536,len(Subjects)])
All_CohITD = np.zeros([65536,len(Subjects)])
All_IAChf = np.zeros([41,len(Subjects)])
All_ITDhf = np.zeros([41,len(Subjects)])

All_CohIAC_nfs = np.zeros([65536,100])
All_CohITD_nfs = np.zeros([65536,100])
All_IAChf_nfs = np.zeros([41,100])
All_ITDhf_nfs = np.zeros([41,100])

# fig_IACht,ax_IACht = plt.subplots(3,3,sharex=True)
# fig_ITDht,ax_ITDht = plt.subplots(3,3,sharex=True)
# fig_IAChf,ax_IAChf = plt.subplots(3,3,sharex=True)
# fig_ITDhf,ax_ITDhf = plt.subplots(3,3,sharex=True)
# fig_IACcoh,ax_IACcoh = plt.subplots(3,3,sharex=True)
# fig_ITDcoh,ax_ITDcoh = plt.subplots(3,3,sharex=True)
# fig_IACplv,ax_IACplv = plt.subplots(3,3,sharex=True)

for sub in range(0,len(Subjects)):
    Subject = Subjects[sub]

    with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc.pickle'),'rb') as f:     
        IAC_Ht, IAC_nfs, IAC_Hf, NF_Hfs_IAC, PLV_IAC, Coh_IAC, PLVnf_IAC, Cohnf_IAC, \
        ITD_Ht, ITD_nfs, ITD_Hf, NF_Hfs_ITD, PLV_ITD, Coh_ITD, PLVnf_ITD, Cohnf_ITD, f1,f2,t = pickle.load(f)
        
    ax_row = int(np.floor(sub/3))
    ax_col = int(np.mod(sub,3))
    
    #%% Plot Stuff for each Subject
    # ax_IACht[ax_row,ax_col].plot(t,IAC_nfs.T,color= mcolors.CSS4_COLORS['grey'])
    # ax_IACht[ax_row,ax_col].plot(t,IAC_Ht,color='k')
    # ax_IACht[ax_row,ax_col].set_xlim([0,1])
    # ax_IACht[ax_row,ax_col].set_title(Subject + ' Ht IAC')
    
    # ax_ITDht[ax_row,ax_col].plot(t,ITD_nfs.T,color= mcolors.CSS4_COLORS['grey'])
    # ax_ITDht[ax_row,ax_col].plot(t,ITD_Ht,color='k')
    # ax_ITDht[ax_row,ax_col].set_xlim([0,1])
    # ax_ITDht[ax_row,ax_col].set_title(Subject + ' Ht ITD')
    
    # ax_IAChf[ax_row,ax_col].plot(f1,10*np.log10(NF_Hfs_IAC.T),color= mcolors.CSS4_COLORS['grey'])
    # ax_IAChf[ax_row,ax_col].plot(f1,10*np.log10(IAC_Hf),color='k')
    # ax_IAChf[ax_row,ax_col].set_xlim([1,20])
    # ax_IAChf[ax_row,ax_col].set_title(Subject + ' CrossCorr IAC')
    # ax_IAChf[ax_row,ax_col].set_xscale('log')
        
    # ax_ITDhf[ax_row,ax_col].plot(f1,10*np.log10(NF_Hfs_ITD.T),color= mcolors.CSS4_COLORS['grey'])
    # ax_ITDhf[ax_row,ax_col].plot(f1,10*np.log10(ITD_Hf),color='k')
    # ax_ITDhf[ax_row,ax_col].set_xlim([1,20])
    # ax_ITDhf[ax_row,ax_col].set_title(Subject + ' CrossCorr ITD')
    # ax_ITDhf[ax_row,ax_col].set_xscale('log')
    
    # ax_IACcoh[ax_row,ax_col].plot(f2,Cohnf_IAC,color= mcolors.CSS4_COLORS['grey'])
    # ax_IACcoh[ax_row,ax_col].plot(f2,Coh_IAC,color='k')
    # ax_IACcoh[ax_row,ax_col].set_xlim([1,20])
    # ax_IACcoh[ax_row,ax_col].set_title(Subject + ' Coh IAC')
    # ax_IACcoh[ax_row,ax_col].set_xscale('log')
    
    # ax_ITDcoh[ax_row,ax_col].plot(f2,Cohnf_ITD,color= mcolors.CSS4_COLORS['grey'])
    # ax_ITDcoh[ax_row,ax_col].plot(f2,Coh_ITD,color='k')
    # ax_ITDcoh[ax_row,ax_col].set_xlim([1,20])
    # ax_ITDcoh[ax_row,ax_col].set_title(Subject + ' Coh ITD')
    # ax_ITDcoh[ax_row,ax_col].set_xscale('log')
    
    # ax_IACplv[ax_row,ax_col].plot(f2,PLVnf_IAC,color= mcolors.CSS4_COLORS['grey'])
    # ax_IACplv[ax_row,ax_col].plot(f2,PLV_IAC,color='k')
    # ax_IACplv[ax_row,ax_col].set_xlim([1,20])
    # ax_IACplv[ax_row,ax_col].set_title(Subject + ' PLV IAC')
    # ax_IACplv[ax_row,ax_col].set_xscale('log')

    #%% Extract Data out from each Subject
    All_CohIAC[:,sub] = Coh_IAC
    All_CohITD[:,sub] = Coh_ITD
    All_IAChf[:,sub] = IAC_Hf
    All_ITDhf[:,sub] = ITD_Hf
    
    All_CohIAC_nfs += Cohnf_IAC / len(Subjects)
    All_CohITD_nfs += Cohnf_ITD / len(Subjects)
    All_IAChf_nfs += NF_Hfs_IAC.T / len(Subjects)
    All_ITDhf_nfs += NF_Hfs_ITD.T / len(Subjects)
    
    
#%%  Average across subjects
# z = (x-u) / sigma

ZIAC_coh, ZIAC_coh_sem = Zscore(All_CohIAC,All_CohIAC_nfs)
ZITD_coh, ZITD_coh_sem = Zscore(All_CohITD,All_CohITD_nfs)

ZIAC_Hf, ZIAC_Hf_sem = Zscore(10*np.log10(All_IAChf),10*np.log10(All_IAChf_nfs))
ZITD_Hf, ZITD_Hf_sem = Zscore(10*np.log10(All_ITDhf),10*np.log10(All_ITDhf_nfs))

plt.figure()
plt.plot(f2,ZIAC_coh)
plt.plot(f2,ZIAC_coh + 1.96*ZIAC_coh_sem,color='r')
plt.plot(f2,ZIAC_coh - 1.96*ZIAC_coh_sem,color='r')
plt.xlim([1,20])
plt.ylim([0,30])
plt.xscale('log')
plt.ylabel('Coherence (Zscore re: NoisFloor)')
plt.title('IAC')

plt.figure()
plt.plot(f2,ZITD_coh)
plt.plot(f2,ZITD_coh + 1.96*ZITD_coh_sem,color='r')
plt.plot(f2,ZITD_coh - 1.96*ZITD_coh_sem,color='r')
plt.xlim([1,20])
plt.ylim([0,30])
plt.xscale('log')
plt.ylabel('Coherence (Zscore re: NoisFloor)')
plt.title('ITD')

plt.figure()
plt.plot(f1,ZIAC_Hf)
plt.plot(f1,ZIAC_Hf + 1.96*ZIAC_Hf_sem,color='r')
plt.plot(f1,ZIAC_Hf - 1.96*ZIAC_Hf_sem,color='r')
plt.xlim([1,20])
plt.xscale('log')
plt.ylabel('Power dB (Zscore re: NoiseFloor)')
plt.title('IAC')

plt.figure()
plt.plot(f1,ZITD_Hf)
plt.plot(f1,ZITD_Hf + 1.96*ZITD_Hf_sem,color='r')
plt.plot(f1,ZITD_Hf - 1.96*ZITD_Hf_sem,color='r')
plt.xlim([1,20])
plt.xscale('log')
plt.ylabel('Power dB (Zscore re: NoiseFloor)')
plt.title('ITD')

plt.figure()
plt.plot(f2,ZITD_coh)
plt.plot(f2,ZIAC_coh,color='r')
plt.xlim([1,20])




