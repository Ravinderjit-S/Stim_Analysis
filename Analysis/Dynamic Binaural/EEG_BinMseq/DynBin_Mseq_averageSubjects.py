#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:57:32 2020

@author: ravinderjit
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pickle
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'

def Zscore(X,noise):
    #X and noise is shape frequency x instances
    Z = (X.mean(axis=1) - noise.mean(axis=1)) / noise.std(axis=1)
    Z_Zsem = (X.mean(axis=1) + X.std(axis=1)/np.sqrt(X.shape[1]) - noise.mean(axis=1)) / noise.std(axis=1)
    Z_sem = Z_Zsem - Z
    return Z, Z_sem


#TW = 20 for files without TW label
data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles/SystemFuncs')
fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/DynBin')

Subjects = ['S001','S132','S203','S204','S205','S206','S207','S208','S211']

All_CohIAC = np.zeros([65536,len(Subjects)])
All_CohITD = np.zeros([65536,len(Subjects)])
All_IAChf = np.zeros([41,len(Subjects)])
All_ITDhf = np.zeros([41,len(Subjects)])

All_PlvIAC = np.zeros([65536,len(Subjects)])
All_PlvITD = np.zeros([65536,len(Subjects)])

All_phaseIAC = np.zeros([65536,len(Subjects)])
All_phaseITD = np.zeros([65536,len(Subjects)])


All_CohIAC_nfs = np.zeros([65536,100])
All_CohITD_nfs = np.zeros([65536,100])
All_PlvIAC_nfs = np.zeros([65536,100])
All_PlvITD_nfs = np.zeros([65536,100])

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

    with open(os.path.join(data_loc, Subject+'_DynBin_SysFuncTW10.pickle'),'rb') as f:     
        IAC_Ht, IAC_nfs, IAC_Hf, NF_Hfs_IAC, PLV_IAC, Coh_IAC, PLVnf_IAC, Cohnf_IAC, \
        ITD_Ht, ITD_nfs, ITD_Hf, NF_Hfs_ITD, PLV_ITD, Coh_ITD, PLVnf_ITD, Cohnf_ITD, f1,f2,t, phase_IAC, phase_ITD = pickle.load(f)
        
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
    
    All_PlvIAC[:, sub] = PLV_IAC
    All_PlvITD[:, sub] = PLV_ITD
    
    All_phaseIAC[:,sub] = phase_IAC
    All_phaseITD[:,sub] = phase_ITD
    
    
    All_CohIAC_nfs += Cohnf_IAC / len(Subjects)
    All_CohITD_nfs += Cohnf_ITD / len(Subjects)
    All_PlvIAC_nfs += PLVnf_IAC / len(Subjects)
    All_PlvITD_nfs += PLVnf_ITD / len(Subjects)
    All_IAChf_nfs += NF_Hfs_IAC.T / len(Subjects)
    All_ITDhf_nfs += NF_Hfs_ITD.T / len(Subjects)

    
    
#%% Plot Responses

plt.figure()
plt.plot(f2,All_PlvIAC)

#%%  Average across subjects
# z = (x-u) / sigma

ZIAC_coh, ZIAC_coh_sem = Zscore(All_CohIAC,All_CohIAC_nfs)
ZITD_coh, ZITD_coh_sem = Zscore(All_CohITD,All_CohITD_nfs)

ZIAC_plv, ZIAC_plv_sem = Zscore(All_PlvIAC,All_PlvIAC_nfs)
ZITD_plv, ZITD_plv_sem = Zscore(All_PlvITD,All_PlvITD_nfs)

ZIAC_Hf, ZIAC_Hf_sem = Zscore(10*np.log10(All_IAChf),10*np.log10(All_IAChf_nfs))
ZITD_Hf, ZITD_Hf_sem = Zscore(10*np.log10(All_ITDhf),10*np.log10(All_ITDhf_nfs))


fig, ax = plt.subplots()
resp = ax.plot(f2,ZIAC_plv,color='black',linewidth=2, label='Response')
conf = ax.fill_between(f2,ZIAC_plv + 1.96*ZIAC_plv_sem,ZIAC_plv - 1.96*ZIAC_plv_sem,
                        color=mcolors.CSS4_COLORS['grey'],alpha=0.7,linewidth=0, label='95% Confidence')
ax.set_xlim([1,25])
ax.set_ylim([0,25])
ax.set_xscale('log')
ax.set_ylabel('IAC PLV (Zscore re: NoisFloor)', fontsize=12,fontweight='bold')
ax.set_xlabel('Frequency (Hz)',fontsize=12,fontweight='bold')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
fig.savefig(os.path.join(fig_path,'IAC_plv.eps'),format='eps')
fig.savefig(os.path.join(fig_path,'IAC_plv.png'),format='png')


fig, ax = plt.subplots()
resp = ax.plot(f2,ZITD_plv,color='black',linewidth=2)
conf = ax.fill_between(f2,ZITD_plv + 1.96*ZITD_plv_sem, ZITD_plv - 1.96*ZITD_plv_sem,
         color=mcolors.CSS4_COLORS['grey'],alpha=0.7,linewidth=0, label='95% Confidence')
ax.set_xlim([1,25])
ax.set_ylim([0,25])
ax.set_xscale('log')
ax.set_ylabel('ITD PLV (Zscore re: NoisFloor)', fontsize=12,fontweight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
fig.savefig(os.path.join(fig_path,'ITD_plv.eps'),format='eps')
fig.savefig(os.path.join(fig_path,'ITD_plv.png'),format='png')


font_size = 10
fig, ax = plt.subplots(figsize=(3.3,3.3))
resp = ax.plot(f2,ZIAC_coh, color='black', linewidth=2, label='Response')
conf = ax.fill_between(f2,ZIAC_coh + 1.96*ZIAC_coh_sem, ZIAC_coh - 1.96*ZIAC_coh_sem,
                 color=mcolors.CSS4_COLORS['grey'],alpha=0.7,linewidth=0, label='95% Confidence')
ax.set_xlim([1,25])
ax.set_ylim([0,25])
ax.set_xscale('log')
ax.set_ylabel('Coherence (Zscore re: NoisFloor)', fontsize=font_size)
ax.set_xlabel('Frequency (Hz)',fontsize=font_size)
#ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xticks([1,10,20],['1','10','20'],fontsize=font_size)
plt.title('IAC')
plt.tight_layout()
fig.savefig(os.path.join(fig_path,'IAC_coh.eps'),format='eps')
fig.savefig(os.path.join(fig_path,'IAC_coh.png'),format='png')
fig.savefig(os.path.join(fig_path,'IAC_coh.svg'),format='svg')

fig, ax = plt.subplots(figsize=(3.3,3.3))
resp = ax.plot(f2,ZITD_coh, color = 'black', linewidth =2, label='Response')
conf = ax.fill_between(f2,ZITD_coh + 1.96*ZITD_coh_sem, ZITD_coh - 1.96*ZITD_coh_sem,
                        color=mcolors.CSS4_COLORS['grey'],alpha=0.7,linewidth=0, label='95% Confidence')
ax.set_xlim([1,25])
ax.set_ylim([0,25])
ax.set_xscale('log')
ax.legend(frameon=False)
ax.set_ylabel('Coherence (Zscore re: NoisFloor)', fontsize=font_size)
ax.set_xlabel('Frequency (Hz)',fontsize=font_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xticks([1,10,20],['1','10','20'],fontsize=font_size)
plt.title('ITD',fontsize=font_size)
plt.tight_layout()
fig.savefig(os.path.join(fig_path,'ITD_coh.eps'),format='eps')
fig.savefig(os.path.join(fig_path,'ITD_coh.png'),format='png')
fig.savefig(os.path.join(fig_path,'ITD_coh.svg'),format='svg')


# plt.figure()
# plt.plot(f1,ZIAC_Hf)
# plt.plot(f1,ZIAC_Hf + 1.96*ZIAC_Hf_sem,color='r')
# plt.plot(f1,ZIAC_Hf - 1.96*ZIAC_Hf_sem,color='r')
# plt.xlim([1,20])
# plt.xscale('log')
# plt.ylabel('Power dB (Zscore re: NoiseFloor)')
# plt.title('IAC')

# plt.figure()
# plt.plot(f1,ZITD_Hf)
# plt.plot(f1,ZITD_Hf + 1.96*ZITD_Hf_sem,color='r')
# plt.plot(f1,ZITD_Hf - 1.96*ZITD_Hf_sem,color='r')
# plt.xlim([1,20])
# plt.xscale('log')
# plt.ylabel('Power dB (Zscore re: NoiseFloor)')
# plt.title('ITD')

# plt.figure()
# plt.plot(f2,ZITD_coh)
# plt.plot(f2,ZIAC_coh,color='r')
# plt.xlim([1,20])

f_1 = np.where(f2>=1)[0][0]
f_15 = np.where(f2>=15)[0][0]
f_index = np.arange(f_1,f_15)

IAC_fmax = f2[ZIAC_coh[f_index].argmax() +f_1]
ITD_fmax = f2[ZITD_coh[f_index].argmax() + f_1]

#%% Phase Response
fig, ax = plt.subplots()
ax.plot(f2,All_phaseIAC)
plt.title('Phase IAC')
plt.xlim([0,20])
plt.ylim([-4,4])

fig, ax = plt.subplots()
ax.plot(f2,All_phaseITD.mean(axis=1))
plt.title('Phase ITD')
plt.xlim([0,20])
plt.ylim([-4,4])













