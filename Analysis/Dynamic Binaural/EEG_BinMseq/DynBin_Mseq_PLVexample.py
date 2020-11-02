#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 00:10:26 2020

@author: ravinderjit
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pickle
import numpy as np
import scipy as sp

matplotlib.rcParams['font.family'] ='Arial'

def Zscore(X,noise):
    #X and noise is shape frequency x instances
    Z = (X.mean(axis=1) - noise.mean(axis=1)) / noise.std(axis=1)
    Z_Zsem = (X.mean(axis=1) + X.std(axis=1)/np.sqrt(X.shape[1]) - noise.mean(axis=1)) / noise.std(axis=1)
    Z_sem = Z_Zsem - Z
    return Z, Z_sem

data_loc = os.path.abspath('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Pickles/SystemFuncs')
fig_path = os.path.abspath('/media/ravinderjit/Data_Drive/Data/Figures/DynBin')



Subject = 'S207'

with open(os.path.join(data_loc, Subject+'_DynBin_SysFunc.pickle'),'rb') as f:     
    IAC_Ht, IAC_nfs, IAC_Hf, NF_Hfs_IAC, PLV_IAC, Coh_IAC, PLVnf_IAC, Cohnf_IAC, \
    ITD_Ht, ITD_nfs, ITD_Hf, NF_Hfs_ITD, PLV_ITD, Coh_ITD, PLVnf_ITD, Cohnf_ITD, f1,f2,t = pickle.load(f)
    
    

fig, ax = plt.subplots()
ax.plot(f2, PLV_IAC, color = 'black' )
ax.plot(f2, PLVnf_IAC, color = mcolors.CSS4_COLORS['grey'])
ax.set_xlim([1,25])
#ax.set_ylim([0,25])
ax.set_xscale('log')
ax.set_ylabel('IAC PLV ', fontsize=12,fontweight='bold')
ax.set_xlabel('Frequency (Hz)',fontsize=12,fontweight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
fig.savefig(os.path.join(fig_path,'S207_IAC_plv.eps'),format='eps')
fig.savefig(os.path.join(fig_path,'S207_IAC_plv.png'),format='png')


  
fig, ax = plt.subplots(figsize=(2.25,2.15))
font_size = 8.5
ax.plot(f2, Coh_IAC, color = 'black',linewidth=2 )
ax.plot(f2, Cohnf_IAC, color = mcolors.CSS4_COLORS['grey'])
ax.set_xlim([1,25])
#ax.set_ylim([0,25])
ax.set_xscale('log')
ax.set_ylabel('Coherence', fontsize=font_size)
ax.set_xlabel('Frequency (Hz)',fontsize=font_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xticks([1,10,20],['1','10','20'],fontsize=font_size)
plt.yticks([.050,.15],['.05', '.15'],fontsize=font_size)
plt.title('H(f)', fontweight='bold',fontsize=font_size)
plt.tight_layout()
fig.savefig(os.path.join(fig_path,'S207_IAC_coh.eps'),format='eps')
fig.savefig(os.path.join(fig_path,'S207_IAC_coh.png'),format='png')
fig.savefig(os.path.join(fig_path,'S207_IAC_coh.svg'),format='svg')

  
    
    
    

    
    
        
        