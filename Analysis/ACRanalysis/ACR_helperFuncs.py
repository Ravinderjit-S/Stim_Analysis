#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:57:46 2021

@author: ravinderjit
"""

import numpy as np
from scipy.signal import freqz

def ACR_sourceHf(split_locs,ACR,t,fs,f1,f2):
    tpks = []
    pks = []
    pks_Hf = []
    pks_w = []
    pks_phase = []
    pks_phaseLine = []
    pks_phaseLineW = []
    pks_gd = []
    
    for pk in range(len(split_locs)):
        if pk ==0:
            t_1 = 0
        else:
            t_1 = split_locs[pk-1]
            
        t_2 = split_locs[pk]
        
        tpks.append(t[t_1:t_2])
        pks.append(ACR[t_1:t_2])
        
        [w, p_Hf] = freqz(b= ACR[t_1:t_2] - ACR[t_1:t_2].mean() ,a=1,worN=np.arange(0,fs/2,2),fs=fs)
        
        f_ind1 = np.where(w>=f1[pk])[0][0]
        f_ind2 = np.where(w>=f2[pk]+2)[0][0]
        
        phase_pkresp = np.unwrap(np.angle(p_Hf))
        coeffs= np.polyfit(w[f_ind1:f_ind2],phase_pkresp[f_ind1:f_ind2],deg=1)
        pks_phaseLine.append(coeffs[0] * w[f_ind1:f_ind2] +coeffs[1])
        pks_phaseLineW.append(w[f_ind1:f_ind2])
        pks_gd.append(-coeffs[0] / (2*np.pi))
        
        pks_w.append(w)
        pks_Hf.append(p_Hf)
        pks_phase.append(phase_pkresp)
         
    
        
    return tpks, pks, pks_Hf, pks_w, pks_phase, pks_phaseLine, pks_phaseLineW, pks_gd
    