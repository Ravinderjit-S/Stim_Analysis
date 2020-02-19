#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:27:00 2019

@author: ravinderjit
"""

import numpy as np
import scipy as sp
import scipy.signal as spg


def periodogram(x,fs,nfft):
    #x should have time in first dimension (columns)
    N = int(nfft)
    Xf = sp.fft(x,axis=0,n=N)
    f = np.arange(0,N)*fs/N
    spec = abs(Xf)/N
    
    if np.mod(N,2) == 0:
        half_index = N/2;
    else:
        half_index = (N-1)/2; 
    
    f = f[:half_index]
    spec = spec[:half_index]
    pxx = 2*spec**2
    return f, pxx
    
    

def mts(x,TBW,fs,nfft):
    #x should have time in first dimension (columns)
    #TBW = full time bandwidth product, number of tapers is 2*TBW-1
    N = int(nfft)
    ntaps = int(TBW/2 - 1)
    w = spg.windows.dpss(x.shape[0],TBW/2.,Kmax = ntaps) #these windows have unity power
    f = np.arange(0,N)*fs/N
    x = np.matlib.repmat(x,ntaps,1).T
    Xf_taps = abs(sp.fft(x*w.T,axis=0,n=N))
    #Xf_taps = Xf_taps / (np.sum(w**2,axis=1).T / N)
    mtspec = Xf_taps.mean(axis=1)
    
    
    if np.mod(N,2) == 0:
        half_index = N/2;
    else:
        half_index = (N-1)/2; 
    
    f = f[:half_index]
    mtspec = mtspec[:half_index]
    
    mtPSD = 2*mtspec**2
    return f, mtPSD
    
    


