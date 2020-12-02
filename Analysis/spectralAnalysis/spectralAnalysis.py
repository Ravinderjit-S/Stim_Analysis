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
        half_index = int(N/2);
    else:
        half_index = int((N-1)/2); 
    
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
    
def PLV_Coh(X,Y,TW,fs):
    """
    X is the Mseq
    Y is time x trials
    TW is half bandwidth product 
    """
    X = X.squeeze()
    ntaps = 2*TW - 1
    dpss = sp.signal.windows.dpss(X.size,TW,ntaps)
    N = int(2**np.ceil(np.log2(X.size)))
    f = np.arange(0,N)*fs/N
    PLV_taps = np.zeros([N,ntaps])
    Coh_taps = np.zeros([N,ntaps])
    
    for k in range(0,ntaps):
        print('tap:',k+1,'/',ntaps)
        Xf = sp.fft(X *dpss[k,:],axis=0,n=N)
        Yf = sp.fft(Y * dpss[k,:].reshape(dpss.shape[1],1),axis=0,n=N)
        XYf = Xf.reshape(Xf.shape[0],1) * Yf.conj()
        PLV_taps[:,k] = abs(np.mean(XYf / abs(XYf),axis=1))
        Coh_taps[:,k] = abs(np.mean(XYf,axis=1) / np.mean(abs(XYf),axis=1))
        
    PLV = PLV_taps.mean(axis=1)
    Coh = Coh_taps.mean(axis=1)
    return PLV, Coh, f


