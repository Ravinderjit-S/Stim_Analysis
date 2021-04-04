#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:44:35 2021

@author: ravinderjit
"""

import numpy as np

def mseqXcorr(epochs,mseq):
    #epochs in shape channels x trials x time
    resp_m = epochs.mean(axis=1)
    resp_m = resp_m - resp_m.mean(axis=1)[:,np.newaxis]
    Ht_m = np.zeros([resp_m.shape[0],resp_m.shape[1]+mseq.size-1])
    for ch in range(resp_m.shape[0]):
        Ht_m[ch,:] = np.correlate(resp_m[ch,:],mseq,mode='full')#[mseq[m].size-1:]
    
    return Ht_m
    