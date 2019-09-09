# -*- coding: utf-8 -*-
"""
Created on Sat Sep 08 21:30:13 2018

@author: StuffDeveloping
EEG pre-processing
"""
from os import listdir
from anlffr.helper import biosemi2mne as bs
from mne import concatenate_raws

def EEGconcatenateFolder(folder, nchans,refchans,exclude=[]):
    #concatenates all the EEG files in one folder
    #Assumes files are in format output from biosemi ... subj.bdf, subj+001.bdf, etc.
    #Also folder should end with a '/' 
    EEGfiles = listdir(folder)
    EEGfiles.sort()# This line and next to fix order of files 
    EEGfiles.insert(0,EEGfiles.pop(len(EEGfiles)-1))
    print(EEGfiles)
    raw = [] 
    events = []
    for eeg_f in EEGfiles:
        raw_f, events_f = bs.importbdf(folder+eeg_f,nchans,refchans,exclude =exclude) #uses EXG1 and EXG2 as reference usually
        raw.append(raw_f)
        events.append(events_f)
    EEG_full, events_full = concatenate_raws(raw,events_list=events)
    return EEG_full, events_full