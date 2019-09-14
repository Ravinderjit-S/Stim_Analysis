# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:43:33 2017

@author: rav28
"""

import mne
from anlffr.helper import biosemi2mne as bs
raw, eves = bs.importbdf('D:/DATA/BindingEEG/SubjMS/SubjMS_Binding.bdf')
from anlffr.preproc import find_blinks
blinks = find_blinks(raw)

scalings = dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4)
scalings['eeg'] = 40e-6
scalings['misc'] = 150e-6
raw.plot(events=blinks, scalings=scalings)
from mne.preprocessing.ssp import compute_proj_epochs
epochs_blinks = mne.Epochs(raw, blinks, 998, tmin=-0.25,
                           tmax=0.25, proj=True,
                           baseline=(-0.25, 0),
                           reject=dict(eeg=500e-6))
blink_projs = compute_proj_epochs(epochs_blinks, n_grad=0,
                                  n_mag=0, n_eeg=2,
                                  verbose='DEBUG')


raw.add_proj(blink_projs)

raw.plot(events=blinks, scalings=scalings, show_options=True)
x = raw._data
x.shape
epochs = mne.Epochs(raw, eves, [1, 2, 3, 4], tmin=-0.4, tmax=1., proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6))

import numpy as np
evoked.plot_topomap(times=np.asarray([0.1, 0.18]), show_names=True)
evoked.plot()
evoked.plot(picks=[4, 25, 30, 32])
raw.info['highpass']
raw.info['lowpass']
raw.info['sfreq']
raw.filter(1.0, 40.)
epochs = mne.Epochs(raw, eves, [1, 2, 3, 4], tmin=-0.4, tmax=1., proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6))
evoked = epochs.average()
evoked
evoked.plot(picks=[4, 25, 30, 32])
eves