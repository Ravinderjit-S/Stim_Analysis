# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy as sp
import mne
from anlffr.helper import biosemi2mne as bs
import pylab as pl

#raw1, events1 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjMS/SubjMS_Binding.bdf')
#raw2, events2 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjMS/SubjMS_Binding+001.bdf')
#raw1, events1 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjHB/SubjHB_Binding.bdf')
#raw2, events2 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjHB/SubjHB_Binding+001.bdf')
#raw1, events1 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjRS/SubjRS_Binding.bdf') 
#raw2, events2 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjRS/SubjRS_Binding+001.bdf')

raw1, events1 = bs.importbdf('/media/ravinderjit/Data_Drive/EEGdata/Summer17/SubjRS_Binding.bdf') 
raw2, events2 = bs.importbdf('/media/ravinderjit/Data_Drive/EEGdata/Summer17/SubjRS_Binding+001.bdf') 

rawlist = [raw1, raw2]
eventlist = [events1, events2]

raw, eves = mne.concatenate_raws(rawlist, events_list=eventlist)
raw.filter(1.0, 150) #filter the eeg 

raw._data[36,:] = raw._data[34,:] - raw._data[35, :] #replacing an empty channel with the difference of the two eog channels to use for blink detection. This is gonna be channel EXG5

from anlffr.preproc import find_blinks 
#The below line was blinks = find_blinks(raw, ch_name = ['EXG5'], thresh = 100e-6) ... changed to below on 06/12/18 
blinks = find_blinks(raw, ch_name = ['EXG5'], thresh = 100e-6,  l_trans_bandwidth=0.5, l_freq = 1.0) 
scalings = dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4)
scalings['eeg'] = 40e-6
scalings ['misc'] = 150e-6
#raw.plot(events =blinks, scalings = scalings, proj = False)

#eog_events = mne.preprocessing.find_eog_events(raw,ch_name='EXG5')
#eog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name='EXG5')


from mne.preprocessing.ssp import compute_proj_epochs
epochs_blinks = mne.Epochs(raw, blinks, 998, tmin=-0.25,
                           tmax=0.25, proj=False, # why is proj true is we havent generated one yet?
                           baseline=(-0.25, 0), # what is point of baseline correction for blinks
                           reject=dict(eeg=500e-6)) 
blink_projs = compute_proj_epochs(epochs_blinks, n_grad=0,
                                  n_mag=0, n_eeg=8, #n_eeg = 2 b/c blinks and horizontal movements
                                  verbose='DEBUG')

blink_projs = [blink_projs[0], blink_projs[2]]

raw.add_proj(blink_projs) #how much do we affect the neural data with this
raw.plot_projs_topomap()

raw.plot(events=blinks, scalings=scalings, show_options=True)


epochsA1 = mne.Epochs(raw, eves, [1, 2, 3], tmin=-0.4, tmax=1., proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 
evokedA1 = epochsA1.average()         

epochsAB1_e1 = mne.Epochs(raw, eves, [1], tmin=0.6, tmax=2., proj=True, baseline=(0.8, 1.), reject=dict(eeg=70e-6)) 
evokedAB1_e1 = epochsAB1_e1.average()

epochsAB1_e2 = mne.Epochs(raw, eves, [2], tmin=0.6, tmax=2., proj=True, baseline=(0.8, 1.), reject=dict(eeg=70e-6)) 
evokedAB1_e2 = epochsAB1_e2.average()

epochsAB1_e3 = mne.Epochs(raw, eves, [3], tmin=0.6, tmax=2., proj=True, baseline=(0.8, 1.), reject=dict(eeg=70e-6)) 
evokedAB1_e3 = epochsAB1_e1.average()

epochsAB2_e1 = mne.Epochs(raw, eves, [1], tmin=2.6, tmax=4., proj=True, baseline=(2.8, 3.), reject=dict(eeg=70e-6)) 
evokedAB2_e1 = epochsAB2_e1.average()

epochsAB2_e2 = mne.Epochs(raw, eves, [2], tmin=2.6, tmax=4., proj=True, baseline=(2.8, 3.), reject=dict(eeg=70e-6)) 
evokedAB2_e2 = epochsAB2_e2.average()

epochsAB2_e3 = mne.Epochs(raw, eves, [3], tmin=2.6, tmax=4., proj=True, baseline=(2.8, 3.), reject=dict(eeg=70e-6)) 
evokedAB2_e3 = epochsAB2_e1.average()

#epochsAB12_e4 = mne.epochs.concatenate_epochs([epochsAB1_e4, epochsAB2_e4])

evokedAB12_e1 = evokedAB1_e1;
evokedAB12_e1._data = (evokedAB1_e1._data + evokedAB2_e1._data) /2

evokedAB12_e2 = evokedAB1_e2;
evokedAB12_e2._data = (evokedAB1_e2._data + evokedAB2_e2._data) /2

evokedAB12_e3 = evokedAB1_e3;
evokedAB12_e3._data = (evokedAB1_e3._data + evokedAB2_e3._data) /2

#plots
ylim_vals = [-3.5,3]
channels = [4, 25, 26, 30,31]
channels2 = [3,4,5,7]
channels3 = [24,25,26]
evokedA1.plot_topomap(times=np.asarray([0.120, 0.180]), show_names=True)
evokedA1.plot(picks=channels, titles = 'A1_eA')
evokedAB1_e1.plot(picks=channels, titles = 'AB1_e1',ylim = dict(eeg = ylim_vals))
evokedAB1_e2.plot(picks=channels, titles = 'AB1_e2',ylim = dict(eeg = ylim_vals))
evokedAB1_e3.plot(picks=channels, titles ='AB1_e3',ylim = dict(eeg = ylim_vals))
evokedAB2_e1.plot(picks=channels, titles = 'AB2_e1', ylim = dict(eeg = ylim_vals))
evokedAB2_e2.plot(picks=channels, titles ='AB2_e2',ylim = dict(eeg = ylim_vals))
evokedAB2_e3.plot(picks=channels, titles = 'AB2_e3',ylim = dict(eeg = ylim_vals))
evokedAB12_e1.plot(picks=channels, titles = 'AB12_e1',ylim = dict(eeg = ylim_vals))
evokedAB12_e2.plot(picks=channels, titles = 'AB12_e2',ylim = dict(eeg = ylim_vals))
evokedAB12_e3.plot(picks=channels, titles = 'AB12_e3',ylim = dict(eeg = ylim_vals))


epochs_e1 = mne.Epochs(raw, eves, 1, tmin=-0.4, tmax=4.4, proj=False, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 
epochs_e2 = mne.Epochs(raw, eves, 2, tmin=-0.4, tmax=4.4, proj=False, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 
epochs_e3 = mne.Epochs(raw, eves, 3, tmin=-0.4, tmax=4.4, proj=False, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 

freqs = np.arange(1.,100.,1.)
n_cycles = freqs/5.
time_bandwidth = 2.0
vmin = -10
vmax = 10

power_e1 = mne.time_frequency.tfr_multitaper(epochs_e1, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False)
power_e1.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e1', vmin=vmin,vmax=vmax)

power_e2 = mne.time_frequency.tfr_multitaper(epochs_e2, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e2.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e2',vmin=vmin,vmax=vmax)

power_e3 = mne.time_frequency.tfr_multitaper(epochs_e3, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e3.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e3',vmin=vmin,vmax=vmax)

power_e1_in = mne.time_frequency.tfr_multitaper(epochs_e1.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e1_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e1_induced',vmin=vmin,vmax=vmax)

power_e2_in = mne.time_frequency.tfr_multitaper(epochs_e2.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e2_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e2_induced',vmin=vmin,vmax=vmax)

power_e3_in = mne.time_frequency.tfr_multitaper(epochs_e3.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False)
power_e3_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e3_induced',vmin=vmin,vmax=vmax)
