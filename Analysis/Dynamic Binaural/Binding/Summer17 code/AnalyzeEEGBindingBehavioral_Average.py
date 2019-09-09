# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:47:23 2017

@author: rav28
"""
import numpy as np
import scipy as sp
import mne
from anlffr.helper import biosemi2mne as bs
import pylab as pl

rawHB_1, eventsHB_1 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjHB/SubjHB_BindingBehavioral.bdf', mask=None) #making mask none changes event numbers
rawHB_2, eventsHB_2 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjHB/SubjHB_BindingBehavioral+001.bdf', mask=None)
#raw1, events1 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjJP/SubjJP_BindingBehavioral.bdf', mask=None) #making mask none changes event numbers
#raw2, events2 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjJP/SubjJP_BindingBehavioral+001.bdf', mask=None)
rawRS_1, eventsRS_1 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjRS/SubjRS_BindingBehavioral.bdf', mask=None) #making mask none changes event numbers
rawRS_2, eventsRS_2 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjRS/SubjRS_BindingBehavioral+001.bdf', mask=None)
rawRS_3, eventsRS_3 = bs.importbdf('C:/Users/rav28/Google Drive/Purdue Summer rotation 2017/Data/Human data/BindingEEG/SubjRS/SubjRS_BindingBehavioral+002.bdf', mask=None)



#if Mask is none: events 1:4 = 65281-84, button presses are 65280
rawlistHB = [rawHB_1, rawHB_2]
eventlistHB = [eventsHB_1, eventsHB_2]

rawlistRS = [rawRS_1, rawRS_2, rawRS_3]
eventlistRS = [eventsRS_1, eventsRS_2, eventsRS_3]

rawHB, evesHB = mne.concatenate_raws(rawlistHB, events_list=eventlistHB)
rawHB.filter(1.0, 100) #filter the eeg 

rawRS, evesRS = mne.concatenate_raws(rawlistRS, events_list=eventlistRS)
rawRS.filter(1.0,100)

rawHB._data[36,:] = rawHB._data[34,:] - rawHB._data[35, :] #replacing an empty channel with the difference of the two eog channels to use for blink detection. This is gonna be channel EXG5
rawRS._data[36,:] = rawRS._data[34,:] - rawRS._data[35, :] #replacing an empty channel with the difference of the two eog channels to use for blink detection. This is gonna be channel EXG5

from anlffr.preproc import find_blinks 
blinksHB = find_blinks(rawHB, ch_name = ['EXG5'], thresh = 100e-6)
blinksRS = find_blinks(rawRS, ch_name = ['EXG5'], thresh = 100e-6)
scalings = dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4)
scalings['eeg'] = 40e-6
scalings ['misc'] = 150e-6
#raw.plot(events =blinks, scalings = scalings, proj = False)

#eog_events = mne.preprocessing.find_eog_events(raw,ch_name='EXG5')
#eog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name='EXG5')


from mne.preprocessing.ssp import compute_proj_epochs
epochs_blinksHB = mne.Epochs(rawHB, blinksHB, 998, tmin=-0.25,
                           tmax=0.25, proj=False, # why is proj true is we havent generated one yet?
                           baseline=(-0.25, 0), # what is point of baseline correction for blinks
                           reject=dict(eeg=500e-6)) 
epochs_blinksRS = mne.Epochs(rawRS, blinksRS, 998, tmin=-0.25,
                           tmax=0.25, proj=False, # why is proj true is we havent generated one yet?
                           baseline=(-0.25, 0), # what is point of baseline correction for blinks
                           reject=dict(eeg=500e-6)) 

blink_projsHB = compute_proj_epochs(epochs_blinksHB, n_grad=0,
                                  n_mag=0, n_eeg=6,verbose='DEBUG')
blink_projsRS = compute_proj_epochs(epochs_blinksRS, n_grad=0,
                                  n_mag=0, n_eeg=6,verbose='DEBUG')

#All_projs = mne.compute_proj_raw(raw,start=0,stop=None,duration=None,n_grad=0,n_mag=0,n_eeg=10,reject=dict(eeg=500e-6),verbose = 'DEBUG')

#ica = mne.preprocessing.ICA()
#ica.fit(raw)

blink_projsHB = [blink_projsHB[0], blink_projsHB[2]] 
blink_projsRS = [blink_projsRS[0], blink_projsRS[1]]
rawHB.add_proj(blink_projsHB)
rawRS.add_proj(blink_projsRS) 
#raw.add_proj(All_projs)
rawHB.plot_projs_topomap()
rawRS.plot_projs_topomap()


#raw2 = raw.copy()
#raw2.filter(40,100, picks = 35)
#
#raw2._data[35,:] = np.maximum(raw2._data[35,:],0)
#
#raw2.filter(0.1,10, picks = 35)
#raw2.plot( scalings = scalings)
#blinks_muscle = find_blinks(raw2, ch_name = ['EXG4'], thresh = 10e-6, event_id = 997)
#raw2.plot(events = blinks_muscle, scalings=scalings)
#epochs_muscle = mne.Epochs(raw, blinks_muscle,997, tmin = -0.5, tmax = 0.5, proj=True, baseline=(-.25,0), reject=dict(eeg=500e-6))
#muscle_projs = compute_proj_epochs(epochs_muscle, n_grad=0, n_mag =0, n_eeg =6, verbose='DEBUG')
#muscle_projs = [muscle_projs[0], muscle_projs[3]]
#raw.add_proj(muscle_projs)
#raw.plot_projs_topomap()
#raw.plot(events=blinks_muscle, scalings=scalings, show_options=True)


e1 = 65281
e2 = 65282
e3 = 65283
e4 = 65284
e_button = 65280 

#epochsA1 = mne.Epochs(raw, eves, [e1, e2, e3, e4], tmin=-0.4, tmax=1., proj=True, baseline=(-0.05, 0.), reject=dict(eeg=70e-6)) 
#evokedA1 = epochsA1.average()         
#
#epochsAB1_e1 = mne.Epochs(raw, eves, [e1], tmin=0.6, tmax=2., proj=True, baseline=(0.8, 1.), reject=dict(eeg=70e-6)) 
#evokedAB1_e1 = epochsAB1_e1.average()
#
#epochsAB1_e2 = mne.Epochs(raw, eves, [e2], tmin=0.6, tmax=2., proj=True, baseline=(0.8, 1.), reject=dict(eeg=70e-6)) 
#evokedAB1_e2 = epochsAB1_e2.average()
#
#epochsAB1_e3 = mne.Epochs(raw, eves, [e3], tmin=0.6, tmax=2., proj=True, baseline=(0.8, 1.), reject=dict(eeg=70e-6)) 
#evokedAB1_e3 = epochsAB1_e1.average()
#      
#epochsAB1_e4 = mne.Epochs(raw, eves, [e4], tmin=0.6, tmax=2., proj=True, baseline=(0.8, 1.), reject=dict(eeg=70e-6)) 
#evokedAB1_e4 = epochsAB1_e4.average()
#
#epochsAB2_e1 = mne.Epochs(raw, eves, [e1], tmin=2.6, tmax=4., proj=True, baseline=(2.8, 3.), reject=dict(eeg=70e-6)) 
#evokedAB2_e1 = epochsAB2_e1.average()
#
#epochsAB2_e2 = mne.Epochs(raw, eves, [e2], tmin=2.6, tmax=4., proj=True, baseline=(2.8, 3.), reject=dict(eeg=70e-6)) 
#evokedAB2_e2 = epochsAB2_e2.average()
#
#epochsAB2_e3 = mne.Epochs(raw, eves, [e3], tmin=2.6, tmax=4., proj=True, baseline=(2.8, 3.), reject=dict(eeg=70e-6)) 
#evokedAB2_e3 = epochsAB2_e1.average()
#
#epochsAB2_e4 = mne.Epochs(raw, eves, [e4], tmin=2.6, tmax=4., proj=True, baseline=(2.8, 3.), reject=dict(eeg=70e-6)) 
#evokedAB2_e4 = epochsAB2_e4.average()     

#epochsAB12_e4 = mne.epochs.concatenate_epochs([epochsAB1_e4, epochsAB2_e4])

#evokedAB12_e1 = evokedAB1_e1;
#evokedAB12_e1._data = (evokedAB1_e1._data + evokedAB2_e1._data) /2
#
#evokedAB12_e2 = evokedAB1_e2;
#evokedAB12_e2._data = (evokedAB1_e2._data + evokedAB2_e2._data) /2
#
#evokedAB12_e3 = evokedAB1_e3;
#evokedAB12_e3._data = (evokedAB1_e3._data + evokedAB2_e3._data) /2
#
#evokedAB12_e4 = evokedAB1_e4;
#evokedAB12_e4._data = (evokedAB1_e4._data + evokedAB2_e4._data) /2

#Make plots
ylim_vals = [-4,2.5]
channels = [4, 25, 26, 30,31]


#evokedA1.plot_topomap(times=np.asarray([0.1, 0.18]), show_names=True) ->HB
#evokedA1.plot_topomap(times=np.asarray([0.120, 0.235]), show_names=True) 
#evokedA1.plot_topomap(times=np.asarray([0.080, 0.183]), show_names=True) 
#evokedA1.plot(picks=channels, titles = 'A1_eA')
#evokedAB1_e1.plot(picks=channels,  titles = 'AB1_e1',  ylim = dict(eeg = ylim_vals))
#evokedAB1_e2.plot(picks=channels,  titles = 'AB1_e2',  ylim = dict(eeg = ylim_vals))
#evokedAB1_e3.plot(picks=channels,  titles ='AB1_e3',   ylim = dict(eeg = ylim_vals))
#evokedAB1_e4.plot(picks=channels,  titles = 'AB1_e4',  ylim = dict(eeg = ylim_vals))
#evokedAB2_e1.plot(picks=channels,  titles = 'AB2_e1',  ylim = dict(eeg =ylim_vals))
#evokedAB2_e2.plot(picks=channels,  titles ='AB2_e2',   ylim = dict(eeg = ylim_vals))
#evokedAB2_e3.plot(picks=channels,  titles = 'AB2_e3',  ylim = dict(eeg = ylim_vals))
#evokedAB2_e4.plot(picks=channels,  titles = 'AB2_e4',  ylim = dict(eeg = ylim_vals))
#evokedAB12_e1.plot(picks=channels, titles = 'AB12_e1', ylim = dict(eeg = ylim_vals))
#evokedAB12_e2.plot(picks=channels, titles = 'AB12_e2', ylim = dict(eeg = ylim_vals))
#evokedAB12_e3.plot(picks=channels, titles = 'AB12_e3', ylim = dict(eeg =ylim_vals))
#evokedAB12_e4.plot(picks=channels, titles = 'AB12_e4', ylim = dict(eeg = ylim_vals))






#Time-Frequency data

#epochs_buttons = mne.Epochs(raw, eves, e_button, tmin = -1, tmax = 0.5, proj = True, baseline=(0, 0.2), reject=dict(eeg=70e-6)) 

epochs_e1HB = mne.Epochs(rawHB, evesHB, e1, tmin=-0.4, tmax=4.4, proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 
epochs_e2HB = mne.Epochs(rawHB, evesHB, e2, tmin=-0.4, tmax=4.4, proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 
epochs_e3HB = mne.Epochs(rawHB, evesHB, e3, tmin=-0.4, tmax=4.4, proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 
epochs_e4HB = mne.Epochs(rawHB, evesHB, e4, tmin=-0.4, tmax=4.4, proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 

epochs_e1RS = mne.Epochs(rawRS, evesRS, e1, tmin=-0.4, tmax=4.4, proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 
epochs_e2RS = mne.Epochs(rawRS, evesRS, e2, tmin=-0.4, tmax=4.4, proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 
epochs_e3RS = mne.Epochs(rawRS, evesRS, e3, tmin=-0.4, tmax=4.4, proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 
epochs_e4RS = mne.Epochs(rawRS, evesRS, e4, tmin=-0.4, tmax=4.4, proj=True, baseline=(-0.2, 0.), reject=dict(eeg=70e-6)) 


freqs = np.arange(1.,100.,1.)
n_cycles = freqs/5.
time_bandwidth = 2.0;
vmin = -10;
vmax = 10; 

#power_buttons = mne.time_frequency.tfr_multitaper(epochs_buttons, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False)
#power_buttons.plot_topo(baseline = (0.2, 0.5), mode = 'zlogratio', title = 'button_press')

power_e1HB = mne.time_frequency.tfr_multitaper(epochs_e1HB, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e2HB = mne.time_frequency.tfr_multitaper(epochs_e2HB, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e3HB = mne.time_frequency.tfr_multitaper(epochs_e3HB, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e4HB = mne.time_frequency.tfr_multitaper(epochs_e4HB, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)

power_e1RS = mne.time_frequency.tfr_multitaper(epochs_e1RS, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e2RS = mne.time_frequency.tfr_multitaper(epochs_e2RS, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e3RS = mne.time_frequency.tfr_multitaper(epochs_e3RS, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
power_e4RS = mne.time_frequency.tfr_multitaper(epochs_e4RS, freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)

power_e1_avg = (power_e1HB + power_e1RS) / 2
power_e2_avg = (power_e2HB + power_e2RS) / 2
power_e3_avg = (power_e3HB + power_e3RS) / 2
power_e4_avg = (power_e4HB + power_e4RS) / 2

#power_e1_in = mne.time_frequency.tfr_multitaper(epochs_e1.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
#power_e2_in = mne.time_frequency.tfr_multitaper(epochs_e2.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
#power_e3_in = mne.time_frequency.tfr_multitaper(epochs_e3.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)
#power_e4_in = mne.time_frequency.tfr_multitaper(epochs_e4.subtract_evoked(), freqs=freqs, n_cycles = n_cycles, time_bandwidth = time_bandwidth, return_itc = False, picks = channels)


power_e1_avg.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e1', vmin=vmin,vmax=vmax)
power_e2_avg.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e2',  vmin=vmin,vmax=vmax)
power_e3_avg.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e3',  vmin=vmin,vmax=vmax)
power_e4_avg.plot_topo(baseline = (-0.4, 0), mode= 'zlogratio', title = 'e4',  vmin=vmin,vmax=vmax)
#power_e1_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e1_induced',  vmin=vmin,vmax=vmax)
#power_e2_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e2_induced', vmin=vmin,vmax=vmax)
#power_e3_in.plot_topo( baseline = (-0.4, 0), mode= 'zlogratio', title = 'e3_induced', vmin=vmin,vmax=vmax)
#power_e4_in.plot_topo(baseline = (-0.4, 0), mode= 'zlogratio', title = 'e4_induced', vmin=vmin,vmax=vmax)



