function [spontRate, drivenRate, spontSTD, drivenSTD] = SpontaneousRate(spikes,stim_time,period)
%Designed for output from NEL
%spikes = 2 columns, first column is trial#, second is spike times
%stim_time = time stimulus is on
%period = stim_time + off time


Trials = max(spikes(:,1));
stim_time = stim_time/1000; period = period/1000; %convert to seconds 
spont_trials = [];
driven_trials = [];
for i=1:Trials
    spks_i = spikes(spikes(:,1) == i,2);
    spont_i = sum(spks_i>stim_time) ./ (period-stim_time); %spont rate for a trial in spks/sec
    driven_i = length(spks_i) - sum(spks_i>stim_time);
    spont_trials(i) = spont_i;
    driven_trials(i) = driven_i;
end

spontRate = mean(spont_trials);
spontSTD = std(spont_trials);
drivenRate = mean(driven_trials);
drivenSTD = std(driven_trials);


