function [t, A_fr] = SpikeProb(spikes, method, binsize, stim_time,plotit)
%Neural Response Tuning Curve = Average firing rate 
%Designed for output from NEL
%A_Fr = #of spks per bin
%t = times corresponding to average firing rate
%spikes = 2 columns, first column is trial, second is spike times 
%method = 'bin', implement others = sliding bin, gaussian, etc.
%binsize = in milliseconds
%stim_time = in millliseconds
%plotit

method == 'bin'; %implement other methods yo
    
Trials = max(spikes(:,1));
binsize = binsize /1000; %convert to seconds
stim_time = stim_time/ 1000; %convert to seconds 
tbins = [0:binsize:stim_time-binsize]+binsize/2; % by adding binsize/2, I center the bin for the hist function in matlab correctly, see documentation for hist for more info
spk_counts_Total = zeros(1,length(tbins));
for i = 1:Trials
    spk_times = spikes(spikes(:,1) == i,2);
    spk_times(spk_times > stim_time) = []; %only evaluate spikes up to stim_time 
    spk_counts = hist(spk_times, tbins);
    spk_counts_Total = spk_counts_Total + spk_counts;
end
A_fr = (spk_counts_Total/Trials);
t = tbins - binsize/2; %this is the time vector; interpretation: the value for time x represents number of spikes in range [x, x+binsize];

if plotit
    figure, plot(t,A_fr), xlabel('Time (sec)'), ylabel(['Prob of Spike per ' num2str(binsize*1000) ' ms ' method])
end


    
    
    
    
    
    
    