function [Spikes_jumbled] = Jumble_spikes(Spikes,stimDur)
%This function will jumble up spike times to generate noise floors for a
%Systems ID type analysis. This function does it by taking a randomizing
%the ISIs of the measured Spikes. Also look at stimDur, to only jumble 
%spikes during driven part of stimulus 
%Spikes is in the NEL format
%stimDur in seconds
%Spikes_jumbled is returned in the NEL format 

Spikes_jumbled = [];
Trials = unique(Spikes(:,1));
for i = 1:numel(Trials)
    spks_i = Spikes(Spikes(:,1)==Trials(i) & Spikes(:,2)<=stimDur,:);
    Init_spiktimes = Spikes(boolean(vertcat(1,diff(Spikes(:,1)))),2);
    ISI_i = diff(spks_i(:,2));
    ISI_jumbled = ISI_i(randperm(length(ISI_i)));%jumble Inter spike intervals (ISIs)
    if isempty(spks_i) % accounting for no spks in a trial
        spks_jumbled_i = [];
        warning(['No spikes in trial ' num2str(Trials(i))])
    else
        spks_jumbled_i = rand*Init_spiktimes(i);% draw first spike time as uniform distribution b/t 0 and first spike time for that trial
    end
    for j = 2:size(spks_i,1)
        spks_jumbled_i(j) = spks_jumbled_i(j-1) + ISI_jumbled(j-1);
    end
    Spikes_jumbled = vertcat(Spikes_jumbled, horzcat(spks_i(:,1),spks_jumbled_i'));
end
    
    



