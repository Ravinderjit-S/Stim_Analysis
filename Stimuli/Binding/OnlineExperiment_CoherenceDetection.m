%% Online Coherence detection Experiment

clear; close all; clc;
path = '../CommonExperiment';
p=genpath(path);
addpath(p);
addpath('Stim_Dev')

Stim_fold = '/home/ravinderjit/Documents/OnlineStim_WavFiles/TemporalCoherence';

%% Stim Parameters 

Mod_cf = [6,40,107];
Mod_halfWidth = 4;

fs = 48828;
f_start = 600;
f_end = 8000;
Tones_num = 8; %ERB spacing will be roughly 3 with 8 tones b/t 600-8000
Corr_inds = 1:Tones_num;


bp_fo = 1/2 * 4 *fs; %setting fo here so that all modulation bands have similar filter edges
stim_dur = 1.0;
risetime = .1;
silence = zeros(2,round(stim_dur/2*fs));

ntrials = 20;
nconds = length(Mod_cf);

Mod_cf = repmat(Mod_cf,1,ntrials*nconds); 
Mod_cf = Mod_cf(randperm(length(Mod_cf)));

%% Generate Wav Files

for i = 1:length(Mod_cf)
    bw = [Mod_cf(i)-Mod_halfWidth Mod_cf(i)+Mod_halfWidth];
    [stim_Ref, stimA, stimB, stimA2, envs_A, envs_B, ~, ERBspace, Tones_f] = Stim_Bind_ABA(Corr_inds, fs, f_start, f_end, Tones_num, [],bw,bp_fo,[]);
    stims = vertcat(stim_Ref, stimA, stimA2,stimB);
    order = randperm(3);
    order = [1 order+1];
    stim = [];
    for j =1:length(order)
        stim_j = [];
        stim_j(1,:) = stims(order(j),:);
        stim_j(1,:) = rampsound(stim_j(1,:),fs,risetime);
        stim_j(1,:) = scaleSound(stim_j(1,:));
        stim_j(2,:) = stim_j(1,:);
       
        stim = horzcat(stim, stim_j, silence); %#ok
    end
    stim = stim(1:end-length(silence)); %remove last silence
    
end


