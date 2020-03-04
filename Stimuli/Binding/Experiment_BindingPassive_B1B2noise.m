%This code will be for the first part of the comodulation binding
%experiment collecting only behavioral data on 12 conditions
clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions
addpath('Stim_Dev')
subj = input('Please subject ID:', 's');

fig_num=99;
USB_ch=1;
FS_tag = 3;
[f1RZ,PS.RP,FS]=load_play_circuit(FS_tag,fig_num,USB_ch);


%% Stim & Experimental parameters
load('s2.mat')
rng(s)

L = 75; %dB SPL
respList = []; %vector that will contain the subject's responses 
risetime = 0.050; %made 50 b/c envelope can change at speed of up to 24 Hz which is .041 secs
TypePhones = 'earphones';
stim_dur = 4.0; %This is set by Stim_bind function, need to regenerate stimuli to change this
f_start = 100;
f_end = 8000;
Tones_num = 16;
fs = 48828;
load('StimActive_AB1AB2noise_passive.mat')

nconds = length(Corr_inds);
ntrials = 200; %trials per cond

jitlist = rand(1, ntrials*nconds)*0.2; %small jit to prevent any periodic background noise becoming in phase with desired signal


%% Startup parameters
FsampTDT = 3; % 48828.125 Hz
useTrigs = 1;
feedback = 1;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox = 1;

feedbackDuration = 0.2;


%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);


% Turns EEG Saving on ('Pause off')
invoke(PS.RP, 'SetTagVal', 'trgname',253);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);
pause(2.0);

    
%% Experiment Begins

for i=1:1:nconds*ntrials

    %% Play Stim
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    stim = [stims{i}; stims{i}];
    trig_i = CorrSet(i);
    PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', trig_i, TypePhones);
    WaitSecs(stim_dur + 1.5 + jitlist(i));
    
end

% Turns EEG Saving off ('Pause on')
invoke(PS.RP, 'SetTagVal', 'trgname', 254);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

fprintf(1,'\n Done with data collection!\n');

