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

L=75; %dB SPL
ntrials = 300;
nconds =1;
jitlist = rand(1,ntrials*nconds)*0.1;
risetime = 1/150; 
TypePhones = 'earphones';
fs =48828.125;
passive =1;

% mseq1 = load('mseqEEG_150_reps1.mat'); %Ponit_len,bits,fs,mseqEEG
% mseq2 = load('mseqEEG_150_reps3.mat'); %Ponit_len,bits,fs,mseqEEG

mseq1 = load('mseqEEG_150_bits7.mat');
mseq2 = load('mseqEEG_150_bits8.mat');
mseq3 = load('mseqEEG_150_bits9.mat');
mseq4 = load('mseqEEG_150_bits10.mat');

mseqs = [mseq1,mseq2, mseq3,mseq4];

%% Startup parameters
FsamptTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =0;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox =0;

feedbackDuration =0.2;

% Turns EEG Saving on ('Pause off')
invoke(PS.RP, 'SetTagVal', 'trgname',253);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);
pause(2.0);


for i =1:ntrials*nconds
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    order = randperm(length(mseqs));
    for j = 1:length(order)
        mseq_j = mseqs(order(j));
        stim_dur = length(mseq_j.mseqEEG)/fs;
        tic();
        stim = AM_mseq(mseq_j.mseqEEG,mseq_j.Point_len);
        genStimTime = toc();
        PlayStim(stim,fs,risetime,PS,L,useTDT, 'NONE', order(j), TypePhones);
        WaitSecs(stim_dur - genStimTime + 0.5 + rand()*0.1);
    end
end

% % Turns EEG Saving off ('Pause on')
invoke(PS.RP, 'SetTagVal', 'trgname', 254);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

close_play_circuit(f1RZ,PS.RP);
fprintf(1,'\n Done with data collection!\n');



