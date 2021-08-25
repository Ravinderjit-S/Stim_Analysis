%This code will be for the first part of the comodulation binding
%experiment collecting only behavioral data on 12 conditions
clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions
addpath('Stim_Dev')
subj = input('Please subject ID:', 's');


%% Stim & Experimental parameters
load('s.mat')
rng(s)

L = 75; %dB SPL
respList = []; %vector that will contain the subject's responses 
correctList = [];
cohList = [];
risetime = 0.050; %made 50 b/c envelope can change at speed of up to 24 Hz which is .041 secs
TypePhones = 'earphones';
stim_dur = 1.0;
stim_isi = 0.3; % time between 3AFC stims
ERB_halfwidth = 0.5;
ERBspacing = 1.5;
target_f=4000;
noise_bands = CMRbands(target_f, ERB_halfwidth, ERBspacing);

mod_band = [4 24];
target_modf = 0;

coh = [0,1];
SNR_0 = [-36,-28, -20];
SNR_1 = [-40, -30, -20];

ntrials = 5;


fs = 48828;


%% Startup parameters
FsampTDT = 3; % 48828.125 Hz
useTrigs = 1;
feedback = 1;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox = 1;

feedbackDuration = 0.2;

PS = psychStarter(useTDT,screenDist,screenWidth,useTrigs,FsampTDT);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);


%% Welcome to experiment
textlocH = PS.rect(3)/4;
textlocV = PS.rect(4)/3;
line2line = 50;

ExperimentWelcome(PS, buttonBox,textlocH,textlocV,line2line);
Screen('Flip',PS.window);

pause(2.0);

    
%% Experiment Begins

%Block Loop
for i=1:ntrials
    %% Break
    if mod(i,3) == 0 % optional break every 3 trials trials
        
        fprintf(1,'Break ----------- \n')
        
        info = sprintf('Break! You are about to start block %d/%d ',i,ntrial);
        info2 = sprintf('Press any button twice to resume');
        Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
        Screen('DrawText',PS.window,info2,textlocH,textlocV+100,PS.white);
        Screen('Flip',PS.window);
        if buttonBox  %Subject pushes button twice
            getResponse(PS.RP);
            getResponse(PS.RP);
        else
            getResponseKb; %#ok<UNRCH>
            getResponseKb;
        end
        Screen('Flip',PS.window);
        pause(2.0);
    end
    
    %% Incoherent mod Loop
  
    [Sig,answer] = CMR_randMod_clicky_3AFC(noise_bands,target_f,SNR_0(1),mod_band,target_modf,fs,tlen,1,risetime, stim_isi);
    for j = 1:length(SNR_0)
        PlayStim(Sig,fs,0,PS,L,useTDT,[],[], TypePhones); 
        tic();
        if j ~= length(SNR_0)
            [Sig,answer] = CMR_randMod_clicky_3AFC(noise_bands,target_f,SNR_0(j+1),mod_band,target_modf,fs,tlen,1,risetime, stim_isi);
        end
        StimGenTime = toc();
        WaitSecs(stim_dur*3 + stim_isi*2 +0.3 - StimGenTime);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  Response Frame
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        correctList = [correctList answer]; %#ok<AGROW>
        cohList = [cohList 0]; %#ok<AGROW>
        resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correctList(end));
        
        respList = [respList resp]; %#ok<AGROW>
        fprintf('Response = %d, correct = %d, coh = %d, SNR = %d dB, block = %d/%d, trial = %d/%d\n', resp, answer, 0, SNR_0(j), i, ntrials, j,length(SNR_0))
        
    end
    
    %% Coherent mod Loop
    [Sig,answer] = CMR_randMod_clicky_3AFC(noise_bands,target_f,SNR_1(1),mod_band,target_modf,fs,tlen,1,risetime, stim_isi);
    for j = 1:length(SNR_1)
        PlayStim(Sig,fs,0,PS,L,useTDT,[],[], TypePhones); 
        tic();
        if j ~= length(SNR_1)
            [Sig,answer] = CMR_randMod_clicky_3AFC(noise_bands,target_f,SNR_1(j+1),mod_band,target_modf,fs,tlen,1,risetime, stim_isi);
        end
        StimGenTime = toc();
        WaitSecs(stim_dur*3 + stim_isi*2 +0.3 - StimGenTime);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  Response Frame
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        correctList = [correctList answer]; %#ok<AGROW>
        cohList = [cohList 1]; %#ok<AGROW>
        resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correctList(end));
        
        respList = [respList resp]; %#ok<AGROW>
        fprintf('Response = %d, correct = %d, coh = %d, SNR = %d dB, block = %d/%d, trial = %d/%d\n', resp, answer, 1, SNR_1(j), i, ntrials, j,length(SNR_1))
        
    end

end
save(strcat(subj, '_CMRrandModClicky'), 'respList','correctList','Corr_inds','CorrSet');

Screen('DrawText',PS.window,'Experiment is Over!',PS.rect(3)/2-150,PS.rect(4)/2-25,PS.white);
Screen('DrawText',PS.window,'Thank You for Your Participation!',PS.rect(3)/2-150,PS.rect(4)/2+100,PS.white);
Screen('Flip',PS.window);
WaitSecs(5.0);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

close_play_circuit(PS.f1,PS.RP);
fprintf(1,'\n Done with data collection!\n');
sca;

