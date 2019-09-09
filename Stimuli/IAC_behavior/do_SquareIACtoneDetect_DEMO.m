%Demo for detecting brief tone in noise with square IAC changes
%OSCOR cases

clear all; close all hidden; clc; %#ok<CLALL>
p = genpath('.');
addpath(p);
load('s.mat')
rng(s)


%% Stim & Experimental parameters

WindowSizes = [.4, .2, .1, .05, .025];
L=70; %dB SPL
nconds = numel(WindowSizes);
ntrials = 2; 
wsize = [ones(1,ntrials)*WindowSizes(1),ones(1,ntrials)*WindowSizes(2),ones(1,ntrials)*WindowSizes(3),ones(1,ntrials)*WindowSizes(4),ones(1,ntrials)*WindowSizes(5)];
SNRs = repmat([6,0],1,length(WindowSizes));
digDrop = 5;

risetime = 0.050;
TypePhones = 'earphones';
stim_dur_part = 0.8; %Don't change ... determined by stim function... to change this need to change function
% ^^ stim_dur will be 0.8 + windowSize
fs =48828.125;
passive =0;
BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);


%% Startup parameters
FsampTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =1;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox =1;

feedbackDuration =0.2;

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

for i = 1:nconds*ntrials
    SNRdes = SNRs(i);
    
    [stim, noiseRMS, ~] = IACtSquare_ToneDetect(fs,BPfilt,wsize(i),SNRdes,[]);

    PlayOrder= randperm(3);
    stim = stim(PlayOrder);
    stim_dur = stim_dur_part + wsize(i);
    for j = 1:3
        PlayStim_Binaural_V3(stim{j},fs,risetime,PS,L, useTDT, num2str(j), 1, TypePhones,passive,noiseRMS,digDrop,0)
        WaitSecs(stim_dur + 0.3); %wait 0.3 seconds b/t each stim
    end
    answer = find(PlayOrder==3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Response Frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, answer,textlocH,textlocV,line2line);
    Correct = resp == answer;
    fprintf(1, 'SNR =%d, Response =%d, answer =%d, Correct = %d \n',SNRdes, resp, answer,Correct);

end

Screen('DrawText',PS.window,'Demo is Over!',PS.rect(3)/2-150,PS.rect(4)/2-25,PS.white);
Screen('DrawText',PS.window,'Thank You for Your Participation!',PS.rect(3)/2-150,PS.rect(4)/2+100,PS.white);
Screen('Flip',PS.window);
WaitSecs(5.0);

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

% Pause On
invoke(PS.RP, 'SetTagVal', 'trgname', 254);
invoke(PS.RP, 'SetTagVal', 'onsetdel',100);
invoke(PS.RP, 'SoftTrg', 6);

close_play_circuit(PS.f1,PS.RP);
fprintf(1,'\n Done with data collection!\n');
sca;





