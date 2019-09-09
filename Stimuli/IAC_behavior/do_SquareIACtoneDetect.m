%3 AFC, 2 down 1up to find SNRs that 'Da' can be detected in different
%OSCOR cases

clear all; close all hidden; clc; %#ok<CLALL>
p = genpath('.');
addpath(p);
load('s.mat')
rng(s)

subj = input('Please subject ID:', 's');


%% Stim & Experimental parameters

WindowSizes = [1.6, .8, .4, .2, .15, .125, .1, 0.075, .05, 0]; %WindowSizes =0 & 1000 are the static cases
WindowSizes = WindowSizes(randperm(numel(WindowSizes)));
L=70; %dB SPL
nconds = numel(WindowSizes);
digDrop = 5;
SubjectBreaks = [3:2:nconds];

risetime = 0.050;
TypePhones = 'earphones';
stim_dur_part = 0.8; %Don't change ... determined by stim function... to change this need to change function 
% ^^ stim_dur will be 0.8 + windowSize
fs =48828.125;
passive =0;
BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);

% 2 down, 1 up parameters
Start_SNR = 6; 
respList = [];
correctList = [];
responseTracks = cell(1,nconds);
StartResolution = 2; %dB
EndResolution = 1; %dB
Reversals_changeRes = 4;
Reversals_stop = 11;

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


for i = 1:nconds
    
    if any(i == SubjectBreaks)
        info = strcat('Break! ', num2str(i-1) ,'/10 Done: Press any button twice to begin...');

        Screen('DrawText',PS.window,info,textlocH,textlocV+line2line,PS.white);
        Screen('Flip',PS.window);
        
        if buttonBox  %Subject pushes button twice to begin
            getResponse(PS.RP);
            getResponse(PS.RP);
        else
            getResponseKb; %#ok<UNRCH>
            getResponseKb;
        end
    end
    
     
    if WindowSizes(i) == 0
        StaticIAC = 0;
        wsize = 0;
    elseif WindowSizes(i) == 1000
        StaticIAC = 1;
        wsize =0;
    else
        StaticIAC = [];
        wsize = WindowSizes(i);
    end
    
    SNRdes = Start_SNR;
    Changes = 0; %need to intialize
    Reversals = 0;
    TrackSNRs = SNRdes;
    Correct_2up = [0 0]; %1st index is current trial, second is last trial
    stim_dur = stim_dur_part + wsize;
    while Reversals < Reversals_stop
        [stim, noiseRMS, ~] = IACtSquare_ToneDetect(fs,BPfilt,wsize,SNRdes,StaticIAC);
        
        if Reversals < Reversals_changeRes %improve SNR resolution after certain number of reversals 
            ChangeSNR = StartResolution;
        else
            ChangeSNR = EndResolution;
        end
        
        PlayOrder= randperm(3);
        stim = stim(PlayOrder);
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
        Correct_2up = circshift(Correct_2up,1); Correct_2up(1) = Correct;
        fprintf(1, 'SNR =%d, Response =%d, answer =%d, Correct = %d, Reversals = %d, Wsize = %d, done= %d/%d, \n',SNRdes, resp, answer,Correct, Reversals,WindowSizes(i),i,nconds);
        
        if Correct
            if all(Correct_2up)
                SNRdes = SNRdes-ChangeSNR;
                Correct_2up = [0 0];
            end
        else
            SNRdes = SNRdes+ChangeSNR;
        end
       
        TrackSNRs = [TrackSNRs SNRdes]; %#ok
        CurLength = length(Changes);
        Changes = sign(diff(TrackSNRs)); Changes = Changes(abs(Changes)>0);
        if length(Changes)>1 && Changes(end)~= Changes(end-1) && length(Changes) > CurLength  
            Reversals = Reversals+1;
        end
    end
    responseTracks{i} = TrackSNRs(1:end-1); %getting rid of last point cause I never play that
    save([subj '_IACtSquareToneDetect.mat'],'WindowSizes','responseTracks')    
end
        
save([subj '_IACtSquareToneDetect.mat'],'WindowSizes','responseTracks')

Screen('DrawText',PS.window,'Experiment is Over!',PS.rect(3)/2-150,PS.rect(4)/2-25,PS.white);
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
    




