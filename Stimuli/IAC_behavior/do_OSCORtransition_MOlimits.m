% Employ method of limits to determine where the OSCOR stimulus goes from
% spatial to a flutter

clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p);
addpath('Stimuli Dev')
load('s.mat')
rng(s)

subj = input('Please subject ID:', 's');


%% Stim & Experimental parameters
L=70; %dB SPL
Direction
nconds = numel(FMs_test);
series_n = 3;
series = [repmat('A',1,series_n) repmat('D',1,series_n)];
series = series(randperm(length(series))); %randomize order of ascending and descending trials
startF = [5 20]; %starting frequencies for ascending and descending


risetime = 0.050;
TypePhones = 'earphones';
stim_dur = 1; 
fs =48828.125;
passive =0;
BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);


%% Startup parameters
FsampTDT = 3; %48828.125 Hz
useTrigs =1;
feedback =0;
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



resp =1; %initializing for while loop
for i=1:length(series)
    if series(i) == 'A'
        OSCOR_fm = startF(1);
    else
        OSCOR_fm = startF(2);
    end
    stim = OSCOR(stim_dur,fs,OSCOR_fm,BPfilt,0);
    respList = [];
    fplayed = [];
    while resp ==1
        
        PlayStim(stim,fs,risetime,PS,L,useTDT,'NONE',[], TypePhones);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  Response Frame
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, []);

        fprintf(1, 'Response =%d, OSCORfm = %d \n', resp, OSCOR_fm);
        respList = [respList, resp]; %#ok<AGROW>
        fplayed = [f_played, OSCOR_fm];
        if series(i) == 'A'
            OSCOR_fm = OSCOR_fm +1;
        else
            OSCOR_fm = OSCOR_fm -1;
        end
    end
    A_respList{i} = respList;
    A_fplayed{i} = fplayed;
end

save([subj '_OSCORtransition.mat'], A_respList, A_fplayed)

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
    




