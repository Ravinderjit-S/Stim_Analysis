%3AFC to find depth at which AM is not detectable anymore

clear all; close all hidden; clc; %#ok<CLALL>
p = genpath('.');
addpath(p);
load('s.mat')
rng(s)

subj = input('Please subject ID:', 's');

%% Stim & Exprimental paramters
Depths_test = [0.2, 0.21, 0.22, 0.23, 0.25];
fm = 40; %hz
L = 70; %db SPL
ntrials = 20;
nconds = numel(Depths_test);

Depths = repmat(Depths_test,1,ntrials);
Depths = Depths(randperm(length(Depths)));

risetime = 0.050;
digDrop = 5;
TypePhones = 'earphones';
stim_dur = 1; %duration of each noise, 3 will be played per trial
fs =48828.125;
passive =0;
BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);
respList = [];
correctList = [];

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

stim = AMdetect(stim_dur,fs,Depths(1),fm, BPfilt, 0);

for i =1:ntrials*nconds
    
    if i == round(ntrials*nconds/2) % Break at half way point
        info = strcat('Break! Half way Done: Press any button twice to begin...');
        
        Screen('DrawText',PS.window,info,textlocH,textlocV+line2line,PS.white);
        Screen('Flip',PS.window);
        fprintf(1, 'Break \n');
        if buttonBox  %Subject pushes button twice to begin
            getResponse(PS.RP);
            getResponse(PS.RP);
        else
            getResponseKb; %#ok<UNRCH>
            getResponseKb;
        end
        fprintf(1,'Subject continued \n')
    end
        
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    
    PlayOrder= randperm(3);
    stim = stim(PlayOrder);
    for j = 1:3
        PlayStim_Binaural_V3(stim{j},fs,risetime,PS,L, useTDT, num2str(j), 1, TypePhones,passive, [], digDrop,1)
        tic();
        if j == 3 && i~= ntrials*nconds
            stim = AMdetect(stim_dur,fs,Depths(i+1),fm, BPfilt, 0);
            StimGenTime = toc();
        else
            StimGenTime = toc();
        end
        WaitSecs(stim_dur + 0.3 - StimGenTime); %wait 0.3 seconds b/t each stim
    end
    correctList = [correctList, find(PlayOrder ==3)]; %#ok
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Response Frame
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correctList(end),textlocH,textlocV,line2line);
    
    fprintf(1, 'Response =%d, answer =%d, Correct = %d, Depth = %.2f \n', resp, correctList(end),resp==correctList(end), Depths(i));
    respList = [respList, resp]; %#ok<AGROW>
    
end

save([subj '_AMdepthBehavior.mat'],'Depths','fm','ntrials','respList','correctList')

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
    








