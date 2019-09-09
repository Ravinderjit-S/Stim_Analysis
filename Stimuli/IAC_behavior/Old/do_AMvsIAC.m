%3AFC to find which depths of AM are confused with OSCOR at what
%frequencies

clear all; close all hidden; clc; %#ok<CLALL>
p = genpath('.');
addpath(p);
load('s.mat')
rng(s)

subj = input('Please subject ID:', 's');

%% Stim & Exprimental paramters
Depths_test = [0.1,0.2,0.3,0.4];
FMs_test = [5,10,20,40]; %hz
Depth_FM = CombVec(Depths_test,FMs_test); 

L = 70; %db SPL
ntrials = 20;
nconds = numel(Depths_test);

Depths = repmat(Depth_FM(1,:),1,ntrials);
FMs = repmat(Depth_FM(2,:),1,ntrials);
reorder = randperm(length(Depths));
Depths = Depths(reorder); FMs = FMs(reorder);
TotTrials = length(FMs);

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

stim = IACsin_AM_AFC3(stim_dur,fs,Depths(1),FMs(1), BPfilt);

for i =1:TotTrials
    
    if mod(i,round(TotTrials/4)) == 0 && i~= 4*round(TotTrials/4) %Breaks
        info = strcat('Break: ', num2str(round((i/TotTrials)*100)) ,'% done! ...: Press any button twice to begin...');
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
        
    fprintf(1, 'Running Trial #%d/%d\n',i, TotTrials);
    
    PlayOrder= randperm(3);
    stim = stim(PlayOrder);
    for j = 1:3
        PlayStim_Binaural_V3(stim{j},fs,risetime,PS,L, useTDT, num2str(j), 1, TypePhones,passive, [], digDrop,1)
        tic();
        if j == 3 && i~= TotTrials
            stim = IACsin_AM_AFC3(stim_dur,fs,Depths(i+1),FMs(i+1), BPfilt);
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
    
    fprintf(1, 'Response =%d, answer =%d, Correct = %d, Depth = %.2f, fm = %d \n', resp, correctList(end),resp==correctList(end), Depths(i),FMs(i));
    respList = [respList, resp]; %#ok<AGROW>
    
end

save([subj '_AMvsIAC.mat'],'Depths','FMs','ntrials','respList','correctList')

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
    








