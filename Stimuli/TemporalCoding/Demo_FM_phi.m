%3 AFC to determine phase difference detectable at different FMrates

clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p);

subj = input('Please subject ID:', 's');
%% Stim & Experimental parameters
FMs_test = [4,16,64,128];
phi_test = [30, 75, 90, 180];
L=70; %dB SPL
ntrials = 5;
nconds = numel(FMs_test) * numel(phi_test);
diotic = 0; %send carriers to different ears if 1
frange = [500 6000]; % range of the carriers
fratio = 4; % ratio of 2 carriers ... 4 = 2 octaves


FMs = repmat(FMs_test,1,ntrials*length(phi_test));
phis = repmat(phi_test,ntrials * length(FMs_test),1);
phis = phis(:)';

rand_order = randperm(length(FMs));
FMs = FMs(rand_order);
phis = phis(rand_order);

risetime = 0.100;
TypePhones = 'earphones';
stim_dur = 1.0; %duration of each FM, 3 will be played per trial
fs =48828.125;
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

%% listen demo
demo1 = true;
while demo1
    f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
    f2 = fratio*f1; 
    stim = FM_phi(f1,f2,fs,stim_dur,FMs_test(1),phis(end-1),diotic); %first stim
    info = sprintf('Answer is 3');
    info2 = sprintf('Press any button to play stim');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('DrawText',PS.window,info2,textlocH,textlocV+100,PS.white);
    Screen('Flip',PS.window);
    if buttonBox  %Subject pushes button twice
        getResponse(PS.RP);
    else
        getResponseKb; %#ok<UNRCH>
    end
    Screen('Flip',PS.window);
    for j = 1:3
        PlayStim(stim{j},fs,risetime,PS,L, useTDT, num2str(j), [], TypePhones);
        WaitSecs(stim_dur + 0.3); %wait 0.3 seconds b/t each stim
    end
    
    info = sprintf('To hear again, press 1. To continue, press 2');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    resp = getResponse(PS.RP);
    if resp ~= 1
        demo1 = false;
    end
end
   
demo2 = true;
while demo2
    f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
    f2 = fratio*f1; 
    stim = FM_phi(f1,f2,fs,stim_dur,FMs_test(3),phis(end-1),diotic); %first stim
    info = sprintf('Answer is 3');
    info2 = sprintf('Press any button to play stim');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('DrawText',PS.window,info2,textlocH,textlocV+100,PS.white);
    Screen('Flip',PS.window);
    if buttonBox  %Subject pushes button once
        getResponse(PS.RP);
    else
        getResponseKb; %#ok<UNRCH>
    end
    Screen('Flip',PS.window);
    for j = 1:3
        PlayStim(stim{j},fs,risetime,PS,L, useTDT, num2str(j), [], TypePhones);
        WaitSecs(stim_dur + 0.3); %wait 0.3 seconds b/t each stim
    end
    
    info = sprintf('To hear again, press 1. To continue, press 2');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    resp = getResponse(PS.RP);
    if resp ~= 1
        demo2 = false;
    end
end

%% practice run
info = sprintf('Now for a full practice run!');
info2 = sprintf('Press any button twice when ready');
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

f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
f2 = fratio*f1; 
stim = FM_phi(f1,f2,fs,stim_dur,FMs(1),phis(1),diotic); %first stim

for i =1:ntrials*nconds
        
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    
    PlayOrder= randperm(3);
    stim = stim(PlayOrder);
    for j = 1:3
        PlayStim(stim{j},fs,risetime,PS,L, useTDT, num2str(j), [], TypePhones);
        tic();
        if j == 3 && i~= ntrials*nconds
            f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
            f2 = fratio*f1; 
            stim = FM_phi(f1,f2,fs,stim_dur,FMs(i+1),phis(i+1),diotic); %first stim
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
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correctList(end));
    
    fprintf(1, 'Response =%d, answer =%d, Correct = %d, fm = %d, phi = %d \n', resp, correctList(end),resp==correctList(end), FMs(i), phis(i));
    respList = [respList, resp]; %#ok<AGROW>
    
end

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
    




