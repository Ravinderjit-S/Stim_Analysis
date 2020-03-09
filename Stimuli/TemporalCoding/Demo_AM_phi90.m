%3 AFC to determine phase difference detectable at different FMrates

clear all; close all hidden; clc; %#ok<CLALL>
path = '../CommonExperiment';
p = genpath(path);
addpath(p);

subj = input('Please subject ID:', 's');
%% Stim & Experimental parameters
AMs_test = [4,16,64];
dichotic_test = [0 1]; %send carriers to different ears if 1
phi = 90;

L=70; %dB SPL
ntrials = 5;
nconds = numel(AMs_test) * numel(dichotic_test);
frange = [500 6000]; % range of the carriers
fratio = 4; % ratio of 2 carriers ... 4 = 2 octaves

AMs = repmat(AMs_test,1,ntrials*length(dichotic_test));
dichotics = repmat(dichotic_test,ntrials * length(AMs_test),1);
dichotics = dichotics(:)';

rand_order = randperm(length(AMs));
AMs = AMs(rand_order);
dichotics = dichotics(rand_order);

risetime = 0.125;
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
    stim = AM_phi(f1,f2,fs,stim_dur,AMs_test(1),phi,dichotic_test(1)); %first stim
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
    stim = AM_phi(f1,f2,fs,stim_dur,AMs_test(3),phi,dichotic_test(1)); %first stim
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

demo3 = true;
while demo3
    f1 = randi(frange(2)/fratio - frange(1)) + frange(1); 
    f2 = fratio*f1; 
    stim = AM_phi(f1,f2,fs,stim_dur,AMs_test(1),phi,dichotic_test(2)); %first stim
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
        demo3 = false;
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
stim = AM_phi(f1,f2,fs,stim_dur,AMs(1),phi,dichotics(1)); %first stim

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
            stim = FM_phi(f1,f2,fs,stim_dur,AMs(i+1),phi,dichotics(i+1)); %first stim
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
    
    fprintf(1, 'Response =%d, answer =%d, Correct = %d, AM = %d, phi = %d \n', resp, correctList(end),resp==correctList(end), AMs(i), dichotics(i));
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
    




