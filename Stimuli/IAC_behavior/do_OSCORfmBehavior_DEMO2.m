%3 AFC to find highest detectable OSCOR fm

clear all; close all hidden; clc; %#ok<CLALL>
p = genpath('.');
addpath(p);
load('s.mat')
rng(s)

%subj = input('Please subject ID:', 's');


%% Stim & Experimental parameters
%FMs_test = [5, 10, 20, 40, 70, 100, 150, 200, 250, 300, 500];

L=70; %dB SPL
ntrials = 5;
FMs = [ones(1,ntrials)*40,ones(1,ntrials)*80, ones(1,ntrials)*160];
nconds = numel(unique(FMs));
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

info = strcat('Case 1: The Correct is Answer is 3. Try to hear whats different!');
Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
info = strcat('When you are ready: Press any button to begin listening...');
Screen('DrawText',PS.window,info,textlocH,textlocV+line2line,PS.white);
Screen('Flip',PS.window);
getResponse(PS.RP);
stim = IACsinAFC3(stim_dur,fs,40,BPfilt);

Demo_Resp = 1;
while Demo_Resp ==1
    for j = 1:3
        PlayStim_Binaural_V3(stim{j},fs,risetime,PS,L, useTDT, num2str(j), 1, TypePhones,passive, [], digDrop,1)
        tic();
        WaitSecs(stim_dur + 0.3); %wait 0.3 seconds b/t each stim
    end
    info = ('Press 1 to repeat. Press 3 to move on');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    Demo_Resp = getResponse(PS.RP);
end
 
info = strcat('Case 2: The Correct is Answer is 3. Try to hear whats different!');
Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
info = strcat('When you are ready: Press any button to begin listening...');
Screen('DrawText',PS.window,info,textlocH,textlocV+line2line,PS.white);
Screen('Flip',PS.window);
getResponse(PS.RP);
stim = IACsinAFC3(stim_dur,fs,80,BPfilt);
stim = IACsinAFC3(stim_dur,fs,80,BPfilt);

Demo_Resp = 1;
while Demo_Resp ==1
    for j = 1:3
        PlayStim_Binaural_V3(stim{j},fs,risetime,PS,L, useTDT, num2str(j), 1, TypePhones,passive, [], digDrop,1)
        tic();
        WaitSecs(stim_dur + 0.3); %wait 0.3 seconds b/t each stim
    end
    info = ('Press 1 to repeat. Press 3 to move on');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    Demo_Resp = getResponse(PS.RP);
end
    
info = strcat('Case 3: The Correct is Answer is 3. Try to hear whats different!');
Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
info = strcat('When you are ready: Press any button to begin listening...');
Screen('DrawText',PS.window,info,textlocH,textlocV+line2line,PS.white);
Screen('Flip',PS.window);
getResponse(PS.RP);
stim = IACsinAFC3(stim_dur,fs,160,BPfilt);
stim = IACsinAFC3(stim_dur,fs,160,BPfilt);

Demo_Resp = 1;
while Demo_Resp ==1
    for j = 1:3
        PlayStim_Binaural_V3(stim{j},fs,risetime,PS,L, useTDT, num2str(j), 1, TypePhones,passive, [], digDrop,1)
        tic();
        WaitSecs(stim_dur + 0.3); %wait 0.3 seconds b/t each stim
    end
    info = ('Press 1 to repeat. Press 3 to move on');
    Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
    Screen('Flip',PS.window);
    Demo_Resp = getResponse(PS.RP);
end



stim = IACsinAFC3(stim_dur,fs,FMs(1),BPfilt); %first stim
for i =1:ntrials*nconds
    fprintf(1, 'Running Trial #%d/%d\n',i, ntrials*nconds);
    
    PlayOrder= randperm(3);
    stim = stim(PlayOrder);
    for j = 1:3
        PlayStim_Binaural_V3(stim{j},fs,risetime,PS,L, useTDT, num2str(j), 1, TypePhones,passive, [], digDrop,1)
        tic();
        if j == 3 && i~= ntrials*nconds
            stim = IACsinAFC3(stim_dur,fs,FMs(i+1),BPfilt);
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
    resp = GetResponse_Feedback(PS, feedback, feedbackDuration,buttonBox, correctList(i),textlocH,textlocV,line2line);
    
    fprintf(1, 'Response =%d, answer =%d, Correct = %d, OSCORfm=%d Hz \n', resp, correctList(i),resp==correctList(i),FMs(i));
    respList = [respList, resp]; %#ok<AGROW>
    
end

%save([subj '_OSCORfmThresh.mat'],'FMs','ntrials','respList','correctList')

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
    




