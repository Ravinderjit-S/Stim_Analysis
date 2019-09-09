%Audiogram
%The goal is to screen subjects for normal hearing for inclusion into a
%study
clear all; close all hidden; clc; %#ok<CLALL>
p = genpath('.');
addpath(p);
subj = input('Please subject ID:', 's');

%% Audiogram parameters
%frequencies = [1000, 2000, 3000, 4000, 8000, 500, 250, 125];
frequencies = [1000, 2000, 4000, 8000, 500, 250];
%HLtoSPL_offsets = [8.8, 16.1, 18, 20.3, 12.3, 20.4]; %for ER2s from Han & Poulsen 1998
HLtoSPL_offsets = [2.7, 0.5, 0.1, 23.1, 8.6, 20.1];  %for HDA300
%HLtoSPL_offsets = [2.7, 0.5, -1.6, 0.1, 23.1, 8.6, 20.1, 26.2];
stim_dur = 1.0; %tone duration in seconds
fs = 48828.125; %sampling rate
risetime = 0.010;
T = 0:1/fs:stim_dur;
button_dur = 2; %wait this amount of time for subject to push a button after a stimulus
Initial_Level = 30; %initial tone played at this level dB HL
TypePhones = 'headphones'; 
ears = ['l' 'r'];
pattern = [0 0 0 0]; % gonna track the subjects correct or wrong
pattern_stop = [1 0 0 1]; %this is the stop pattern ... ex. patient gets 10db, misses 0dB, misses 5dB, gets 10dB again -> take 10 dB as thresh
FP_checks = []; %This will track false positives for random periods where no sound is played
fp_check = randi(5); %this variable is in how many trials should a FP_check be run
Exclude_Thresh = 25; %exclude subject if any threshold higher (HL)
%noise = rand(1,fs*stim_dur); %noise to filter to create narrow-band noise
%fm = 4; % frequency modulation of nbn

%transtone parameters
m = 0.8;
bw = 50;


%% Startup parameters
FsampTDT = 3; % 48828.125 Hz
useTrigs = 1;
feedback = 0;
useTDT = 1;
screenDist = 0.4;
screenWidth = 0.3;
buttonBox = 1;
resp = []; %need to initialize response variable

feedbackDuration = 0.2;

PS = psychStarter(useTDT,screenDist,screenWidth,useTrigs,FsampTDT);


%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

%% Experiment Welcome
textlocH = PS.rect(3)/4;
textlocV = PS.rect(4)/3;
line2line = 50;
AudiogramWelcome(PS, buttonBox,textlocH,textlocV,line2line);

%% Begin Audiogram

for j =1:2 %left ear and right ear
    Thresholds = nan(1,numel(frequencies)); %Audiogram Thresholds
    for i = 1:numel(frequencies)
        %stim = sin(2*pi*frequencies(i)*T); %tone to be played
        fc = frequencies(i);
        stim = maketranstone(fc,4,m,bw,fs,stim_dur,risetime);
        %stim = bandpass(noise,frequencies(i)-fm,frequencies(i)+fm,fs);
        TestTone = true;
        Level_HL = Initial_Level;
        Correct_Levels = [];
        info = 'Press a button every time you hear a sound';
        Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
        info = 'regardles of  how distant or faint it sounds';
        Screen('DrawText',PS.window,info,textlocH,textlocV+line2line,PS.white);
        Screen('Flip',PS.window);
        WaitSecs(1.0);
        while TestTone
            
            if fp_check == 0 %this will randomly check for false positives when silence is played
                stim_nada = zeros(size(stim));
                PlayStim_AG(stim_nada,fs,PS,Level_SPL, useTDT, TypePhones, fc, ears(j),0); %dummy stim to check for fp
                if buttonBox
                    resp = getResponse(PS.RP, button_dur+stim_dur);
                else
                    resp = getResponseKb(button_dur+stim_dur); %#ok
                end
                if isempty(resp)
                    fprintf('CheckedFP: no fp \n')
                    fp_check = randi(10);
                else
                    fprintf('CheckedFP: FP!!! \n')
                    fp_check = randi(5);%check again sooner if subject has fp
                end
                FP_checks = [FP_checks ~isempty(resp)];%#ok
            end
            
            Level_SPL = Level_HL + HLtoSPL_offsets(i);
            PlayStim_AG(stim,fs,PS,Level_SPL, useTDT,TypePhones,fc,ears(j),1);
            if buttonBox
                resp = getResponse(PS.RP, button_dur+stim_dur);
            else
                resp = getResponseKb(button_dur+stim_dur); %#ok
            end
            fprintf(1, 'Level = %d, Resp =%d, Freq =%d Hz \n', Level_HL, resp,fc);
            
            pattern = circshift(pattern,1);
            if isempty(resp)
                Level_HL = Level_HL + 5;
                pattern(1) = 0; %no response
            else
                Correct_Levels = [Correct_Levels Level_HL]; %#ok
                Level_HL = Level_HL - 10;
                pattern(1) = 1;%response
            end
            if Level_HL >= 60
                Correct_Levels = 60; % if hearing is that bad no point in continuing testing
                break
            end
            
            jit = 2*rand; % creating this so sound isn't always played at same time
            WaitSecs(jit);
            if all(pattern == pattern_stop)
                TestTone = 0;
            end
            fp_check = fp_check - 1;
        end
        Thresholds(i) = Correct_Levels(end);

    end
    if j == 1
        fprintf(1, 'Done with first ear \n')
        PlayStim_AG(stim_nada,fs,PS,Level_SPL, useTDT, TypePhones, fc, ears(2),0); %dummy stim for transition click
        info = 'Done with first ear';
        Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
        info = strcat('When you are ready: Press any button twice to begin...');
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
    LR_Thresholds{j} = Thresholds; %#ok
    save(strcat(subj, '_Audiogram_',date,'.mat'),'frequencies','LR_Thresholds','FP_checks')
end


info = 'Done!';
Screen('DrawText',PS.window,info,textlocH,textlocV,PS.white);
Screen('Flip',PS.window);
save(strcat(subj, '_Audiogram_',date,'.mat'),'frequencies','LR_Thresholds','FP_checks')

fprintf(1,' \nFP_rate: %d%% \n ', round(100*(sum(FP_checks)/numel(FP_checks))))


if any(LR_Thresholds{1} > Exclude_Thresh) || any(LR_Thresholds{2} > Exclude_Thresh) || any(isnan(LR_Thresholds{1})) || any(isnan(LR_Thresholds{2}))
    fprintf(2,'!!!!!!! \n Subject has too high Threshold(s), Exclude Subject!!!!!!!! \n!!!!!!! \n')
else
    fprintf(1,' \nSubject has appropriate Thresholds! \n')
end

[f, f_ord] = sort(frequencies);
figure,semilogx(f, LR_Thresholds{1}(f_ord),'-bx',f,LR_Thresholds{2}(f_ord),'-ro',f,ones(size(f))*25,'--m'), set(gca,'Ydir','reverse')
ylim([-20 65]),xlim([100 9000]), title('Audiogram'),xlabel('Frequency (Hz)'), ylabel('Threhsolds (dB HL)'),xticks(f)
legend('Left Ear','Right Ear')

%Clearing I/O memory buffers:
invoke(PS.RP,'ZeroTag','datainL');
invoke(PS.RP,'ZeroTag','datainR');
pause(3.0);

close_play_circuit(PS.f1,PS.RP);
sca;









