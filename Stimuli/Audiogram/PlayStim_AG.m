function [] = PlayStim_AG(stim,fs,PS,L, useTDT, TypePhones,fc,ear,ScaleHere)
%%This function will play a Stim once
%stim = Stim to be played
%PS = psychstarter variable
%L = level to be played at
%StimText = text displayed on screen while stim is played
%TypePhones = 'headphones' or 'earphones'
%Stim_Trig = assign stimulus a trigger value the TDT will output (typically used for EEG)
%fc = should be frequency of tone played
%ScaleHere

%% AB: to show information about the current repetition on screen to the subject,
% and to get the subject's response to proceed the task

%     renderVisFrame(PS,'FIX'); % This shows a fixex dot while stimulus is being played
%     Screen('Flip',PS.window); % I think this is showing the dot -- ask Hari

%stim = scaleSound(rampsound(stim,fs,risetime)); %ramp stim

% if(useTDT)
%     %Clearing I/O memory buffers:
%     invoke(PS.RP,'ZeroTag','datainL');
%     invoke(PS.RP,'ZeroTag','datainR');
%     WaitSecs(3.0);
% end


if ScaleHere==1
    stim = scaleSound(stim);
end

scale = rms(stim);

switch TypePhones
    case 'headphones'
        sens = phoneSens(fc); % in dB SPL / 0 dBV (frequency specific)
        % Without attenuation, RZ6 gives 10.5236 dBV (matlab is restricted
        % to +/- 0.95 by scaleSound). So you would get sens + 10.5236 dB
        % SPL for pure tones occupying full range in MATLAB. To get a level
        % of 'L' dB SPL, you need to attenuate by sens + 10.5236 - L. This
        % is for a tone which would have an rms of 0.95/sqrt(2).
        % For a different waveform of rms 'scale', we should adjust further
        % by db(scale*sqrt(2)/0.95).
        
    case 'earphones'
        sens = phoneSens_ER2(fc);
    otherwise
        error('Incorrect input for TypePhones')
end

if any(strcmp(ear,{'r', 'R'}))
    stim = [zeros(size(stim));stim];
    attFlag = 1;
elseif any(strcmp(ear, {'l','L'}))
    stim = [stim; zeros(size(stim))];
    attFlag = 0;
else
    error('Ear input not understood')
end


digDrop = sens - 90; %Have digital drop in such a way that drop stays above 60 dB --> prevents clicks from changes in level
if fc == 8000 || fc == 250  % these freqeuncies have higher SPL thresholds so give less digital drop to keep drop above 60 dB
    digDrop = digDrop - 15;
end
if all(all(stim')) == 0
    drop = sens + 10.5236 - L - digDrop; %db(0) = -Inf so have this if else satement
else
    drop = sens + 10.5236 - L - digDrop + db(scale*sqrt(2)/0.95);
end
fprintf(1, 'Drop= %d \n', drop); 

if drop <0
    error('Drop is going negative')
end

%Start dropping from maximum RMS (actual RMS not peak-equivalent)
wavedata = stim * db2mag(-1 * digDrop); % AB: signal remains the same when digDrop = 0

if any(abs(wavedata(:)) >= 1)
    warning('Data is clipping.. what did you do?');
end
%-----------------------------------------
% Attenuate both sides, just in case


if attFlag == 1
    invoke(PS.RP, 'SetTagVal', 'attA', 120); %settting analog antenation L
    invoke(PS.RP, 'SetTagVal', 'attB', drop); %setting analog attenuation R
else
    invoke(PS.RP, 'SetTagVal', 'attA', drop); %settting analog antenation L
    invoke(PS.RP, 'SetTagVal', 'attB', 120); %setting analog attenuation R
end


if useTDT
    %Load data onto RZ6
    invoke(PS.RP, 'SetTagVal', 'nsamps', size(wavedata,2));
    invoke(PS.RP, 'WriteTagVEX', 'datainL', 0, 'F32', wavedata(1, :)); %Write to buffer left ear
    invoke(PS.RP, 'WriteTagVEX', 'datainR', 0, 'F32', wavedata(2, :)); %Write to buffer right ear
    WaitSecs(0.1);
    %Start playing from the buffer:
    invoke(PS.RP, 'SoftTrg', 1); %Playback trigger
end



end


