function [] = PlayStim(stim,fs,risetime,PS,L, useTDT, StimText, Stim_Trig, TypePhones)
%%This function will play a Stim once
%stim = Stim to be played, [2 x nsamps]
%fs = sampling rate
%risetime = risetime for stimulus 
%PS = psychstarter variable
%L = level to be played at in dB SPL
%StimText = text displayed on screen while stim is played ... if 'NONE', nothing is shown
%TypePhones = 'headphones' or 'earphones'
%Stim_Trig = assign stimulus a trigger value the TDT will output (typically used for EEG), leave empty if not using 

%%

stim = scaleSound(stim);
if risetime ~=0
    stim(1,:) = rampsound(stim(1,:),fs,risetime); %ramp stim
    stim(2,:) = rampsound(stim(2,:),fs,risetime);
end
scale = rms(stim(1,:)); 
fc = 1000; %account for inverse filter 
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

digDrop = 0; % How much to drop digitally
drop = sens + 10.5236 - L - digDrop + db(scale*sqrt(2)/0.95);
if drop < 0
    error('Drop is negative')
end
%Start dropping from maximum RMS (actual RMS not peak-equivalent)
wavedata = stim * db2mag(-1 * digDrop); % AB: signal remains the same when digDrop = 0
%-----------------------------------------
% Attenuate both sides, just in case
invoke(PS.RP, 'SetTagVal', 'attA', drop); %settting analog antenation L
invoke(PS.RP, 'SetTagVal', 'attB', drop); %setting analog attenuation R

if drop <0
    error('Drop is going negative')
end
if any(abs(wavedata(:)) >= 1)
    warning('Data is clipping');
end

if useTDT 
    %Load data onto RZ6
    invoke(PS.RP, 'SetTagVal', 'nsamps', size(wavedata,2));
    invoke(PS.RP, 'WriteTagVEX', 'datainL', 0, 'F32', wavedata(1, :)); %Write to buffer left ear
    invoke(PS.RP, 'WriteTagVEX', 'datainR', 0, 'F32', wavedata(2, :)); %Write to buffer right ear
    WaitSecs(0.1);
    if ~strcmp(StimText,'NONE')
        [nx, ny, ~] = DrawFormattedText(PS.window, StimText,'center','center');
        %Screen('DrawText',PS.window,StimText,nx,ny,PS.white);
        Screen('Flip',PS.window);
    end
    if ~isempty(Stim_Trig)
        invoke(PS.RP, 'SetTagVal', 'trgname',Stim_Trig);
    end
    invoke(PS.RP, 'SoftTrg', 1); %Playback trigger
end


    
end


