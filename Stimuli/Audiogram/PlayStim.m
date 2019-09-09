function [] = PlayStim_AG(stim,fs,risetime,PS,L, useTDT, StimText, Stim_Trig, TypePhones,fc)
%%This function will play a Stim once
%stim = Stim to be played
%PS = psychstarter variable
%L = level to be played at
%StimText = text displayed on screen while stim is played 
%TypePhones = 'headphones' or 'earphones'
%Stim_Trig = assign stimulus a trigger value the TDT will output (typically used for EEG) 
%fc = should be frequency of tone played

 %% AB: to show information about the current repetition on screen to the subject, 
    % and to get the subject's response to proceed the task 

%     renderVisFrame(PS,'FIX'); % This shows a fixex dot while stimulus is being played
%     Screen('Flip',PS.window); % I think this is showing the dot -- ask Hari 
    
    stim = rampsound(stim,fs,risetime); %ramp stim
    scale = rms(stim);
    %fc = 1000; %account for inverse filter 
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

        stim = vertcat(stim,stim);
   

            digDrop = 0; % How much to drop digitally
            drop = sens + 10.5236 - L - digDrop + db(scale*sqrt(2)/0.95);
            %Start dropping from maximum RMS (actual RMS not peak-equivalent)
            wavedata = stim * db2mag(-1 * digDrop); % AB: signal remains the same when digDrop = 0
            %-----------------------------------------
            % Attenuate both sides, just in case
            invoke(PS.RP, 'SetTagVal', 'attA', drop); %settting analog antenation L
            invoke(PS.RP, 'SetTagVal', 'attB', drop); %setting analog attenuation R


            if useTDT  
                %Load data onto RZ6
                invoke(PS.RP, 'SetTagVal', 'nsamps', size(wavedata,2));
                invoke(PS.RP, 'WriteTagVEX', 'datainL', 0, 'F32', wavedata(1, :)); %Write to buffer left ear
                invoke(PS.RP, 'WriteTagVEX', 'datainR', 0, 'F32', wavedata(2, :)); %Write to buffer right ear
                WaitSecs(0.1);
                %Start playing from the buffer:
                [nx, ny, ~] = DrawFormattedText(PS.window, StimText,'center','center');
                %Screen('DrawText',PS.window,StimText,nx,ny,PS.white);
                Screen('Flip',PS.window);
                if ~isempty(Stim_Trig)
                    invoke(PS.RP, 'SetTagVal', 'trgname',Stim_Trig);
                end
                WaitSecs(0.3); %Let the subject see text briefly before hearing sound
                invoke(PS.RP, 'SoftTrg', 1); %Playback trigger
            end

    
    
end


