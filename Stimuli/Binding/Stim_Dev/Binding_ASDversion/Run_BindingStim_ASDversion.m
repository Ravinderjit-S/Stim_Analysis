clear all; close all hidden; clc; %#ok<CLALL>

try
    % Setting random generator seed and state
    load('s.mat');
    rng(s);
    
    fig_num=99;
    USB_ch=1;
    
    
    IAC = -1;
    FS_tag = 3;
    
    Fs = 48828.125;
    
    
    [f1RZ,RZ,FS]=load_play_circuit(FS_tag,fig_num,USB_ch,0,IAC);
    
    % Experiment parameters
    ntrials = 500;
    level = 83; % Target level (each tone will be 13 dB lower than overall)
    ntones = 20;
    ngroup = 18;
    dur = 1.0; % Times four for four sections in the sequence
    seq = [1, 2, 1, 2];
    ecorr = 1.0; %rho: envelope correlation
    isi = 1.0;
    jitterrange = 0.2;
    isi_adj = isi - jitterrange/2;
    
    % Pause Off
    invoke(RZ, 'SetTagVal', 'trigval',253);
    invoke(RZ, 'SoftTrg', 6);
    
    WaitSecs(2.0);
    
    
    
    % Using jitter to make sure that background noise averages out across
    % trials. We use jitter that is random between 0 and 200ms (0.2s). So
    % average duration added by the jitter is 100 ms (0.1s).
    jitlist = rand(ntrials, 1)*jitterrange;
    
    tstart = tic;
    for j = 1:ntrials
        
        [y, ~] = texture_ERB(ntones, ngroup, ecorr, seq, Fs, dur, 0);
        stimrms = rms(y); % For later use in calibration
        
        
        chanL = y;
        chanR = y;
        
        stimTrigger = 1;
        
        jit = jitlist(j);
        
        stimlength = numel(y);
        
        %-----------------
        % Why 111 for ER-2?
        %-----------------
        % ER-2s give about 100dB SPL for a 1kHz tone with a 1V-rms drive.
        % Max output is +/-5V peak i.e 3.54V-rms which is 11 dB higher.
        % Thus 111 dB-SPL is the sound level for tones when they occupy full
        % range.
        
        % Full range in MATLAB for a pure tone is +/- 1 which is 0.707 rms and
        % that corresponds to 111 dB SPL at the end. So if we want a signal
        % with rms sigrms to be x dB, then (111 - x) should be
        % db(sigrms/0.707).
        
        
        dropL = 111 - level + 3 + db(stimrms); % The extra 3 for the 0.707 factor
        
        dropR = 111 - level + 3 + db(stimrms);
        
        
        invoke(RZ, 'SetTagVal', 'trigval', stimTrigger);
        invoke(RZ, 'SetTagVal', 'nsamps', stimlength);
        invoke(RZ, 'WriteTagVEX', 'datainL', 0, 'F32', chanL); %write to buffer left ear
        invoke(RZ, 'WriteTagVEX', 'datainR', 0, 'F32', chanR); %write to buffer right ear
        
        invoke(RZ, 'SetTagVal', 'attA', dropL); %setting analog attenuation c
        invoke(RZ, 'SetTagVal', 'attB', dropR); %setting analog attenuation R
        
        WaitSecs(0.05); % Just giving time for data to be written into buffer
        %Start playing from the buffer:
        invoke(RZ, 'SoftTrg', 1); %Playback trigger
        fprintf(1,' Trial Number %d/%d\n', j, ntrials);
        WaitSecs(dur*numel(seq) + isi_adj + jit);
    end
    
    toc(tstart);
    %Clearing I/O memory buffers:
    invoke(RZ,'ZeroTag','datainL');
    invoke(RZ,'ZeroTag','datainR');
    WaitSecs(3.0);
    
    % Pause On
    invoke(RZ, 'SetTagVal', 'trigval', 254);
    invoke(RZ, 'SoftTrg', 6);
    
    close_play_circuit(f1RZ,RZ);
    fprintf(1,'\n Done with data collection!\n');
    
catch me
    close_play_circuit(f1RZ,RZ);
    rethrow(me);
end

