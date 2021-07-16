function [stim_Ref, stimA, stimB1, stimA2, ERBspace, Tones_f, envs_A, envs_B1] = StimBind_noRect_3AFC(Corr_inds, fs,bw)
% IF ERB_spacing is given, that will be used instead of Tones_num
% This version of the binding stimulus will return the A and B parts
% seperatley for a 3AFC experiment. No half wave rectification is done on
% the envelope applied to the tones so the envelopes are less clicky but
% this also prevents addition of low frequency energy from the half-wave
% rectification

    T_a = 1.0; %Time of a part of the stimulus
    
    %bw = [4 24]; %bandwidth of envelope
    %lpf = 40; %low pass filter

    [Tones_f, ERBspace] = Get_Tones(20, [], 200, 8000); %returns tone frequencies 
    T = 0:1/fs:T_a-1/fs;
    
    [corr_env1] = create_envelope_ifft(bw,T_a,fs); 
    
    % initialize stim variables
    envs_A  = nan(length(Tones_f),length(T));
    envs_B1 = nan(length(Tones_f),length(T));
%     envs_A2 = nan(length(Tones_f),length(T));
    stimA  = zeros(1,length(T));
    stimB1 = zeros(1,length(T));
    stimA2 = zeros(1,length(T));
    stim_Ref = zeros(1,length(T));
    
    
    for k = 1:length(Tones_f)
        f = Tones_f(k);
        sig_tone = sin(2*pi*f*T+rand()*2*pi);
        env_A = create_envelope_ifft(bw,T_a,fs); 
        env_A2 = create_envelope_ifft(bw,T_a,fs); 
        env_Ref = create_envelope_ifft(bw,T_a,fs); 
        
        if any(k==Corr_inds)
            env_B1 = corr_env1;
        else
            env_B1 =  create_envelope_ifft(bw,T_a,fs); 
        end

        stim_Ref = stim_Ref + sig_tone .* env_Ref;
        stimA  = stimA  + sig_tone .* env_A;
        stimB1 = stimB1 + sig_tone .* env_B1;
        stimA2 = stimA2 + sig_tone .* env_A2;
 
        envs_A(k,:)  = env_A;
        envs_B1(k,:) = env_B1;
%         envs_A2(k,:) = env_A2;
    end
end
