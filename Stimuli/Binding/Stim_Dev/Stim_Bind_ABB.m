function [stimA, stimB1, stimB2, envs_A, envs_B1, envs_B2, ERBspace, Tones_f] = Stim_Bind_ABB(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing)
% IF ERB_spacing is given, that will be used instead of Tones_num
% This version of the binding stimulus will return the A and B parts seperatley

    T_a = 0.7; %Time of a part of the stimulus
    
    bw = [4 24]; %bandwidth of envelope
    lpf = 40; %low pass filter

    [Tones_f, ERBspace] = Get_Tones(Tones_num, ERB_spacing, f_start, f_end); %returns tone frequencies 
    T = 0:1/fs:T_a-1/fs;

    [corr_env1, ~] = create_envelope(bw,lpf,T_a,fs); 
    [corr_env2, ~] = create_envelope(bw,lpf,T_a,fs); 
    
    % initialize stim variables
    envs_A  = nan(length(Tones_f),length(T));
    envs_B1 = nan(length(Tones_f),length(T));
    envs_B2 = nan(length(Tones_f),length(T));
    stimA  = zeros(1,length(T));
    stimB1 = zeros(1,length(T));
    stimB2 = zeros(1,length(T));
    
    
    for k = 1:length(Tones_f)
        f = Tones_f(k);
        sig_tone = sin(2*pi*f*T);
        env_A = create_envelope(bw,lpf,T_a,fs);
        
        if any(k==Corr_inds)
            env_B1 = corr_env1;
            env_B2 = corr_env2;
        else
            env_B1 = create_envelope(bw,lpf,T_a,fs);
            env_B2 = create_envelope(bw,lpf,T_a,fs);
        end

        stimA  = stimA  + sig_tone .* env_A;
        stimB1 = stimB1 + sig_tone .* env_B1;
        stimB2 = stimB2 + sig_tone .* env_B2;

        envs_A(k,:)  = env_A;
        envs_B1(k,:) = env_B1;
        envs_B2(k,:) = env_B2;
    end
end
