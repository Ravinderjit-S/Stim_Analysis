function [stim, All_envs, ERBspace, Tones_f] = Stim_Bind_ABAB(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing)
% IF ERB_spacing is given, that will be used instead of Tones_num
% This version of the binding stimulus will return one stim in ABAB format

    T_a = 1.0; %Time of a part of the stimulus
    
    bw = [4 24]; %bandwidth of envelope 4-24
    lpf = 40; %low pass filter

    [Tones_f, ERBspace] = Get_Tones(Tones_num, ERB_spacing, f_start, f_end); %returns tone frequencies 
    T = 0:1/fs:4*T_a-1/fs;
    T = T(1:end-2);

    [corr_env1, Lp_Filt] = create_envelope(bw,lpf,T_a,fs); 
    [corr_env2, ~] = create_envelope(bw,lpf,T_a,fs); 
    
    % initialize stim variables

    stim = zeros(1,length(T));
    
    
    for k = 1:length(Tones_f)
        f = Tones_f(k);
        sig_tone = sin(2*pi*f*T);
        
        env_A1 = create_envelope(bw,lpf,T_a,fs);
        env_A2 = create_envelope(bw,lpf,T_a,fs);
        
        if any(k==Corr_inds)
            env_B1 = corr_env1;
            env_B2 = corr_env2;
        else
            env_B1 = create_envelope(bw,lpf,T_a,fs);
            env_B2 = create_envelope(bw,lpf,T_a,fs);
        end

        env = [env_A1; env_B1; env_A2; env_B2];
        env = filter(Lp_Filt,1,env,[],1);
        
        if length(env) ~= length(sig_tone)
            env = env(2:end-1); %works for T_a = 1.0 ... this is essentially due to needing fractional samples and concatenating 4 supposedly 1 second (exactly) signals
        end

        stim = stim+sig_tone.*env';
        All_envs(:,k) = env;
    end
end
