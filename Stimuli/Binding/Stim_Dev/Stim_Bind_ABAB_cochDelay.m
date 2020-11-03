function [stim, All_envs, ERBspace, Tones_f] = Stim_Bind_ABAB_cochDelay(Corr_inds, fs, f_start, f_end, Tones_num, ERB_spacing)
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
    
    cochDelays = [3.5, 3.15, 2.8, 2.55, 2.3,2.1, 1.85, 1.65, 1.5, 1.5, 1.5,1.4,1.25,1.25,1.2,1.2 ]/2; %ms
    %cochDelays = [12.5, 9.1, 7, 5.6, 4.75, 4.1, 3.75, 3.15, 3.1, 2.9, 2.77,2.5, 2.45, 2.25,2.15, 2];
 
    cochDelays = cochDelays / 1000; % ms -> sec
    
    for k = 1:length(Tones_f)
        f = Tones_f(k);
        sig_tone = sin(2*pi*f*T);
        ch_delay = cochDelays(k);
        
        env_A1 = create_envelope(bw,lpf,T_a+ch_delay,fs);
        env_A2 = create_envelope(bw,lpf,T_a+ch_delay,fs);
        
        if any(k==Corr_inds)
            env_B1 = corr_env1(1:end-round(ch_delay*fs));
            env_B2 = corr_env2(1:end-round(ch_delay*fs));
        else
            env_B1 = create_envelope(bw,lpf,T_a-ch_delay,fs);
            env_B2 = create_envelope(bw,lpf,T_a-ch_delay,fs);
        end

        env = [env_A1, env_B1, env_A2, env_B2];
        env = [env_A1, env, env_A1]; %added envA1 to beginning and end to deal with filter transients
        
        env = fftfilt(Lp_Filt,env);
        env = env(length(env_A1)+1:end-length(env_A1));
        if length(env) ~= length(sig_tone)
            env = env(2:end-1); %works for T_a = 1.0 ... this is essentially due to needing fractional samples and concatenating 4 supposedly 1 second (exactly) signals
        end
        stim = stim+sig_tone.*env;
        All_envs(:,k) = env;
    end
end
