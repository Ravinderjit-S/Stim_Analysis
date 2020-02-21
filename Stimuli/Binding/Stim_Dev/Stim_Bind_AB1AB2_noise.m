function [stim, All_envs, ERBspace, Tones_f] = Stim_Bind_AB1AB2_noise(Corr_inds1, Corr_inds2, fs, f_start, f_end, Tones_num, ERB_spacing)
% IF ERB_spacing is given, that will be used instead of Tones_num
% This version of the binding stimulus will return one stim in ABAB format
% Corr_inds1 and Corr_inds2 can be different giving AB1AB2

    T_a = 1.0; %Time of a part of the stimulus
    noise_octaves = 0.1; %noise bandwidth will be this many octaves
    
    bw = [4 24]; %bandwidth of envelope
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
        sig_noise = randn(1,2*length(T)); %white noise double the necessary length
        bandwidth = f*2^noise_octaves-f;
        BPfilt = designfilt('bandpassfir', 'FilterOrder', fs/2,'CutoffFrequency1',f-bandwidth/2, ...
            'CutoffFrequency2',f+bandwidth/2,'SampleRate',fs);
        sig_filt = filter(BPfilt,sig_noise);
        sig_filt = sig_filt(round(length(T)/2)+1:round(length(T)/2) + length(T));
        
        env_A1 = create_envelope(bw,lpf,T_a,fs);
        env_A2 = create_envelope(bw,lpf,T_a,fs);
        
        if any(k==Corr_inds1)
            env_B1 = corr_env1;
        else
            env_B1 = create_envelope(bw,lpf,T_a,fs);
        end

        if any(k==Corr_inds2)
            env_B2 = corr_env2;
        else
            env_B2 = create_envelope(bw,lpf,T_a,fs);
        end
        
        
        env = [env_A1, env_B1, env_A2, env_B2];
        env = [env_A1, env, env_A1]; %added envA1 to beginning and end to deal with filter transients
        
        env = fftfilt(Lp_Filt,env);
        env = env(length(env_A1)+1:end-length(env_A1));
        if length(env) ~= length(sig_filt)
            env = env(2:end-1); %works for T_a = 1.0 ... this is essentially due to needing fractional samples and concatenating 4 supposedly 1 second (exactly) signals
        end
        stim = stim+sig_filt.*env;
        All_envs(:,k) = env;
    end
end
