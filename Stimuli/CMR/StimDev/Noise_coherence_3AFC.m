function  [Sig] = Noise_coherence_3AFC(noise_bands,mod_band,mod_bpfo,t_len,fs)
%Order is t_incoh, t_coh, t_incoh, t_incoh
%Noise_bands = each row should be a band (low_f, high_f) and should go
%lowest band to highest band

bp_filt_mod = fir1(mod_bpfo,mod_band * 2/fs,'bandpass');

lowest_bound = min(min(noise_bands)); %extracting to make each bandpass filter equally sharp
bp_fo = round(1/lowest_bound *20 * fs); 
Sig = cell(1,3);

for j = 1:3

    noise_bp = zeros(size(noise_bands,1),round(t_len*fs));

    for i = 1:size(noise_bands,1)
        noise = randn(1,round(fs*1.2*t_len + bp_fo)); %making noise longer to deal with filter transient
        bp_filt = fir1(bp_fo, noise_bands(i,:) * 2/fs, 'bandpass');    
        noise_filt = filter(bp_filt,1,noise);
        noise_bp(i,:) = noise_filt(bp_fo+1:bp_fo+round(t_len*fs));
    end

    %% make modulations
    if j ==3
        noise = randn(1,round(fs*1.2*t_len+mod_bpfo));
        noise = filter(bp_filt_mod,1,noise);
        noise = noise(mod_bpfo+1:mod_bpfo + round(t_len*fs));
        noise = noise - min(noise);
        mod = noise / max(noise);
    else
        noise = randn(size(noise_bands,1),round(fs*1.2*t_len+mod_bpfo));
        noise = filter(bp_filt_mod,1,noise,[],2);

        noise = noise(:,mod_bpfo+1:mod_bpfo + round(t_len*fs));
        mod = zeros(size(noise_bands,1),round(t_len*fs));
        for i = 1:size(noise_bands,1)
            noise(i,:) = noise(i,:) - min(noise(i,:));
            mod(i,:) = noise(i,:) / max(noise(i,:));
        end
        
    end


    %% Make Sig
    Sig{j} = sum(noise_bp .* mod);
end







