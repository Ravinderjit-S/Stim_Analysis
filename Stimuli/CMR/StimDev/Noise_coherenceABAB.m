function  [Sig,mods] = Noise_coherenceABAB(noise_bands,mod_band,mod_bpfo,t_coh,t_incoh,fs)
%Order is t_incoh, t_coh, t_incoh, t_incoh
%Noise_bands = each row should be a band (low_f, high_f) and should go
%lowest band to highest band

Total_t = 2*t_coh + 2*t_incoh;
lowest_bound = min(min(noise_bands)); %extracting to make each bandpass filter equally sharp
bp_fo = round(1/lowest_bound *20 * fs); 
noise_bp = zeros(size(noise_bands,1),round(Total_t*fs));

for i = 1:size(noise_bands,1)
    noise = randn(1,round(fs*1.2*Total_t + bp_fo)); %making noise longer to deal with filter transient
    bp_filt = fir1(bp_fo, noise_bands(i,:) * 2/fs, 'bandpass');    
    noise_filt = filter(bp_filt,1,noise);
    noise_bp(i,:) = noise_filt(bp_fo+1:bp_fo+round(Total_t*fs));
end

%% make modulations
bp_filt_mod = fir1(mod_bpfo,mod_band * 2/fs,'bandpass');

mod_A1s = zeros(size(noise_bands,1),round(t_incoh*fs));
mod_A2s = zeros(size(noise_bands,1),round(t_incoh*fs));
for i = 1:size(noise_bands,1)
    noise_A1 = randn(1,round(fs*1.2*t_incoh+mod_bpfo));
    noise_A2 = randn(1,round(fs*1.2*t_incoh+mod_bpfo));
    noise_A1 = filter(bp_filt_mod,1,noise_A1);
    noise_A2 = filter(bp_filt_mod,1,noise_A2);
    
    mod_A1 = noise_A1(mod_bpfo+1:mod_bpfo + round(t_incoh*fs));
    mod_A2 = noise_A2(mod_bpfo+1:mod_bpfo + round(t_incoh*fs));
    
    mod_A1 = mod_A1 - min(mod_A1);
    mod_A1 = mod_A1 ./ max(mod_A1);
    mod_A2 = mod_A2 - min(mod_A2);
    mod_A2 = mod_A2 ./ max(mod_A2);
    
    mod_A1s(i,:) = mod_A1;
    mod_A2s(i,:) = mod_A2;
end

noise_B1 = randn(1,round(fs*1.2*t_coh+mod_bpfo));
noise_B2 = randn(1,round(fs*1.2*t_coh+mod_bpfo));

noise_B1 = filter(bp_filt_mod,1,noise_B1);
noise_B2 = filter(bp_filt_mod,1,noise_B2);

mod_B1 = noise_B1(mod_bpfo+1:mod_bpfo + round(t_coh*fs));
mod_B2 = noise_B2(mod_bpfo+1:mod_bpfo + round(t_coh*fs));

mod_B1 = mod_B1 - min(mod_B1);
mod_B1 = mod_B1 ./ max(mod_B1);
mod_B2 = mod_B2 - min(mod_B2);
mod_B2 = mod_B2 ./ max(mod_B2);

lp_co = mod_band(2) + 20 * log10(mod_band(2));
lp_fo = round(1/(mod_band(2)*1.5) * 3 * fs);
lp_filt = fir1(lp_fo, lp_co *2/fs,'low');

if lp_fo > size(noise_bp,2)
    error('Lp transient will be presetn')
end

mods = zeros(size(noise_bp));
noise_extra = randn(3,round(mod_bpfo + fs*0.2 + lp_fo));
noise_extra = filter(bp_filt_mod,1,noise_extra,[],2);
noise_extra = noise_extra(:,mod_bpfo+1:end); % extra noise for low pass filter transient

for i =1:size(noise_bands,1)
    n_extra = noise_extra(i,:) - min(noise_extra(i,:));
    n_extra = n_extra ./ max(n_extra);
    mod = horzcat(mod_A1s(i,:),mod_B1,mod_A2s(i,:),mod_B2);
    mod_lp = filter(lp_filt,1,[n_extra,mod]);
    mod = mod_lp(length(n_extra)+1:end);
    mods(i,:) = mod;
end
mods = mods - min(min(mods));
mods = mods /max(max(mods));

%% Make Sig
Sig = sum(mods .* noise_bp,1);








