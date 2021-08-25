function [Sig] = CMR_randMod_3AFC(noise_bands,target_f,SNRdB,n_mod_cuts,target_modf,fs,tlen,coh,bp_mod_fo)

t = 0:1/fs:tlen-1/fs;

if size(noise_bands,1) ~=3
    error('Currently expect 3 noise bands')
end


bp_filt_mod = fir1(bp_mod_fo, [n_mod_cuts(1) n_mod_cuts(2)]*2/fs,'bandpass');
bp_fo = round(1/(min(min(noise_bands(1)))) * 20 *fs);
noise_bp = zeros(3,length(t));



%% Frozen modulation in a single 3AFC trial

if coh ==1
    noise_mod = randn(1,1.5*length(t) + bp_mod_fo + 1);
    noise_mod = filter(bp_filt_mod,1,noise_mod);

    noise_mod = noise_mod(bp_mod_fo+1:bp_mod_fo+length(t));
    noise_mod = noise_mod - min(noise_mod);
    noise_mod = noise_mod / max(noise_mod);

else
    noise_mod = zeros(3,length(t));
    for i = 1:3
        noise_mod_i = randn(1,1.5*length(t) + bp_mod_fo + 1);
        noise_mod_i = filter(bp_filt_mod,1,noise_mod_i);

        noise_mod_i = noise_mod_i(bp_mod_fo+1:bp_mod_fo+length(t));
        noise_mod_i = noise_mod_i - min(noise_mod_i);
        noise_mod(i,:) = noise_mod_i / max(noise_mod_i);
    end
end



%% Make stim

for j=1:3
    for i =1:length(noise_bands)
        noise = randn(length(t)*1.5+bp_fo,1);
        lb = noise_bands(i,1);
        ub = noise_bands(i,2);

        bp_filt = fir1(bp_fo, [lb*2/fs ub*2/fs],'bandpass');
        noise_filt = filter(bp_filt,1,noise);
        noise_bp(i,:) = noise_filt(bp_fo+1:bp_fo+length(t));
    end
    

    noise_bp = noise_bp.*noise_mod;
    noise_on = noise_bp(2,:);
    noise_full = sum(noise_bp,1);
    
    Sig{j} = noise_full;
    
    if j == 3 %add target
        target = sin(2*pi*target_f.*t);
        target_mod = 0.5 + 0.5*sin(2*pi*target_modf.*t);
        target = target_mod .*target;
        
        desired_SNR = 10^(SNRdB/20);
        Cur_SNR = rms(target)/rms(noise_on);
        target_adjust = desired_SNR/Cur_SNR;
        target = target_adjust*target;
        Sig{j} = Sig{j} + target;
    end
end

end


        

    
    
