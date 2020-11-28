function [Sig] = CMR_randMod_giveNoiseMod(noise_bands,target_f,SNRdB,n_mod_cuts,target_modf,fs,tlen,noise_mod)

t = 0:1/fs:tlen-1/fs;

if size(noise_bands,1) ~=3
    error('Currently expect center_freqs to be of size 3')
end


bp_fo = round(1/(min(min(noise_bands(1)))) * 20 *fs);
noise_bp = zeros(3,length(t));

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
        

    
    