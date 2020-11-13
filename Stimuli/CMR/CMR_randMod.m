function [Sig] = CMR_randMod(center_freqs,noise_half_bw,SNRdB,n_mod_cuts,target_modf,fs,tlen,coh,bp_mod_fo)

t = 0:1/fs:tlen-1/fs;

if length(center_freqs) ~=3
    error('Currently expect center_freqs to be of size 3')
end


bp_filt_mod = fir1(bp_mod_fo, [n_mod_cuts(1) n_mod_cuts(2)]*2/fs,'bandpass');

for j=1:3
    for i =1:length(center_freqs)
        noise = randn(length(t)*3,1);
        lb = center_freqs(i)-noise_half_bw;
        ub = center_freqs(i)+noise_half_bw;
        bp_fo = round(1/(center_freqs(1)-noise_half_bw) * 20 *fs);
        if bp_fo > length(t)
            error('Filter order too large for singal length')
        end
        bp_filt = fir1(bp_fo, [lb*2/fs ub*2/fs],'bandpass');
        noise_filt = filter(bp_filt,1,noise);
        noise_bp(i,:) = noise_filt(length(t)+1:2*length(t));
    end
    
    if coh ==1
        noise_mod = randn(1,1.5*length(t) + bp_mod_fo + 1);
        noise_mod = filter(bp_filt_mod,1,noise_mod);
        
        noise_mod = noise_mod(bp_mod_fo+1:bp_mod_fo+length(t));
        noise_mod = noise_mod - min(noise_mod);
        noise_mod = noise_mod / max(noise_mod);
        
    else
        for i = 1:3
            noise_mod_i = randn(1,1.5*length(t) + bp_mod_fo + 1);
            noise_mod_i = filter(bp_filt_mod,1,noise_mod_i);

            noise_mod_i = noise_mod_i(bp_mod_fo+1:bp_mod_fo+length(t));
            noise_mod_i = noise_mod_i - min(noise_mod_i);
            noise_mod(i,:) = noise_mod_i / max(noise_mod_i);
        end
    end
    noise_bp = noise_bp.*noise_mod;
    noise_on = noise_bp(2,:);
    noise_full = sum(noise_bp,1);
    
    Sig{j} = noise_full;
    
    if j == 3 %add target
        target = sin(2*pi*center_freqs(2).*t);
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
        

    
    