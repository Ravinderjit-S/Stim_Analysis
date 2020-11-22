function [Sig] = CMR_randMod_giveNoiseMod(center_freqs,noise_half_bw,SNRdB,target_modf,fs,tlen,noise_mod)

t = 0:1/fs:tlen-1/fs;

if length(center_freqs) ~=3
    error('Currently expect center_freqs to be of size 3')
end


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
        

    
    