function [Sig] = CMRrandMod_mseqTargetMod(noise_bands,target_f,SNRdB,n_mod_cuts,fs,coh,mseq,Point_len)
%USAGE:
%  noise_bands: 3x2 vector containng lower and upper bounds for noise bands
%  target_f: target tone frequnecy to detect



%  mseq: The Msequence to modulate with of appropriate duration
%  Point_len: The number of data points per Mseq point in the mseq



tlen = length(mseq)/fs;
t = 0:1/fs:tlen-1/fs;

if size(noise_bands,1) ~=3
    error('Currently expect 3 noise bands')
end

%% Generate Noise
noise_bp = zeros(3,length(t));
for i =1:length(noise_bands)
    lb = noise_bands(i,1);
    ub = noise_bands(i,2);
    noise_bp(i,:) = makeNBNoiseFFT(ub-lb, (lb+ub)/2, tlen,fs,0,0);
end

%% Generate Modulations
if coh ==1
    noise_mod = makeNBNoiseFFT(n_mod_cuts(2)-n_mod_cuts(1), mean(n_mod_cuts),tlen,fs,0,0)';
    noise_mod = noise_mod - min(noise_mod);
    noise_mod = noise_mod / max(noise_mod);
else
    noise_mod = zeros(3,length(t));
    for i = 1:3
        noise_mod_i = makeNBNoiseFFT(n_mod_cuts(2)-n_mod_cuts(1), mean(n_mod_cuts),tlen,fs,0,0);
        noise_mod_i = noise_mod_i - min(noise_mod_i);
        noise_mod(i,:) = noise_mod_i / max(noise_mod_i);
    end
end

%% Add noise and give it modulation
noise_bp = noise_bp.*noise_mod;
noise_on = noise_bp(2,:);
noise_full = sum(noise_bp,1);

%% Generate Mseq Modulation
mseqAM = (mseq+1) /2;
w = dpss(Point_len,1,1);
w = w-w(1); w = w/max(w); %dpss ramp so energy in modulation doesn't spread out far beyond upperF
mseqAM = conv(mseqAM,w,'same');
mseqAM = mseqAM / max(mseqAM);

%% Give Target Tone Mseq Modulation
target = sin(2*pi*target_f.*t);
target = mseqAM .*target;

%% Set appropriate SNR
desired_SNR = 10^(SNRdB/20);
Cur_SNR = rms(target)/rms(noise_on);
target_adjust = desired_SNR/Cur_SNR;
target = target_adjust*target;
Sig = noise_full + target;


end