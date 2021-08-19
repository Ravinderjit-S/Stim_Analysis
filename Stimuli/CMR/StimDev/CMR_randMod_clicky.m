function [Sig] = CMR_randMod_clicky(noise_bands,target_f,SNRdB,n_mod_cuts,target_mod_f,fs,tlen,coh)
% This function will generate a CMR stimulus with 3 noise bands which have
% a random moudlation and a modulated tone embedded in the noise 
%noise_bands = 3 x 2 matrix providing the upper and lower cutoff of each
%               noise band
%target_f = Frequency of tone to be detected
%SNRdB = SNR of tone to the central noise band in dB
%n_mod_cuts = 1 x 2 matrix providing the upper and lower cutoff for the
%               modulation band on the noise
%target_mod_f = modulation to put on the tone
%fs = sampling rate
%tlen = length of stimulus in seconds
%coh = (boolean) wheter to return the coherent or incoherent condition


t = 0:1/fs:tlen-1/fs;

if size(noise_bands,1) ~=3
    error('Currently expect 3 noise bands')
end

%% Generate Noise Carriers
noise_bp = zeros(3,length(t));
for i =1:length(noise_bands)
    bw = diff(noise_bands(i,:));
    fc = mean(noise_bands(i,:));
    noise_bp(i,:) = makeNBNoiseFFT(bw,fc,tlen,fs,0,0);
end

%% Generate Noise modulations
bw = diff(n_mod_cuts);
fc = mean(n_mod_cuts);

if coh ==1
    [noise_mod, ~] = create_envelope(n_mod_cuts,40,tlen,fs);
    noise_mod = noise_mod';
else
    noise_mod = zeros(3,length(t));
    for i = 1:3
        [noise_mod_i, ~] = create_envelope(n_mod_cuts,40,tlen,fs);
        noise_mod(i,:) = noise_mod_i;
    end
end


noise_bp = noise_bp.*noise_mod;
noise_on = noise_bp(2,:);
noise_full = sum(noise_bp,1);

target = sin(2*pi*target_f.*t);
target_mod = 0.5 + 0.5*sin(2*pi*target_mod_f.*t);
target = target_mod .*target;

desired_SNR = 10^(SNRdB/20);
Cur_SNR = rms(target)/rms(noise_on);
target_adjust = desired_SNR/Cur_SNR;
target = target_adjust*target;
Sig = noise_full + target;


end