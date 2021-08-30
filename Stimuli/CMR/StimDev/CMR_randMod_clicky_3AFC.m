function [Sig, answer] = CMR_randMod_clicky_3AFC(noise_bands,target_f,SNRdB,n_mod_cuts,target_mod_f,fs,tlen,coh,risetime,silence_period)
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
%risetime = ramp applied to each stim n 3AFC in seconds
%silence_period = silence inbetween 3AFC stims in seconds


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

t_target = 0:1/fs:(tlen-0.1)-1/fs; %start and end tone 50 ms after onset and before noise turns off
target = sin(2*pi*target_f.*t_target);
target_mod = 0.5 + 0.5*sin(2*pi*target_mod_f.*t_target);
target = target_mod .*target;
target = rampsound(target,fs,risetime);

tone_start = round(.050*fs);
tone_end = tone_start + length(target);

noise_full = rampsound(noise_full,fs,risetime);

desired_SNR = 10^(SNRdB/20);
Cur_SNR = rms(target)/rms(noise_on(tone_start:tone_end));
target_adjust = desired_SNR/Cur_SNR;
target = target_adjust*target;
target = horzcat(zeros(1,tone_start), target, zeros(1,length(noise_full) - tone_end));
Sig_target = noise_full + target;

silence = zeros(1,round(fs*silence_period));

Sigs = {noise_full, noise_full, Sig_target};
order = randperm(3);
Sigs = Sigs(order);

Sig = horzcat(Sigs{1}, silence, Sigs{2}, silence, Sigs{3});
Sig = [Sig;Sig];
answer = find(order==3);


end

function y = rampsound(x,fs,risetime)
% Function to ramp a sound file using a dpss ramp
% USAGE:
% y = rampsound(x,fs,risetime)
%
% risetime in seconds, fs in Hz
% Hari Bharadwaj

    Nramp = ceil(fs*risetime*2)+1;
    w = dpss(Nramp,1,1);

    w = w - w(1);
    w = w/max(w);
    sz = size(x);
    half = ceil(Nramp/2);
    wbig = [w(1:half); ones(numel(x)- 2*half,1); w((end-half+1):end)];

    if(sz(1)== numel(x))
        y = x.*wbig;
    else
        y = x.*wbig';
    end

end


