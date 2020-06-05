function [plv, f] = PLV_nerve(spikes,stim,fs,TW)
%% Calculate PLV from spike data
%Sooo, how should I do this ....
%Gonna start with treating each spike like a delta function and see how
%well this train of deltas phase lock to a stimulus
%fs = sampling rate of stimulus
%spikes = in Nel format
%stim = vertical vector
%TW = half time bandwidth product 

%% Make Spike Delta waveform (single trial PSTH with 1/fs blocks)

Trials = max(spikes(:,1));
spk_delta = zeros(length(stim),Trials);
for k =1:Trials
    spk_times = spikes(spikes(:,1) ==k,2);
    spk_times(spk_times >length(stim)/fs) = []; % ignore spikes when stim off
    spk_delta(round(spk_times*fs),k) = 1;
end

%% Calc PLV
ntaps = 2*TW-1;
taps = dpss(length(stim),TW,ntaps);
nfft = 2^ceil(log2(length(stim)));
plv_taps = zeros(ntaps,nfft);

for m = 1:ntaps
    fprintf(1, 'Taper %d \n', m)
    Xf = fft(spk_delta.*taps(:,m)',nfft,2);
    Yf = fft(stim.*taps(:,m)',nfft,2);
    XYf = Xf .* conj(Yf);
    plv_taps(m,:) = abs(mean(XYf ./ abs(Xmf),1));
end

plv = mean(plv_taps,1);
f = 0:fs/nfft:fs-1/nfft; 









