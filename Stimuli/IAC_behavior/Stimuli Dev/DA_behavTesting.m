clear all
fs = 48828.125;
BPfilt = designfilt('bandpassiir', 'StopbandFrequency1', 190, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 1550, 'StopbandAttenuation1', 30, 'PassbandRipple', 1, 'StopbandAttenuation2', 30, 'SampleRate', fs);

[Da, ~] = audioread('DAbase_48828.wav');

Da_len = length(Da);
Da = filtfilt(BPfilt, vertcat(zeros(size(Da)),Da,zeros(size(Da))));
Da = Da(Da_len+1:2*Da_len);
Da = rampsound(Da,fs,0.010);

Da = vertcat(Da,Da,Da,Da); %4 DAs to have it long enough to have the slowest dynamics (2Hz) capture in it
Da = vertcat(Da',-Da');
Da_dur = length(Da)/fs;

Pre_Post_dur = 0.2;
total_dur = Pre_Post_dur*2 + Da_dur;

nbn = randn(2, round(total_dur*fs));

fm = 2;
t = 0:1/fs:total_dur-1/fs;
A = cos(2*pi*fm.*t);
B = sqrt(1-A.^2);
nbn(2,:) = A.*nbn(1,:) + B.*nbn(2,:);

lenNBN = size(nbn,2);

nbn = filtfilt(BPfilt, vertcat(zeros(size(nbn')), nbn', zeros(size(nbn'))));
nbn = nbn(lenNBN+1:2*lenNBN,:)';
nbn(1,:) = nbn(1,:)./rms(nbn(1,:)); nbn(2,:) = nbn(2,:)./rms(nbn(2,:));

Da_sil = horzcat(zeros(2,floor(Pre_Post_dur*fs)),Da,zeros(2,ceil(Pre_Post_dur*fs)));

Da_on_index = floor(Pre_Post_dur*fs)+1:length(Da_sil)-ceil(Pre_Post_dur*fs);
SNRdes = -7;
Cur_SNR = snr(Da_sil(:,Da_on_index), nbn(:,Da_on_index));

alpha1 = 10^(SNRdes/20)*rms(nbn(1,Da_on_index))./rms(Da_sil(1,Da_on_index));
alpha2 = 10^(SNRdes/20)*rms(nbn(2,Da_on_index))./rms(Da_sil(2,Da_on_index));
Da_sil(1,:) = alpha1.*Da_sil(1,:); Da_sil(2,:) = alpha2.*Da_sil(2,:);

New_SNR = snr(Da_sil(:,Da_on_index),nbn(:,Da_on_index));

ramp = 0.010; 
nbn(1,:) = rampsound(nbn(1,:),fs,ramp);nbn(2,:) = rampsound(nbn(2,:),fs,ramp);


stim = Da_sil + nbn;

stim = stim./10;

sound(stim,fs)

