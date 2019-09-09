clear all
fs = 48828.125;
% [Da, ~] = audioread('DAbase_48828.wav');
BPfilt = designfilt('bandpassiir', 'StopbandFrequency1', 190, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 1550, 'StopbandAttenuation1', 30, 'PassbandRipple', 1, 'StopbandAttenuation2', 30, 'SampleRate', fs);
%load('DAbase_48828_filtered200.1500.mat'); %loads Da

t = (0:1/fs:1.4)';
Da = 0;
for harmonic = 300:100:1400
    Da = Da+sin(2*pi*harmonic*t);
end
Da = Da *0.1 / rms(Da);


fm = 2;
SNRdes = 5; %dB
tic()
%[stim, noiseRMS, total_dur] = IACt_DaDetect(Da, fs, BPfilt,fm,SNRdes,[]);
%[stim, DaRMS, ~] = IACt_NoiseDetect(Da,fs,BPfilt,fm,SNRdes,1);
%[stim, noiseRMS,total_dur] = ITDt_ToneDetect(1, fs, BPfilt,fm,SNRdes,[]);
[stim, noiseRMS,total_dur] = IACt_ToneDetect(fs, BPfilt,fm,SNRdes,[]);
toc()
a = [1 2 3];

for j = 3
    play = stim{a(j)};
    play(1,:) = rampsound(play(1,:),fs,.050)./noiseRMS(1);
    play(2,:) = rampsound(play(2,:),fs,.050)./noiseRMS(2);
    play = horzcat(play, zeros(2,round(fs/2)));
    sound(play/5,fs)
    pause(2.0)
end


t = 0:1/fs:length(stim{1}(1,:))/fs-1/fs;
figure,plot(t,stim{3}')
