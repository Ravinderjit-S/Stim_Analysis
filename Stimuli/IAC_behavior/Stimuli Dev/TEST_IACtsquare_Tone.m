clear all
fs = 48828.125;
BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);
windowSize = .1;
SNRdes = 6; %dB
tic()

[stim, noiseRMS,total_dur] = IACtSquare_ToneDetect(fs, BPfilt, windowSize,SNRdes, []);
toc()
a = [1 2 3];

for j = 3
    play = stim{a(j)};
    play(1,:) = rampsound(play(1,:),fs,.050)./noiseRMS(1);
    play(2,:) = rampsound(play(2,:),fs,.050)./noiseRMS(2);
    play = horzcat(play, zeros(2,round(fs/2)));
    sound(play/10,fs)
    pause(2.0)
end


t = 0:1/fs:length(stim{1}(1,:))/fs-1/fs;
figure,plot(t,stim{3}')
