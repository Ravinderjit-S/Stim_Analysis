clear all
dur = 1;
fs = 48828.125;
%BPfilt = designfilt('bandpassiir', 'StopbandFrequency1', 190, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 1550, 'StopbandAttenuation1', 30, 'PassbandRipple', 1, 'StopbandAttenuation2', 30, 'SampleRate', fs);
% BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);
load('BPfilt.mat')


depth = 0.5;
randDepth = 1;
randIAC = 1;
fm = 40;

tic()
%stim = AMdetect(dur,fs,depth,fm,BPfilt,IAC);
[stim] = IACsin_AM_AFC3(dur, fs, depth, fm, BPfilt, randDepth,randIAC);
% figure,plot(stim{2}')
toc()
a = randperm(3); a = [1 2 3];
for i =1:2
    play = stim{a(i)};
    play(1,:) = rampsound(play(1,:),fs,.050); 
    play(2,:) = rampsound(play(2,:),fs,.050);
%     normalizeSound = (rms(play(1,:)) + rms(play(2,:))) ./ 2;
%     play(1,:) = play(1,:) ./ normalizeSound; play(2,:) = play(2,:) ./ normalizeSound;
    
    sound(scaleSound([play,zeros(2,round(fs/2))]),fs)
    
    pause(2)
end

