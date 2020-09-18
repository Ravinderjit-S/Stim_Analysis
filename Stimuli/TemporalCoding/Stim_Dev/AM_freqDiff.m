clear

path = '../../CommonExperiment';
p = genpath(path);
addpath(p);

fs = 41000;
t = 0:1/fs:1-1/fs;

f1 = 50 ;
f2 = 55;

AM1 = 0.5 + 0.5 *sin(2*pi*f1.*t);
AM2 = 0.5 + 0.5 *sin(2*pi*f2.*t);

stim{1} = randn(1,fs) .* AM1;
stim{2} = randn(1,fs) .* AM1;
stim{3} = randn(1,fs) .* AM2;

order = randperm(3);
stim = stim(order);
for i = 1:length(stim)
    stim{i} = scaleSound(stim{i});
    soundsc(stim{i},fs)
    pause(1.5)
end


