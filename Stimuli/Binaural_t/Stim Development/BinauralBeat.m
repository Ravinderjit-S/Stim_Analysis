function [stim] = BinauralBeat(f1,f2,dur,fs)

nn = randn(2,dur*fs);
t = 0:1/fs:1-1/fs;
x1 = 0.5 * (sin(2*pi*f1*t - pi/2) + 1);
x2 = 0.5 * (sin(2*pi*f2*t - pi/2) + 1);

nn(1,:) = x1.*nn(1,:);
nn(2,:) = x2.*nn(2,:);


stim = nn;


