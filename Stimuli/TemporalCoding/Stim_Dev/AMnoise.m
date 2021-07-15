function [stim] = AMnoise(f, dur, fs)

t = 0:1/fs:dur-1/fs;
noise = randn(1,length(t));
AM = 0.5 + 0.5*sin(2*pi*f.*t-pi/2);

stim = AM .* noise;

stim = [stim;stim];






