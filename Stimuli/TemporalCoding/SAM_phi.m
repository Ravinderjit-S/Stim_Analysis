function [stim] = SAM_phi(f1,f2,fs,tlen,fm,phi_deg,diotic)
%This fucntion will return a 3 AFC stim where the 3rd stimulus has an AM
%with a phase difference 
%f1 = frequency of tone 1
%f2 = frequency of tone 2
%fs = sampling rate
%tlen = length of stim in seconds
%fm = SAM rate 
%phi_deg = phase difference between two AMs in degrees
%diotic = send two different AM signals to different ears

phi_rad = (phi_deg/360) * 2*pi;
t = 0:1/fs:tlen-1/fs;

AM1 = 0.5 * (sin(2*pi*fm.*t) + 1);
AM2 = 0.5 * (sin(2*pi*fm.*t + phi_rad) + 1);

sig1 = AM1 .* sin(2*pi*f1*t) + AM1 .* sin(2*pi*f2*t);
sig2 = AM1 .* sin(2*pi*f1*t) + AM2 .* sin(2*pi*f2*t);

if diotic
    sig1(1,:) = AM1 .* sin(2*pi*f1*t);
    sig1(2,:) = AM1 .* sin(2*pi*f2*t);
    sig2(1,:) = AM1 .* sin(2*pi*f1*t);
    sig2(2,:) = AM2 .* sin(2*pi*f2*t);
else
    sig1(2,:) = sig1;
    sig2(2,:) = sig2;
end

stim = {sig1, sig1, sig2};