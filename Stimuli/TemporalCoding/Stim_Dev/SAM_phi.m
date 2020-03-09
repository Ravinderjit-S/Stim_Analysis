function [stim] = SAM_phi(f1,f2,fs,tlen,fm,phi_deg,dichotic)
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

rand_phase1 = rand() * 2*pi;
rand_phase2 = rand() * 2*pi;
rand_phase3 = rand() * 2*pi;
AM1 = 0.5 * (sin(2*pi*fm.*t + rand_phase1) + 1);
AM2 = 0.5 * (sin(2*pi*fm.*t + rand_phase2) + 1);
AM3 = 0.5 * (sin(2*pi*fm.*t + rand_phase3) + 1);
AM4 = 0.5 * (sin(2*pi*fm.*t + rand_phase3 + phi_rad) + 1);


sig1 = AM1 .* sin(2*pi*f1*t) + AM1 .* sin(2*pi*f2*t);
sig2 = AM2 .* sin(2*pi*f1*t) + AM2 .* sin(2*pi*f2*t);
sig3 = AM3 .* sin(2*pi*f1*t) + AM4 .* sin(2*pi*f2*t);

if dichotic
    sig1(1,:) = AM1 .* sin(2*pi*f1.*t);
    sig1(2,:) = AM1 .* sin(2*pi*f2.*t);
    sig2(1,:) = AM2 .* sin(2*pi*f1.*t);
    sig2(2,:) = AM2 .* sin(2*pi*f2.*t);
    sig3(1,:) = AM3 .* sin(2*pi*f1.*t);
    sig3(2,:) = AM4 .* sin(2*pi*f1.*t);
else
    sig1(2,:) = sig1;
    sig2(2,:) = sig2;
    sig3(3,:) = sig3;
end

stim = {sig1, sig2, sig3};


end