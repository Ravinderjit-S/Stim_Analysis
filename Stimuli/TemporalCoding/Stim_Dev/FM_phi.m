function [stim] = FM_phi(f1,f2,fs,tlen,fm,phi_deg,dichotic)
%This fucntion will return a 3 AFC stim where the 3rd stimulus has an FM
%with a phase difference 
%f1 = frequency of tone 1
%f2 = frequency of tone 2
%fs = sampling rate
%tlen = length of stim in seconds
%fm = FM rate
%phi_deg = phase difference between two FMs in degrees

dF1 = f1 * 0.05;
dF2 = f2 * 0.05;

phi_rad = (phi_deg/360) * 2*pi;
t = 0:1/fs:tlen-1/fs;

rand_phase1 = rand() * 2*pi;
rand_phase2 = rand() * 2*pi;
rand_phase3 = rand() * 2*pi;

phase1 = sin(2*pi*fm.*t + rand_phase1);
phase2 = sin(2*pi*fm.*t + rand_phase2);
phase3 = sin(2*pi*fm.*t + rand_phase3);
phase4 = sin(2*pi*fm.*t + rand_phase3 + phi_rad);

mod_f1 = dF1/fm;
mod_f2 = dF2/fm;

x1 = sin(2*pi*f1.*t + mod_f1 * phase1);
x2 = sin(2*pi*f2.*t + mod_f2 * phase1);
x3 = sin(2*pi*f1.*t + mod_f1 * phase2);
x4 = sin(2*pi*f2.*t + mod_f2 * phase2);
x5 = sin(2*pi*f1.*t + mod_f1 * phase3);
x6 = sin(2*pi*f1.*t + mod_f2 * phase4);

sig1 = x1 + x2;
sig2 = x3 + x4;
sig3 = x5 + x6;

if dichotic
    sig1(1,:) = x1;
    sig1(2,:) = x2;
    sig2(1,:) = x3;
    sig2(2,:) = x4;
    sig3(1,:) = x5;
    sig3(2,:) = x6;
else
    sig1(2,:) = sig1;
    sig2(2,:) = sig2;
    sig3(2,:) = sig3;
end

stim = {sig1, sig2, sig3};


end
