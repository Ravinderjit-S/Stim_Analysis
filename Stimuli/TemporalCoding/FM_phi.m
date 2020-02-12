function [stim] = FM_phi(f1,f2,fs,tlen,fm,phi_deg)
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

phase1 = dF1/fm * sin(2*pi*fm*t);
phase2 = dF2/fm * sin(2*pi*fm*t);
phase3 = dF2/fm * sin(2*pi*fm*t + phi_rad);

x1 = sin(2*pi*f1*t + phase1);
x2 = sin(2*pi*f2*t + phase2);
x3 = sin(2*pi*f2*t + phase3);

sig  = x1 + x2;
sig2 = x1 + x3; 

stim = {sig, sig, sig2};