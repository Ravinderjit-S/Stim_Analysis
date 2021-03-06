function [stim] = SAM_phi_incoh(f1,f2,fs,tlen,fm,base_phi,phi_deg,dichotic,ref)
%This fucntion will return a 3 AFC stim where the 3rd stimulus has an AM
%with a phase difference.  In this task, the modulations are always incoh-
%erent but the target has a different incoherence that must be detected
%f1 = frequency of tone 1
%f2 = frequency of tone 2
%fs = sampling rate
%tlen = length of stim in seconds
%fm = SAM rate 
%base_phi = base phase incoherence in degrees
%phi_deg = change in phase difference between two AMs in degrees
%diotic = (boolean) send two different AM signals to different ears
%ref = (boolean) add a reference for participants to hear before the 3-AFC


phi_rad = (phi_deg/360) * 2*pi;
base_rad = (base_phi/360) * 2*pi;
t = 0:1/fs:tlen-1/fs;

rand_phase1 = rand() * 2*pi;
rand_phase2 = rand() * 2*pi;
rand_phase3 = rand() * 2*pi;
rand_phaseRef = rand() * 2*pi;
AM1 = 0.5 * (sin(2*pi*fm.*t + rand_phase1) + 1);
AM1_1 = 0.5 * (sin(2*pi*fm.*t + rand_phase1 + base_rad) + 1);
AM2 = 0.5 * (sin(2*pi*fm.*t + rand_phase2) + 1);
AM2_1 = 0.5 * (sin(2*pi*fm.*t + rand_phase2 + base_rad) + 1);
AM3 = 0.5 * (sin(2*pi*fm.*t + rand_phase3) + 1);
AM4 = 0.5 * (sin(2*pi*fm.*t + rand_phase3  + base_rad + phi_rad) + 1);
AM_ref = 0.5 * (sin(2*pi*fm.*t + rand_phaseRef) + 1);
AM_ref_1 = 0.5 * (sin(2*pi*fm.*t + rand_phaseRef + base_rad) + 1);


sig1 = AM1 .* sin(2*pi*f1.*t) + AM1_1 .* sin(2*pi*f2.*t);
sig2 = AM2 .* sin(2*pi*f1.*t) + AM2_1 .* sin(2*pi*f2.*t);
sig3 = AM3 .* sin(2*pi*f1.*t) + AM4 .* sin(2*pi*f2.*t);
sig_ref = AM_ref .* sin(2*pi*f1.*t) + AM_ref_1 .*sin(2*pi*f2.*t);

if dichotic
    sig1(1,:) = AM1 .* sin(2*pi*f1.*t);
    sig1(2,:) = AM1_1 .* sin(2*pi*f2.*t);
    sig2(1,:) = AM2 .* sin(2*pi*f1.*t);
    sig2(2,:) = AM2_1 .* sin(2*pi*f2.*t);
    sig3(1,:) = AM3 .* sin(2*pi*f1.*t);
    sig3(2,:) = AM4 .* sin(2*pi*f2.*t);
    sig_ref(1,:) = AM_ref .* sin(2*pi*f1.*t);
    sig_ref(2,:) = AM_ref_1 .* sin(2*pi*f2.*t);
else
    sig1(2,:) = sig1;
    sig2(2,:) = sig2;
    sig3(2,:) = sig3;
    sig_ref(2,:) = sig_ref;
end

if ref
    stim = {sig_ref, sig1, sig2, sig3};
else
    stim = {sig1, sig2, sig3};
end


end