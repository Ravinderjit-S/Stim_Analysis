function [stim] = EFR(dur,fs,fm,filt_struct)
%dur=duration of stim
%fs=sampling rate
%fm = frequency of OSCOR
%filt_struct = filter structure
%no_filt = boolean, if 1 then no filtering

nn = randn(1,round(dur*3*fs));

t = 0:1/fs:size(nn,2)/fs-1/fs; 
A = 0.5*cos(2*pi*fm.*t)+0.5;
nn = nn.*A;


lenNN = round(dur*fs);

nn = nn';

nn = filter(filt_struct,nn);
nn = nn(lenNN+1:2*lenNN,:)';

stim =nn;





