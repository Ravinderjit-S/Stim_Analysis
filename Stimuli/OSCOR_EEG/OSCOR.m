function [stim] = OSCOR(dur,fs,fm,filt_struct,no_filt)
%dur=duration of stim
%fs=sampling rate
%fm = frequency of OSCOR
%filt_struct = filter structure
%no_filt = boolean, if 1 then no filtering

nn = randn(2,round(dur*3*fs));

t = 0:1/fs:size(nn,2)/fs-1/fs; 
A = cos(2*pi*fm.*t);
B = sqrt(1-A.^2);
nn(2,:) = A.*nn(1,:) + B.*nn(2,:);

lenNN = round(dur*fs);

nn = nn';
if no_filt
    nn = nn(lenNN+1:2*lenNN,:)';
else
    nn = filter(filt_struct,nn);
    nn = nn(lenNN+1:2*lenNN,:)';
end
stim =nn;





