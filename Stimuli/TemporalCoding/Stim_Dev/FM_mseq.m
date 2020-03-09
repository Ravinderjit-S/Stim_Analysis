function [stim] = FM_mseq(range_carrier,mseqFM,Point_len,fs)
%frange = frequency carrier is inbetween
%m = bits of mseq
%upperF = highest frequency to characterize with mseq
%fs = sampling rate

UpperF = fs/Point_len;
t = 0:1/fs:length(mseqFM)/fs -1/fs;

f1 = range_carrier(1) + (range_carrier(2) -range_carrier(1))*rand();
f2 = 1.05*f1;

x1 = sin(2*pi*f1.*t);
x1(mseqFM ==-1) = 0;
x2 = sin(2*pi*f2.*t);
x2(mseqFM==1) = 0;
AM = 0.5* (sin(2*pi.*t*UpperF-pi/2)+1); %AM to mask phase discontinuities

stim = AM .*(x1 + x2);
stim = [stim;stim];

end