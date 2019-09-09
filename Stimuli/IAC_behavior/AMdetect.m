function [stim] = AMdetect(dur, fs, depth, fm, BPfilt, IAC)
%AM detection 3AFC
%this will return a stim with 2 nbn that have no AM and one that does at
%the specified depth and fm

nbn1 = randn(2,round(dur*3*fs));
nbn2 = randn(2,round(dur*3*fs));
nbn3 = randn(2,round(dur*3*fs));

lenNBN = round(dur*fs);

nbn1 = filter(BPfilt, nbn1');
nbn1 = nbn1(lenNBN+1:2*lenNBN,:)';

nbn2 = filter(BPfilt, nbn2');
nbn2 = nbn2(lenNBN+1:2*lenNBN,:)';

nbn3 = filter(BPfilt, nbn3');
nbn3 = nbn3(lenNBN+1:2*lenNBN,:)';

t = 0:1/fs:size(nbn3,2)/fs-1/fs;
Amod = (depth/2)*cos(2*pi*fm.*t) + depth/2 + (1-depth);

switch IAC  % should only be correlated or uncorrelated 
    case 1
        nbn1(2,:) = nbn1(1,:);
        nbn2(2,:) = nbn2(1,:);
        nbn3(2,:) = nbn3(1,:);
    case 0
    otherwise 
        error('Check IAC value')
end


nbn3 = nbn3 .* [Amod;Amod]; 
stim = [{nbn1}, {nbn2}, {nbn3}];
