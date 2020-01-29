function [stim] = IACsinAFC3(dur,fs,fm,BPfilt)
%Oscillating IAC 3 AFC
%this function will return a stim containing 2 noises with a correlation of
%0 and one with a sinusoidally oscilliating correlation at frequency fm
%Ramping and scaling will have to be done outside this function
% dur in seconds, fm in Hz

nbn1 = randn(2,round(dur*3*fs)); %generating longer noise to deal with filter transients
nbn2 = randn(2,round(dur*3*fs));
nbn3 = randn(2,round(dur*3*fs));

t = 0:1/fs:size(nbn1,2)/fs-1/fs; 
A = sin(2*pi*fm.*t);
B = sqrt(1-A.^2);
nbn3(2,:) = A.*nbn3(1,:) + B.* nbn3(2,:);

lenNBN = round(dur*fs);

if ~isempty(BPfilt)
    nbn1 = filter(BPfilt, nbn1');
    nbn2 = filter(BPfilt, nbn2');
    nbn3 = filter(BPfilt, nbn3');
end

nbn1 = nbn1(lenNBN+1:2*lenNBN,:)';
nbn2 = nbn2(lenNBN+1:2*lenNBN,:)';
nbn3 = nbn3(lenNBN+1:2*lenNBN,:)';


stim = [{nbn1}, {nbn2}, {nbn3}];


end
