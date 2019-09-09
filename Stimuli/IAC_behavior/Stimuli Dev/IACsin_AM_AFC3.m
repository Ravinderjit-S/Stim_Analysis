function [stim] = IACsin_AM_AFC3(dur, fs, depth, fm, BPfilt,randDepth,randIAC)
%Detect OSCOR with other 2 options being AM with a certain depth at same fm
%randDepth = 1 means the modulations on the two noises will be random
%randIAC =1 means the IAC of the 2 noises will be random


nbn1 = randn(2,round(dur*3*fs));
nbn2 = randn(2,round(dur*3*fs));
nbn3 = randn(2,round(dur*3*fs));
%nbn2 = nbn1; nbn3 = nbn1;

t = 0:1/fs:size(nbn1,2)/fs-1/fs; 
A = cos(2*pi*fm.*t);
B = sqrt(1-A.^2);
nbn3(2,:) = A.*nbn3(1,:) + B.*nbn3(2,:);

if randIAC ==1
    A1 = 0+rand()*0.00%rand()*0.25+0.25;  %rand*0.25 .. 0.3
    B1 = sqrt(1-A1.^2);
    A2 = 0+rand()*0.00%rand()*0.25+0.25; %rand*0.25+0.25  0.27
    B2 = sqrt(1-A2.^2);
    if randi(2) == 1  %randomly apply lower IAC to nbn1 or nbn2
        nbn1(2,:) = A1.*nbn1(1,:) + B1.*nbn1(2,:);
        nbn2(2,:) = A2.*nbn2(1,:) + B2.*nbn2(2,:);
    else
        nbn1(2,:) = A2.*nbn1(1,:) + B2.*nbn1(2,:);
        nbn2(2,:) = A1.*nbn2(1,:) + B1.*nbn2(2,:);
    end
end
    

lenNBN = round(dur*fs);

nbn1 = filter(BPfilt, nbn1');
nbn1 = nbn1(lenNBN+1:2*lenNBN,:)';

nbn2 = filter(BPfilt, nbn2');
nbn2 = nbn2(lenNBN+1:2*lenNBN,:)';

nbn3 = filter(BPfilt, nbn3');
nbn3 = nbn3(lenNBN+1:2*lenNBN,:)';

t = 0:1/fs:size(nbn3,2)/fs-1/fs;

if randDepth ==1
    depth1 = 0.7+0.02*rand;%0.2 + 0.08*rand(); %0.2 + 0.08
    depth2 = 0.98+0.02*rand;%0.28 + 0.08*rand(); % 0.28 + 0.08
    Amod1 = (depth1/2)*cos(2*pi*fm.*t) + depth1/2 + (1-depth1);% + (randi([0,1])*2-1)*linspace(0,0.1,length(t));
    Amod2 = (depth2/2)*cos(2*pi*fm.*t) + depth2/2 + (1-depth2);% + (randi([0,1])*2-1)*linspace(0,0.1,length(t));
else
    Amod = (depth/2)*cos(2*pi*fm.*t) + depth/2 + (1-depth);
    Amod1 = Amod; Amod2 = Amod;
end
    

% if AM_IAC == -1
%     nbn1(2,:) = -nbn1(1,:);
%     nbn2(2,:) = -nbn2(1,:);
% elseif AM_IAC == 0
% else
%     error('Check IAC input')
% end


nbn1 = nbn1 .* [Amod1;Amod1];
nbn2 = nbn2 .* [Amod2;Amod2];


stim = [{nbn1}, {nbn2}, {nbn3}];

end

