
clear all

fs = 48828.125;
flow =0;
fhigh = fs/2;
dur = 2.0;
rho = -1; 
ITD =0;
IPD = 0;

BPfilt = designfilt('bandpassiir', 'StopbandFrequency1', 190, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 1550, 'StopbandAttenuation1', 30, 'PassbandRipple', 1, 'StopbandAttenuation2', 30, 'SampleRate', fs);
% BPfilt = designfilt('bandpassiir', 'StopbandFrequency1', 190, 'PassbandFrequency1', 200, 'PassbandFrequency2', 800, 'StopbandFrequency2', 850, 'StopbandAttenuation1', 30, 'PassbandRipple', 1, 'StopbandAttenuation2', 30, 'SampleRate', fs);


[nbn] = randn(2, round(dur*fs)); %makeNBNfft_binaural_V3(flow,fhigh,dur,fs,1,ITD,IPD);
[nbn2] = randn(2, round(dur*fs)); %makeNBNfft_binaural_V3(flow,fhigh,dur,fs,-1,ITD,IPD);
nbn3 = randn(2, round(dur*fs));


fm = 100;
t = 0:1/fs:dur-1/fs;
A = cos(2*pi*fm.*t);
% A = cos(2*pi*fm.*t);
%B = 0.5*sin(2*pi*fm.*t)+0.5;
B = sqrt(1-A.^2);
nbn3(2,:) = A.*nbn3(1,:)+B.*nbn3(2,:);
nbn3 = nbn3';

% A2 = 0.5;
% B2 = sqrt(1-A2.^2);
% nbn(2,:) = A2.*nbn(1,:)+B2.*nbn(2,:);
% nbn2(2,:) = A2.*nbn2(1,:)+B2.*nbn2(2,:);


nbn4 = vertcat(nbn(1,:),nbn(1,:));


% nbn1 = nbn';
nbn1 = filtfilt(BPfilt,vertcat(zeros(size(nbn')), nbn', zeros(size(nbn'))));
nbn1 = nbn1(size(nbn,2)+1:2*size(nbn,2),:);
nbn1 = nbn1';

nbn2 = filtfilt(BPfilt, vertcat(zeros(size(nbn')), nbn2', zeros(size(nbn'))));
nbn2 = nbn2(size(nbn,2)+1:2*size(nbn,2),:);
nbn2 = nbn2';

nbn3 = filtfilt(BPfilt,vertcat(zeros(size(nbn3)), nbn3, zeros(size(nbn3))));
nbn3 = nbn3(size(nbn,2)+1:2*size(nbn,2),:);
nbn3 = nbn3';

ramp = .2;
nbn1(1,:) = rampsound(nbn1(1,:),fs,ramp);nbn1(2,:) = rampsound(nbn1(2,:),fs,ramp);
nbn2(1,:) = rampsound(nbn2(1,:),fs,ramp); nbn2(2,:) = rampsound(nbn2(2,:),fs,ramp);
nbn3(1,:) = rampsound(nbn3(1,:),fs,ramp); nbn3(2,:) = rampsound(nbn3(2,:),fs,ramp);
nbn4(1,:) = rampsound(nbn4(1,:),fs,ramp); nbn4(2,:) = rampsound(nbn4(2,:),fs,ramp);

nbn1(1,:) = nbn1(1,:)./rms(nbn1(1,:)); nbn1(2,:) = nbn1(2,:)./rms(nbn1(2,:));
nbn2(1,:) = nbn2(1,:)./rms(nbn2(1,:)); nbn2(2,:) = nbn2(2,:)./rms(nbn2(2,:));
nbn3(1,:) = nbn3(1,:)./rms(nbn3(1,:)); nbn3(2,:) = nbn3(2,:)./rms(nbn3(2,:));

% nbn1 = nbn1./max(max(abs(nbn1))); nbn2 = nbn2./max(max(abs(nbn2))); nbn3 = nbn3./max(max(abs(nbn3)));
nbn1 = nbn1./20; nbn2 = nbn2./20; nbn3 = nbn3./20;

% sound(nbn3,fs)
% sound(nbn1,fs)

% [p1 f] = pmtm(nbn3(1,:),2.5,[],fs);
% [p2 f] = pmtm(nbn3(2,:),2.5,[],fs);
% 
% figure,plot(f,pow2db(p1),f,pow2db(p2),'r'), title(['IACfm: ' num2str(fm)])

sig = [{nbn1}, {nbn2}, {nbn3}];
a = randperm(3);
for i =1:3
    play = horzcat(sig{a(i)},zeros(2,round(fs/2)));
    sound(play,fs)
    pause(3)
end
% figure,plot(nbn3'), title('nbn3')
% figure,plot(nbn1'),title('nbn1')
% figure,plot(nbn2'),title('nbn2')
% 20*log10(rms(sig{1}')/(20e-6))
% 20*log10(rms(sig{3}')/(20e-6))

%audiowrite(['BS' num2str(fm) '_' num2str(find(a==3)) '.wav'],play',round(fs))
