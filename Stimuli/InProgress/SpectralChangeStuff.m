% Testing various ways to do Spectral Mseq
clear

%% Bounce between two frequencies
fs = 48828;
f1 = 500;
f2 = 550;
dur = 1;
t = 0:1/fs:dur-1/fs;
fm = 5;

x1 = sin(2*pi*f1.*t);
x2 = sin(2*pi*f2.*t);

sqt = linspace(0,2*pi*fm*dur,length(t));
AM = 0.5*(sin(2*pi*2*fm.*t)+1);
A = square(sqt);
A = (A+1) /2; %make A a sqaure wave b/t 0 & 1
B = circshift(A,round(fs/(2*fm)));

figure,plot(t,A,t,B,'r')
sig = A.*x1 + B.*x2;
sig = sig.*AM;
figure,plot(t,sig)

soundsc(sig,fs)

%% bounce between two phases
x1 = sin(2*pi*f1.*t+A*pi/4);
figure,plot(t,x1)
%soundsc(x1,fs)

%% AM with FM
f1 = 40;
x2 = sin(2*pi*f1.*t + f1*.05/fm * sin(2*pi*fm.*t));
nn = randn(1,dur*fs);
sig = nn.*x2;
figure,plot(t,sig)
% soundsc(sig,fs)

%% Make noise with bunch of tones with same FM

phases = 2*pi*rand(1,floor(fs/2*dur)); %random starting phases
freqs = 0:1/dur:floor(fs/2);
stimL = zeros(1,dur*fs);
stimR = zeros(1,dur*fs);
for n = 1:length(phases)
    stimL = stimL + sin(2*pi*freqs(n).*t + phases(n)) / length(phases);
    stimR = stimR + sin(2*pi*freqs(n).*t + phases(n) + 2*pi*sin(2*pi*fm.*t)) / length(phases);
end
stim = [stimL;stimR];
figure,plot(t,stim')
soundsc(stim,fs)


