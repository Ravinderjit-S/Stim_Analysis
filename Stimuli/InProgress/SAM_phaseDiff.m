% Detect phase difference of 2 AM stimuli
clear
path = '../CommonExperiment';
p = genpath(path);
addpath(p); %add path to commonly used functions

fs = 48828;
f1 = 2000;
f2 = f1*4;
t = 0:1/fs:1-1/fs;

fm = 4;
phi_deg = 45;
phi_rad = (phi_deg/360) * 2*pi;
AM1 = 0.5 * (sin(2*pi*fm.*t) +1);
AM2 = 0.5 * (sin(2*pi*fm.*t + phi_rad) +1);


x1 = AM1 .* sin(2*pi*f1*t);
x2 = AM2 .* sin(2*pi*f2*t);

sig1 = AM1.* sin(2*pi*f1*t) + AM1.*sin(2*pi*f2*t);
sig2 = x1+x2;

sig1 = rampsound(sig1,fs,0.010);
sig2 = rampsound(sig2,fs,0.010);

sigs = [sig1;sig2];
a = randperm(2);
sigs = sigs(a,:);

for i =1:length(a)
    soundsc(sigs(i,:),fs)
    pause(1.5)
end

figure, hold on
plot(t,x1,'b')
plot(t,x2,'r')
hold off
