clear

f1 = 1000;
f2 = 4*f1;
fs = 44100;
tlen = 1;
fm = 4;
phi_deg = 90;
dichotic = 0;
risetime = .125;

[stim_fm] = FM_phi(f1,f2,fs,tlen,fm,phi_deg,dichotic,1);
[stim_am] = SAM_phi(f1,f2,fs,tlen,fm,phi_deg,dichotic,0);

phi_rad = (phi_deg/360) * 2*pi;
t = 0:1/fs:tlen-1/fs;
x = 0.5+ 0.5*sin(2*pi*fm.*t);
y = 0.5+ 0.5*sin(2*pi*fm.*t+phi_rad);

figure,plot(t,x,t,y)

figure,
spectrogram(stim_fm{4}(1,:), round(.05*fs), round(0.05*fs*0.8),800:1:4500,fs,'yaxis')
set(gca,'fontsize',18)

figure,
spectrogram(stim_am{3}(1,:), round(.05*fs), round(0.05*fs*0.8),0:1:5000,fs,'yaxis')

for j = 1:4
    stimulus = stim_fm{j};
    energy = mean(rms(stimulus'));
    stimulus(1,:) = rampsound(stimulus(1,:),fs,risetime) / energy;
    stimulus(2,:) = rampsound(stimulus(2,:),fs,risetime) / energy;
    stim{j} = stimulus;
end

ISI = zeros(2,fs* (tlen /2)); 

stimulus_all = horzcat(stim{1}, ISI, stim{2}, ISI, stim{3}, ISI, stim{4});
stimulus_all = scaleSound(stimulus_all);

soundsc(stimulus_all,fs)
    
fm = 8
f_ex = 1000;
phase1 = (f_ex*0.3/fm) *sin(2*pi*fm.*t);
fm_ex = sin(2*pi*f_ex.*t + phase1);
figure,spectrogram(fm_ex,round(.05*fs),round(0.05*fs*0.8),0:1:12000,fs,'yaxis')

soundsc(fm_ex,fs);
%     
    
    
    