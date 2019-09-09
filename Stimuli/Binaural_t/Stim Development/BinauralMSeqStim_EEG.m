%Generating IAC(t) & ITD(t) stim for EEG

clear all
fs = 48828.125;
load('Mseq_IAC_ITD.mat'); %loads Mseq
load('Seed16.mat'); %load s
rng(s);
M_dur = .050;%50 ms
M_samps = round(M_dur*fs); %number of samples for each point in Mseq


flow = 200;
fhigh =1.5e3; 

Trials = 300;
%windw = hann(round(M_dur*fs))'; windw = [windw;windw];
t_windw = 0:1/fs:M_samps/fs-1/fs;
windw = 0.5*(1+sin(2*pi*(1/M_dur)*t_windw-pi/2)); windw = [windw;windw];

for j = 1:300
    stimL = [];
    stimR = [];
    for i =1:length(Mseq)
        nbn = makeNBNfft_binaural_V2(flow,fhigh,M_dur,fs,Mseq(i),0);
        nbn = nbn.*windw;
        stimL = [stimL nbn(1,:)]; stimR = [stimR nbn(2,:)]; %#ok
    end
    stimIAC = [stimL;stimR];
    save(['StimMseq_IAC/stim_Mseq_IAC' num2str(j)],'stimIAC')
end
%sound(stim,fs)
   
% t = 0:1/fs:length(stimR)/fs-1/fs;
% % WindowCrossCorr(stimL,stimR, round(600e-6*fs), round(.002*fs),'hann',round(0.5*.002*fs),fs,1);
% figure,plot(t,stimIAC(1,:),'b',t,stimIAC(2,:),'r')


ITD_jump = 500e-6;
for j = 1:300
    stimL = [];
    stimR = [];
    for i =1:length(Mseq)
        if Mseq(i) == 1
            IPD = pi/2;
        else
            IPD = 0;
        end
        nbn = makeNBNfft_binaural_V3(flow,fhigh,M_dur,fs,1,0,IPD);
        nbn = nbn.*windw;
        stimL = [stimL nbn(1,:)]; stimR = [stimR nbn(2,:)]; %#ok
    end
    stimIPD = [stimL;stimR];
    save(['StimMseq_IPD/stim_Mseq_IPD' num2str(j)],'stimIPD')
end


% WindowCrossCorr(stimL,stimR, round(800e-6*fs), round(.020*fs),'hann',round(0.0*.020*fs),fs,1);
% 
% figure,plot(t,stimITD(1,:),'b',t,stimITD(2,:),'r')

