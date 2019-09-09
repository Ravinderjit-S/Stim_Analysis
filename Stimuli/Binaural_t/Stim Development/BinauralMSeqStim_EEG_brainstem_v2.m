%Generating IAC(t) & ITD(t) stim for EEG

clear all
fs = 48828.125;
load('Mseq_IAC_ITD_bs.mat'); %loads Mseq
load('Seed16.mat'); %load s
rng(s);
M_dur = .002;%2 ms
M_samps = round(M_dur*fs); %number of samples for each point in Mseq
%LPfilt = designfilt('lowpassfir', 'StopbandFrequency', 2000, 'PassbandFrequency', 1500, 'StopbandAttenuation', 60, 'PassbandRipple', 1, 'SampleRate', fs);
Msequence = [];
for i =1:length(Mseq_bs)
    Msequence = [Msequence Mseq_bs(i)*ones(1,M_samps)];
end

Trials = 3000;
t = 0:1/fs:length(Msequence)/fs-1/fs;
AM = 1/2*sin(2*pi*1/M_dur*t-pi/2)+1/2;

for j = 1:1
    stimL = randn(1,length(Msequence));
    stimL = stimL.*AM;
    stimIAC_bs = filtfilt(BPfilt,stimL);
    stimIAC_bs(2,:) = stimIAC_bs(1,:);
    stimIAC_bs(2,Msequence==-1) = -stimIAC_bs(1,Msequence==-1);
    save(['StimMseq_IAC_bs/stim_Mseq_IAC_bs' num2str(j)],'stimIAC_bs')
end

stimIAC_bs = scaleSound(stimIAC_bs);
t = 0:1/fs:length(stimIAC_bs)/fs-1/fs;
figure,plot(t,stimIAC_bs)
hold on, plot(t,Msequence,'k')
sound(stimIAC_bs,fs)
   
% t = 0:1/fs:length(stimR)/fs-1/fs;
% % WindowCrossCorr(stimL,stimR, round(600e-6*fs), round(.002*fs),'hann',round(0.5*.002*fs),fs,1);
% figure,plot(t,stimIAC(1,:),'b',t,stimIAC(2,:),'r')


% ITD_jump = 500e-6;
% for j = 1:300
%     stimL = [];
%     stimR = [];
%     for i =1:length(Mseq)
%         if Mseq(i) == 1
%             ITD = ITD_jump;
%         else
%             ITD = 0;
%         end
%         nbn = makeNBNfft_binaural_V2(flow,fhigh,M_dur,fs,1,ITD);
%         nbn = nbn.*windw;
%         stimL = [stimL nbn(1,:)]; stimR = [stimR nbn(2,:)]; %#ok
%     end
%     stimITD_bs = [stimL;stimR];
%     save(['StimMseq_ITD_bs/stim_Mseq_ITD_bs' num2str(j)],'stimITD_bs')
% end
% 

% WindowCrossCorr(stimL,stimR, round(800e-6*fs), round(.020*fs),'hann',round(0.0*.020*fs),fs,1);
% 
% figure,plot(t,stimITD(1,:),'b',t,stimITD(2,:),'r')

