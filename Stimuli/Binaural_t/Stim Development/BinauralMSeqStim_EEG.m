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
shifts = [];
for j = 1:Trials
    stimL = [];
    stimR = [];
    shift_j = randi(length(Mseq));
    shifts = [shifts,shift_j]; 
    Mseq_j = circshift(Mseq,shift_j);
    for i =1:length(Mseq)
        nbn = makeNBNfft_binaural_V2(flow,fhigh,M_dur,fs,Mseq_j(i),0);
        nbn = nbn(:,1:end-1);
        nbn = nbn.*windw;
        stimL = [stimL nbn(1,:)]; stimR = [stimR nbn(2,:)]; %#ok
    end
    stimIAC = [stimL;stimR];
    save(['StimMseqShuff_IAC/stim_Mseq_IAC' num2str(j)],'stimIAC','shift_j')
end
save('StimMseqShuff_IAC/MseqShifts.mat','shifts','Mseq')
%sound(stim,fs)
   
% t = 0:1/fs:length(stimR)/fs-1/fs;
% % WindowCrossCorr(stimL,stimR, round(600e-6*fs), round(.002*fs),'hann',round(0.5*.002*fs),fs,1);
% figure,plot(t,stimIAC(1,:),'b',t,stimIAC(2,:),'r')


ITD_jump = 500e-6;
for j = 1:Trials
    stimL = [];
    stimR = [];
    for i =1:length(Mseq)
        if Mseq(i) == 1
            ITD = ITD_jump;
        else
            ITD = 0;
        end
        nbn = makeNBNfft_binaural_V2(flow,fhigh,M_dur,fs,1,ITD);
        nbn = nbn(:,1:end-1);
        nbn = nbn.*windw;
        stimL = [stimL nbn(1,:)]; stimR = [stimR nbn(2,:)]; %#ok
    end
    stimITD = [stimL;stimR];
    save(['StimMseqShuff_ITD/stim_Mseq_ITD' num2str(j)],'stimITD')
end


% WindowCrossCorr(stimL,stimR, round(800e-6*fs), round(.020*fs),'hann',round(0.0*.020*fs),fs,1);
% 
% figure,plot(t,stimITD(1,:),'b',t,stimITD(2,:),'r')

