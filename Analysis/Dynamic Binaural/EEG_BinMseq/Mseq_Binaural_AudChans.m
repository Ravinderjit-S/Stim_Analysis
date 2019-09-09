% Look at some processed binaural EEG data mseq
clear
subj = 'Rav';
load([subj 'EEGMseqProcessed.mat'])
load('Mseq_4096fs_compensated.mat')

fs = 4096;
Keep_H = 0.5; %time length of impulse response to keep

Num_noiseFloors = size(NoiseFloors_IAC,2);
t = (0:length(Mseq_sig)-1)/fs;
figure(), hold on
Aud_channels=1:32;
Rev_IAC_AudChan = mean(Rev_IAC(Aud_channels,:));
Rev_ITD_AudChan = mean(Rev_ITD(Aud_channels,:));
for kk = 1:Num_noiseFloors
    dummy = NoiseFloors_IAC{kk};
    dummy_AudChan = mean(dummy(Aud_channels,:));
    plot(t,dummy_AudChan,'r')
end
plot(t,Rev_IAC_AudChan,'b','linewidth',2), hold off, xlim([0 Keep_H]), title('Averaged Auditory Channels IAC H(t)'), xlabel('Time (sec)')

figure(), hold on

for kk = 1:Num_noiseFloors
    dummy = NoiseFloors_IAC{kk};
    dummy_AudChan = mean(dummy(Aud_channels,:));
    [pdumb_audch,f] = pmtm(dummy_AudChan(1:round(Keep_H*fs)),2.5,[],fs);
    A_pdumb_audch(kk,:) = pdumb_audch;
    plot(f,pow2db(pdumb_audch),'r')
end
[pIAC_audch, f] = pmtm(Rev_IAC_AudChan(1:round(Keep_H*fs)),2.5,[],fs);
plot(f,pow2db(pIAC_audch),'b','linewidth',3), hold off,xlim([0 30]),title('Averaged Auditory Channels IAC H(f)'), xlabel('Frequency'), ylabel('Power (dB/Hz)')
Hf_zspec = (pIAC_audch - mean(A_pdumb_audch)') ./ std(A_pdumb_audch)';
figure,plot(f,Hf_zspec,'linewidth',2), title('IAC Z-spectrum'),xlim([0 20]),xlabel('Frequency (Hz)'), ylabel('Z-score')
% set(gca,'fontsize',35)
figure(), hold on
for kk = 1:Num_noiseFloors
    dummy = NoiseFloors_ITD{kk};
    dummy_AudChan = mean(dummy(Aud_channels,:));
    plot(t,dummy_AudChan,'r')
end
plot(t,Rev_ITD_AudChan,'b'), hold off, xlim([0 Keep_H]), title('Averaged Auditory Channels ITD H(t)')

figure(), hold on
 
for kk = 1:Num_noiseFloors
    dummy = NoiseFloors_ITD{kk};
    dummy_AudChan = mean(dummy(Aud_channels,:));
    [pdumb_audch,f] = pmtm(dummy_AudChan(1:round(Keep_H*fs)),2.5,[],fs);
    A_pdumb_audch(kk,:) = pdumb_audch;
    plot(f,pow2db(pdumb_audch),'r')
end
[pIAC_audch, f] = pmtm(Rev_ITD_AudChan(1:round(Keep_H*fs)),2.5,[],fs);
plot(f,pow2db(pIAC_audch),'b','linewidth',3), hold off,xlim([0 30]),title('Averaged Auditory Channels ITD H(f)'), xlabel('Frequency'), ylabel('Power (dB/Hz)')
Hf_zspecITD = (pIAC_audch - mean(A_pdumb_audch)') ./ std(A_pdumb_audch)';
figure,plot(f,Hf_zspecITD,'linewidth',2),title('ITD Zspec'), xlim([0 20]), xlabel('Frequency (Hz)'), ylabel('Z-score')



