clear

%load('Mseq_4096fs.mat') %Mseq_sig
%load('Mseq_dumb.mat') %Mseq_dumb
load('Mseq_4096fs_compensated.mat')

Subjects = [{'Rav'},{'S203'},{'S204'},{'S132'},{'S206'},{'S207'}];
subj = Subjects{4};
load(['IAC_evoked20_' subj '.mat']) % this data is lowpassed at 50 Hz
load(['ITD_evoked20_' subj '.mat'])
load(['IAC_epochs20_' subj '.mat'])
load(['ITD_epochs20_' subj '.mat'])

IAC_EEG_epochs = IAC_EEG_epochs(:,:,1:length(Mseq_sig));
IAC_EEG_avg = IAC_EEG_avg(:,1:length(Mseq_sig));
ITD_EEG_epochs = ITD_EEG_epochs(:,:,1:length(Mseq_sig));
ITD_EEG_avg = ITD_EEG_avg(:,1:length(Mseq_sig));

% load('HPfilter_3.mat') %high pass filter ... var is HPfilter2
% HPfilter = HPfilter3;
%HPfilter = designfilt('highpassiir', 'StopbandFrequency', 0.1, 'PassbandFrequency', 0.5, 'StopbandAttenuation',5, 'PassbandRipple', 1, 'SampleRate', 4096);

% if strcmp(subj,'Rav')
%     Aud_channels_IAC = [5,25,6,26,27,31,4,8,32]; %channels you can see an evoked response in
% elseif strcmp(subj,'S204')
%     Aud_channels_IAC = [1:23,25:27,29:32]; %not 24 & 28
%     Aud_channels_IAC_strong = [4,5,26,30,31,32]; 
%     Aud_channels_ITD = [1:9,11:22, 24:27, 29:32];
%     Aud_channels_ITD_strong = [5,9,11,15,16,20,23,24,26,31,32];
% end
% Aud_channels_IAC =1;
% Aud_channels = Aud_channels_IAC; 

Aud_channels=[31,32,26,27,5];
    
% Aud_channels = Rav_ER_channels;
fs =4096;
Mseq_ITD = Mseq_sig;
Num_noiseFloors = 50;
order = 15;

figure(), hold on
t = 0:1/fs:(size(ITD_EEG_avg,2)-1)/fs;
for i=1:32
%     IAC_EEG_avg(i,:) = filtfilt(HPfilter,IAC_EEG_avg(i,:));
%     Trend = EEG_Trend(IAC_EEG_avg,fs,32,order);
%     IAC_EEG_avg = IAC_EEG_avg(1:32,:) - Trend;
    subplot(8,4,i),plot(t,IAC_EEG_avg(i,:)),title(['IAC' num2str(i)]),xlim([0 2])
    if any(i==Aud_channels)
        set(gca,'Color','y')
    end
end
hold off

figure(), hold on
t = 0:1/fs:(size(ITD_EEG_avg,2)-1)/fs;
for i=1:32
%     IAC_EEG_avg(i,:) = filtfilt(HPfilter,IAC_EEG_avg(i,:));
%     Trend = EEG_Trend(IAC_EEG_avg,fs,32,order);
%     IAC_EEG_avg = IAC_EEG_avg(1:32,:) - Trend;
    [p, f] = pmtm(IAC_EEG_avg(i,:),2.5,[],fs);
    subplot(8,4,i),plot(f,pow2db(p)),title(['IAC' num2str(i)]),xlim([0 50])
    if any(i==Aud_channels)
        set(gca,'Color','y')
    end
end


figure(), hold on
for i=1:32
%      ITD_EEG_avg(i,:) = filtfilt(HPfilter,ITD_EEG_avg(i,:));
%     Trend = EEG_Trend(ITD_EEG_avg,fs,32,order);
%     ITD_EEG_avg = ITD_EEG_avg(1:32,:) - Trend;
    subplot(8,4,i),plot(t,ITD_EEG_avg(i,:)),title(['ITD' num2str(i)]),xlim([0 2])
    if any(i==Aud_channels)
        set(gca,'Color','y')
    end
end
hold off

figure(), hold on
for i=1:32
%     IAC_EEG_avg(i,:) = filtfilt(HPfilter,IAC_EEG_avg(i,:));
%     Trend = EEG_Trend(IAC_EEG_avg,fs,32,order);
%     IAC_EEG_avg = IAC_EEG_avg(1:32,:) - Trend;
    [p, f] = pmtm(ITD_EEG_avg(i,:),2.5,[],fs);
    subplot(8,4,i),plot(f,pow2db(p)),title(['ITD' num2str(i)]),xlim([0 50])
    if any(i==Aud_channels)
        set(gca,'Color','y')
    end
end
hold off


NF_Powers_IAC = [];
for jj=1:Num_noiseFloors
    epochs_num_IAC = size(IAC_EEG_epochs,1);
    RandInds_IAC = randperm(epochs_num_IAC);
    epochs_num_ITD = size(ITD_EEG_epochs,1);
    RandInds_ITD = randperm(epochs_num_ITD);
    
    Noise_EEG_IAC = (sum(IAC_EEG_epochs(RandInds_IAC(1:round(epochs_num_IAC/2)),:,:)) - sum(IAC_EEG_epochs(RandInds_IAC(round(epochs_num_IAC/2)+1:end),:,:)))./epochs_num_IAC; %adding half of one with the other
    Noise_EEG_IAC = reshape(Noise_EEG_IAC,[34 length(Mseq_sig)]);
    Noise_EEG_ITD = (sum(ITD_EEG_epochs(RandInds_ITD(1:round(epochs_num_ITD/2)),:,:)) - sum(ITD_EEG_epochs(RandInds_ITD(round(epochs_num_ITD/2)+1:end),:,:)))./epochs_num_ITD;
    Noise_EEG_ITD = reshape(Noise_EEG_ITD,[34 length(Mseq_sig)]);
    
%     Trend_NIAC = EEG_Trend(Noise_EEG_IAC,fs,32,order);
%     Noise_EEG_IAC = Noise_EEG_IAC(1:32,:) - Trend_NIAC;
%     Trend_NITD = EEG_Trend(Noise_EEG_ITD,fs,32,order);
%     Noise_EEG_ITD = Noise_EEG_ITD(1:32,:) - Trend_NITD;
%     

    for kk = 1:32
%        Noise_EEG_IAC(kk,:) = filtfilt(HPfilter,Noise_EEG_IAC(kk,:));
%        Noise_EEG_ITD(kk,:) = filtfilt(HPfilter,Noise_EEG_ITD(kk,:));
        [Rev_IAC_dumbkk] = cconv(Noise_EEG_IAC(kk,:),Mseq_sig(end:-1:1));
        [Rev_ITD_dumbkk] = cconv(Noise_EEG_ITD(kk,:),Mseq_sig(end:-1:1));
        Rev_IACdumb(kk,:) = Rev_IAC_dumbkk(length(Mseq_sig):end);
        Rev_ITDdumb(kk,:) = Rev_ITD_dumbkk(length(Mseq_sig):end);
    end
    [pzz f] = pmtm(Noise_EEG_IAC(32,:),2.5,[],fs);
    NF_Powers_IAC = horzcat(NF_Powers_IAC,pzz);
    NoiseFloors_IAC{jj} = Rev_IACdumb;
    NoiseFloors_ITD{jj} = Rev_ITDdumb;
end
        

for i = 1:32
    [Rev_IAC_i] = cconv(IAC_EEG_avg(i,:),Mseq_sig(end:-1:1));
    [Rev_ITD_i] = cconv(ITD_EEG_avg(i,:),Mseq_ITD(end:-1:1));
    Rev_IAC(i,:) = Rev_IAC_i(length(Mseq_sig):end);
    Rev_ITD(i,:) = Rev_ITD_i(length(Mseq_sig):end);
end
    t = (0:length(Mseq_sig)-1)/fs;
%save([subj '_RevCorMseq.mat'],'Rev_IAC_dummies','Rev_ITD_dummies','Rev_IAC','Rev_ITD')
    
figure()
Keep_H = 1.0; %time length of impulse response to keep
for pp = 1:32
    subplot(8,4,pp)
    hold on
    for kk = 1:Num_noiseFloors
        dummy = NoiseFloors_IAC{kk};
        plot(t,dummy(pp,:),'Color',[0 1 1])
    end
    plot(t,Rev_IAC(pp,:),'b'),title(['IAC H(t):' num2str(pp)]), xlim([0 Keep_H])
    hold off
    if any(pp==Aud_channels)
        set(gca,'Color','y')
    end
end
figure()
for pp = 1:32
    subplot(8,4,pp)
    hold on
    for kk=1:Num_noiseFloors
        dummy = NoiseFloors_ITD{kk};
        plot(t,dummy(pp,:),'Color',[0 1 1])
    end
    plot(t,Rev_ITD(pp,:),'b'),title(['ITD H(t):' num2str(pp)]), xlim([0 Keep_H])
    hold off
    if any(pp==Aud_channels)
        set(gca,'Color','y')
    end
end

figure()
%[pIAC, f] = pmtm(Rev_IAC(32,:),5,0:1:50,fs);
for pp=1:32
    subplot(8,4,pp)
    hold on
    for kk=1:Num_noiseFloors
        dummy = NoiseFloors_IAC{kk};
        [pdumb, f] = pmtm(dummy(pp,1:round(Keep_H*fs)),2.5,[],fs);
        plot(f,pow2db(pdumb),'Color',[0 1 1])
    end
    [pIAC, f] = pmtm(Rev_IAC(pp,1:round(Keep_H*fs)),2.5,[],fs);
    plot(f,pow2db(pIAC),'b'),title(['IAC H(f)' num2str(pp)]), xlim([0 20])
    hold off
    if any(pp==Aud_channels)
        set(gca,'Color','y')
    end
end
    
figure()
for pp=1:32
    subplot(8,4,pp)
    hold on
    for kk=1:Num_noiseFloors
        dummy = NoiseFloors_ITD{kk};
        [pdumb, f] = pmtm(dummy(pp,1:round(Keep_H*fs)),2.5,[],fs);
        plot(f,pow2db(pdumb),'Color',[0 1 1])
    end
    [pITD,f] = pmtm(Rev_ITD(pp,1:round(Keep_H*fs)),2.5,[],fs);
    plot(f,pow2db(pITD),'b'),title(['ITD H(f)' num2str(pp)]),xlim([0 20])
    hold off
    if any(pp==Aud_channels)
        set(gca,'Color','y')
    end
end


figure(), hold on
Aud_channels=[31,32,5,26,8,9];
Rev_IAC_AudChan = mean(Rev_IAC(Aud_channels,:));
Rev_ITD_AudChan = mean(Rev_ITD(Aud_channels,:));
for kk = 1:Num_noiseFloors
    dummy = NoiseFloors_IAC{kk};
    dummy_AudChan = mean(dummy(Aud_channels,:));
    plot(t,dummy_AudChan,'r')
end
plot(t,Rev_IAC_AudChan,'b','linewidth',2), hold off, xlim([0 Keep_H]), title('Averaged Auditory Channels IAC H(t)'), xlabel('Time (sec)')

saveas(gcf,['DataFigs/'subj  data_date '/Fig' num2str(i) '.' num2str(k) '.png'])

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




% [pxx,f] =pmtm(r,2,[],4096);
% [pjj,f2] = pmtm(j,2,[],4096);
% figure,plot(f,pow2db(pxx),'b',f2,pow2db(pjj),'r'),title('pmtm(revcor)')
% 
% 
% [aa lags] = xcorr(Mseq_sig, Mseq_sig(end:-1:1));
% figure,plot(lags,aa)
% 
[pzz, f3] = pmtm(IAC_EEG_avg(32,:),2.5,[],4096);
figure,plot(f3,pow2db(pzz)),title('EEG spect IAC'),hold on
for i =1:Num_noiseFloors
    plot(f3,pow2db(NF_Powers_IAC(:,i)),'r')
end


[pzz, f3] = pmtm(ITD_EEG_avg(32,:),2.5,[],4096);
figure,plot(f3,pow2db(pzz)),title('EEG spect ITD')

save([subj 'EEGMseqProcessed.mat'],'Rev_IAC','Rev_ITD','NoiseFloors_IAC','NoiseFloors_ITD')

