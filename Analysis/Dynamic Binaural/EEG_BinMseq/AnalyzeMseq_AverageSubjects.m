% Look at some processed binaural EEG data mseq
clear
Subjects = [{'S001'},{'S132'},{'S203'},{'S204'},{'S205'},{'S206'},{'S207'},{'S208'}];
data_path = '/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Mats';
Fig_path = '/media/ravinderjit/Data_Drive/Data/Figures/DynBin';
addpath(data_path)
load('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Mseq_4096fs_compensated.mat')

fs = 4096;
Keep_H = 1; %time length of impulse response to keep
t = (0:length(Mseq_sig)-1)/fs;

% 
% if isprime(numel(Aud_channels))
%     plottxy=factor(numel(Aud_channels)+1);
% else
%     plottxy = factor(numel(Aud_channels));
% end
% plottingX = prod(plottxy(1:round(numel(plottxy)./2)));
% plottingY = prod(plottxy(round(numel(plottxy)./2)+1:end));
% figure()
% for pp= 1:numel(Aud_channels)
%     subplot(plottingX,plottingY,pp)
%     hold on
%     for kk=1:Num_noiseFloors
%         dummy = NoiseFloors_IAC{kk};
%         [pdumb, f] = pmtm(dummy(pp,1:round(Keep_H*fs)),2.5,[],fs);
%         plot(f,pow2db(pdumb),'Color',[0 1 1])
%     end
%     [pIAC, f] = pmtm(Rev_IAC(pp,1:round(Keep_H*fs)),2.5,[],fs);
%     plot(f,pow2db(pIAC),'b'),title(['IAC H(f)' num2str(Aud_channels(pp))]), xlim([0 20])
%     hold off
% end


% figure(), hold on
% [Rev_IAC_Hf, f] = pmtm(Rev_IAC(:,1:round(Keep_H*fs))',2.5,[],fs);
% plot(f,pow2db(Rev_IAC_Hf)),xlim([0 20])
% plot(f,mean(pow2db(Rev_IAC_Hf),2),'k','linewidth',2)
% Subj_Hf(:,1) = mean(pow2db(Rev_IAC_Hf),2);


UseElectrodes = 5; %index 6 is electrode 32

for i =1:numel(Subjects)
    subj = Subjects{i}
    load([subj '_DynBinMseqAnalyzed.mat'])
    Rev_IAC = Rev_IAC(UseElectrodes,:); Rev_ITD = Rev_ITD(UseElectrodes,:);
    Num_noiseFloors = size(NoiseFloors_IAC,2);
    [Rev_IAC_Hf, f] = pmtm(Rev_IAC(:,1:round(Keep_H*fs))',2.5,[],fs);
    SubjHf_IAC(:,i) = mean(pow2db(Rev_IAC_Hf),2);
    for kk = 1:Num_noiseFloors
        dummyIAC = NoiseFloors_IAC{kk}; dummyIAC = dummyIAC(UseElectrodes,:);
        [Rev_dummyHf , f] = pmtm(dummyIAC(:,1:round(Keep_H*fs))',2.5,[],fs);
        IAC_NoiseFloors(:,kk) = mean(pow2db(Rev_dummyHf),2);
    end
    Subj_IACnoisefloors(:,:,i) = IAC_NoiseFloors;
    figure(), hold on
    plot(f,IAC_NoiseFloors,'r'), plot(f,SubjHf_IAC(:,i),'b'),xlim([0 30]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)')
    title(['IAC: ' subj])

    [Rev_ITD_Hf, f] = pmtm(Rev_ITD(:,1:round(Keep_H*fs))',2.5,[],fs);
    SubjHf_ITD(:,i) = mean(pow2db(Rev_ITD_Hf),2);
    for kk = 1:Num_noiseFloors
        dummyITD = NoiseFloors_ITD{kk}; dummyITD = dummyITD(UseElectrodes,:);
        [Rev_dummyHf , f] = pmtm(dummyITD(:,1:round(Keep_H*fs))',2.5,[],fs);
        ITD_NoiseFloors(:,kk) = mean(pow2db(Rev_dummyHf),2);
    end
    Subj_ITDnoisefloors(:,:,i) = ITD_NoiseFloors;
    figure(), hold on
    plot(f,ITD_NoiseFloors,'r'), plot(f,SubjHf_ITD(:,i),'b'),xlim([0 30]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)')
    title(['ITD: ' subj])
end

AcrossSubjectNF_IAC = mean(Subj_IACnoisefloors,3);
AcrossSubjectNF_ITD = mean(Subj_ITDnoisefloors,3);

AcrossSubjectHf_IAC = mean(SubjHf_IAC,2);
AcrossSubjectHf_ITD = mean(SubjHf_ITD,2);

AcrossSubjectHf_IAC_SEM = 1.96*std(SubjHf_IAC,[],2)./sqrt(numel(Subjects));
AcrossSubjectHf_ITD_SEM = 1.96*std(SubjHf_ITD,[],2)./sqrt(numel(Subjects));


% figure(), hold on
% plot(f,AcrossSubjectHf_IAC,'k','linewidth',3), xlim([0 30]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)')
% Index_log = 2:length(f); %fill doesn't work on logscale if you include x data point of 0
% plotsem = fill([f(Index_log);flipud(f(Index_log))],[AcrossSubjectHf_IAC_SEMdown(Index_log);flipud(AcrossSubjectHf_IAC_SEMup)],'k','linestyle','none');
% set(plotsem,'facealpha',0.5)
% title('IAC Across Subjects')
% set(gca, 'XScale', 'log')

figure(), hold on
plot(f,AcrossSubjectHf_ITD,'b'), xlim([0 30]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)')
Index_log = 2:length(f); %fill doesn't work on logscale if you include x data point of 0
plotsem = fill([f(Index_log);flipud(f(Index_log))],[AcrossSubjectHf_ITD(Index_log)-AcrossSubjectHf_ITD_SEM(Index_log);flipud(AcrossSubjectHf_ITD(Index_log)+AcrossSubjectHf_ITD_SEM(Index_log))],'k','linestyle','none');
set(plotsem,'facealpha',0.5)
title('ITD Across Subjects')
set(gca,'XScale','log')


% figure(), hold on
% plot(f,AcrossSubjectNF_IAC,'r'), plot(f,AcrossSubjectHf_IAC,'b'), xlim([0 30]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)')
% title('IAC Across Subjects')
% 
% figure(), hold on
% plot(f,AcrossSubjectNF_ITD,'r'), plot(f,AcrossSubjectHf_ITD,'b'), xlim([0 30]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)')
% title('ITD Across Subjects')


NFmean_IAC = mean(AcrossSubjectNF_IAC,2);
NFstd_IAC = 1.96*std(AcrossSubjectNF_IAC,[],2);
NFmean_ITD = mean(AcrossSubjectNF_ITD,2);
NFstd_ITD = 1.96*std(AcrossSubjectNF_ITD,[],2);

Hf_IAC_6dBIndex = find(AcrossSubjectHf_IAC <= (max(AcrossSubjectHf_IAC) - 6),1);

figure(), hold on
plot(f,AcrossSubjectHf_IAC,'k','linewidth',5,'HandleVisibility','off'), xlim([0 20]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)')
Index_log = 2:length(f); %fill doesn't work on logscale if you include x data point of 0
plotsem = fill([f(Index_log);flipud(f(Index_log))],[AcrossSubjectHf_IAC(Index_log)-AcrossSubjectHf_IAC_SEM(Index_log);flipud(AcrossSubjectHf_IAC(Index_log)+AcrossSubjectHf_IAC_SEM(Index_log))],'k','linestyle','none');
set(plotsem,'facealpha',0.3)

plot(f,NFmean_IAC,'r','linewidth',5,'HandleVisibility','off')
plotstd = fill([f(Index_log);flipud(f(Index_log))],[NFmean_IAC(Index_log)-NFstd_IAC(Index_log);flipud(NFmean_IAC(Index_log)+NFstd_IAC(Index_log))],'r','linestyle','none');
set(plotstd,'facealpha',0.3)
legend('Mean w/ 95% CI','Noise Floor w/ 95% CI','location','southwest')

% title('IAC Across Subjects')
% legend('Power Function', 'NF Mean', 'NF+-2std')
set(gca,'fontsize',25)
set(gca,'XScale', 'log')
xticks([0:5:20])
ylim([-95 -60])
yticks([-90:10:-60])
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
%print('IAC_EEGPower','-dpng','-r0')



figure(), hold on
plot(f,AcrossSubjectHf_ITD,'b','linewidth',2), xlim([0 20]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)')



plot(f,NFmean_IAC,'r','linewidth',2)
plotstd = fill([f;flipud(f)],[NFmean_ITD-NFstd_ITD;flipud(NFmean_ITD+NFstd_ITD)],'r','linestyle','none');
set(plotstd,'facealpha',0.5)
% title('ITD Across Subjects')
% legend('Power Function', 'NF Mean', 'NF+-2std')
set(gca,'fontsize',25)
xticks([0:5:20])
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
%print('ITD_EEGPower','-dpng','-r0')


figure(), hold on
plot(f,AcrossSubjectHf_IAC-NFmean_IAC,'k','linewidth',5,'HandleVisibility','off'), xlim([0 20]), ylim([0 12]), xlabel('Frequency (Hz)')
ylabel('Response SNR (dB)')%ylabel('Response Power (dB/Hz)')
Index_log = 2:length(f); %fill doesn't work on logscale if you include x data point of 0
plotsem = fill([f(Index_log);flipud(f(Index_log))],[AcrossSubjectHf_IAC(Index_log)-AcrossSubjectHf_IAC_SEM(Index_log)-NFmean_IAC(Index_log);flipud(AcrossSubjectHf_IAC(Index_log)+AcrossSubjectHf_IAC_SEM(Index_log)-NFmean_IAC(Index_log))],'k','linestyle','none');
set(plotsem,'facealpha',0.5)

plotstd = fill([f(Index_log);flipud(f(Index_log))],[zeros(length(f(Index_log)),1);flipud(NFstd_IAC(Index_log))],'r','linestyle','none');
set(plotstd,'facealpha',0.5)
% title('IAC Across Subjects')
% legend('Response Power', 'CI>95%')
legend('Mean w/ 95% CI','Noise Floor w/ 95% CI','location','northeast')
set(gca,'fontsize',20)
set(gca,'XScale','log')
yticks([0:2:10])
xticks([0:5:20])
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
% print([Fig_path 'IAC_EEGResponsePower'],'-dpng','-r0')

figure(), hold on
plot(f,AcrossSubjectHf_ITD-NFmean_ITD,'k','linewidth',5), xlim([0 20]), ylim([0,7]), xlabel('Frequency (Hz)'), ylabel('Response Power (dB/Hz)')
plotsem = fill([f(Index_log);flipud(f(Index_log))],[AcrossSubjectHf_ITD(Index_log)-AcrossSubjectHf_ITD_SEM(Index_log)-NFmean_ITD(Index_log);flipud(AcrossSubjectHf_ITD(Index_log)+AcrossSubjectHf_ITD_SEM(Index_log)-NFmean_ITD(Index_log))],'k','linestyle','none');
set(plotsem,'facealpha',0.5)

plotstd = fill([f(Index_log);flipud(f(Index_log))],[zeros(length(f(Index_log)),1);flipud(NFstd_ITD(Index_log))],'r','linestyle','none');
set(plotstd,'facealpha',0.5)
% title('ITD Across Subjects')
% legend('Response Power', 'CI>95%')
set(gca,'fontsize',20)
set(gca,'XScale','log')
yticks([0:1:7])
xticks([0:5:20])
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
%print('ITD_EEGResponsePower','-dpng','-r0')





