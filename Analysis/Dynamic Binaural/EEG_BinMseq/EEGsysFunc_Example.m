% make example EEG figures
clear
load('/media/ravinderjit/Data_Drive/Data/EEGdata/DynamicBinaural/Mats/S207_DynBinMseqAnalyzed.mat')
load('/media/ravinderjit/Data_Drive/Stim_Analysis/Stimuli/Binaural_t/Stim Development/Mseq_4096fs_compensated.mat')
fs = 4096; %eeg fs
Keep_H = 1;

t = (0:length(Mseq_sig)-1)/fs;
Num_noiseFloors = numel(NoiseFloors_IAC);

grey = [126,126,126]/255;

figure, subplot(3,1,1), hold on
pp = 5; %electrode 31
[pIAC, f] = pmtm(Rev_IAC(pp,1:round(Keep_H*fs)),2.5,[],fs);
semilogx(f,pow2db(pIAC),'k','linewidth',3),title(['IAC H(f)' num2str(Aud_channels(pp))]), xlim([0 25]), ylim([-115 -55]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)'), title('H(f)')
for kk=1:Num_noiseFloors
    dummy = NoiseFloors_IAC{kk};
    [pdumb,f] = pmtm(dummy(pp,1:round(Keep_H*fs)),2.5,[],fs);
    semilogx(f,pow2db(pdumb),'Color',grey)
end
set(gca, 'XScale', 'log');
xticks([1,10,20])
xticklabels({'1','10','20'})
yticks([-100,-80,-60])
%set(gca,'fontsize',25)
set(gca, 'FontName','Arial')
hold off


subplot(3,1,3), hold on
plot(t,Rev_IAC(pp,:),'k','linewidth',3),title(['IAC H(t):' num2str(Aud_channels(pp))]), xlim([0 Keep_H]), ylabel('V / IAC'), title('H(t)'), xlabel('Time (sec)')
for kk = 1:Num_noiseFloors
    dummy = NoiseFloors_IAC{kk};
    plot(t,dummy(pp,:),'Color', grey)
end
xticks([0,0.5,1])
yticks([0, 3e-3, 6e-3])
hold off
%set(gca,'fontsize',25)
set(gca, 'FontName','Arial')

phase = unwrap(angle(fft(Rev_IAC(pp,1:round(Keep_H*fs)))));
f_phase = fs*(0:length(phase)-1)/length(phase);
cycles = phase/(2*pi);
subplot(3,1,2),plot(f_phase,phase-phase(2),'k','linewidth',3), xlim([1 10]), xlabel('Frequency (Hz)'), ylabel('Radians'), title('Phase(f)')
hold on
PhaseFit = polyfit(f_phase(2:8),phase(2:8),1);
GD_line = PhaseFit(1)*f_phase(2:8)+PhaseFit(2);
% plot(f_phase(2:8),GD_line)
GD = abs(PhaseFit(1)) ./ (2*pi);
xticks([1,5,10])
hold off
%set(gca,'fontsize',25)
set(gca, 'FontName', 'Arial')
box off
 
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 2 6];
print('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/IAC_EEGHex','-dpng','-r0')
print('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/IAC_EEGHex','-depsc','-r0')
print('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/IAC_EEGHex','-dsvg','-r0')

% figure()
% if isprime(numel(Aud_channels))
%     plottxy=factor(numel(Aud_channels)+1);
% else
%     plottxy = factor(numel(Aud_channels));
% end
% plottingX = prod(plottxy(1:round(numel(plottxy)./2)));
% plottingY = prod(plottxy(round(numel(plottxy)./2)+1:end));
% for pp = 1:numel(Aud_channels)
%     subplot(plottingX,plottingY,pp)
%     hold on
%     for kk = 1:Num_noiseFloors
%         dummy = NoiseFloors_IAC{kk};
%         plot(t,dummy(pp,:),'Color',[0 1 1])
%     end
%     plot(t,Rev_IAC(pp,:),'b'),title(['IAC H(t):' num2str(Aud_channels(pp))]), xlim([0 Keep_H])
%     hold off
% end
% 
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