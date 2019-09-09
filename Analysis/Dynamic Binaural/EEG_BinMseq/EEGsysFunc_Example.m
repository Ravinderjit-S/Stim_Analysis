% make example EEG figures
clear
load('S207_DynBinMseqAnalyzed.mat')
load('Mseq_4096fs_compensated.mat')
fs = 4096; %eeg fs
Keep_H = 1;

t = (0:length(Mseq_sig)-1)/fs;
Num_noiseFloors = numel(NoiseFloors_IAC);



figure, subplot(3,1,2), hold on
pp = 5; %electrode 31
[pIAC, f] = pmtm(Rev_IAC(pp,1:round(Keep_H*fs)),2.5,[],fs);
semilogx(f,pow2db(pIAC),'k','linewidth',3),title(['IAC H(f)' num2str(Aud_channels(pp))]), xlim([0 20]), ylim([-85 -55]), xlabel('Frequency (Hz)'), ylabel('Power (dB/Hz)'), title('H(f)')
for kk=1:Num_noiseFloors
    dummy = NoiseFloors_IAC{kk};
    [pdumb,f] = pmtm(dummy(pp,1:round(Keep_H*fs)),2.5,[],fs);
    semilogx(f,pow2db(pdumb),'Color',[1 0 0])
end
set(gca, 'XScale', 'log');
set(gca,'fontsize',25)
hold off


subplot(3,1,1), hold on
plot(t,Rev_IAC(pp,:),'k','linewidth',3),title(['IAC H(t):' num2str(Aud_channels(pp))]), xlim([0 Keep_H]), ylabel('V / IAC'), title('H(t)'), xlabel('Time (sec)')
for kk = 1:Num_noiseFloors
    dummy = NoiseFloors_IAC{kk};
    plot(t,dummy(pp,:),'Color',[1 0 0])
end
hold off
set(gca,'fontsize',25)

phase = unwrap(angle(fft(Rev_IAC(pp,1:round(Keep_H*fs)))));
f_phase = fs*(0:length(phase)-1)/length(phase);
cycles = phase/(2*pi);
subplot(3,1,3),plot(f_phase,phase-phase(2),'k','linewidth',3), xlim([1 10]), xlabel('Frequency (Hz)'), ylabel('Radians'), title('Phase(f)')
hold on
PhaseFit = polyfit(f_phase(2:8),phase(2:8),1);
GD_line = PhaseFit(1)*f_phase(2:8)+PhaseFit(2);
% plot(f_phase(2:8),GD_line)
GD = abs(PhaseFit(1)) ./ (2*pi)
hold off
set(gca,'fontsize',25)

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 18];
print('IAC_EEGHex','-dpng','-r0')


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