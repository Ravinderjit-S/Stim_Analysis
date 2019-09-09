% this script is to generate an example of a single unit transfer function
clear
load('MseqAnalyzed_10.12.18.mat')
addpath('../../Neuron Analysis Functions')
data = data_NOSCOR{4}; 
fs = 48828.125; %sampling rate of audio system
stim_per = data.Stimuli.Gating.Period;

figure,
subplot(3,2,1), plot(data.MSO.simulated{2}(:,2),data.MSO.simulated{2}(:,1),'.','MarkerEdgeColor', [0 0.5 0]), title('MSO via ANF coincidence'), ylabel('Trial'), xlabel('Time (sec)')
set(gca,'fontsize',25)
subplot(3,2,3), plot(data.MSO.LeftSpks{2}(:,2), data.MSO.LeftSpks{2}(:,1),'.','MarkerEdgeColor', [0 0.5 0]), title('Nerve Left')
set(gca,'fontsize',25)
subplot(3,2,5), plot(data.MSO.RightSpks{2}(:,2), data.MSO.RightSpks{2}(:,1),'.','MarkerEdgeColor', [0 0.5 0]), title('Nerve Right')
set(gca,'fontsize',25)
[t, FRate] = FiringRate(data.MSO.simulated{2},'bin',1,stim_per,0);
subplot(3,2,2), plot(t,FRate), ylabel('FiringRate (Hz)'), xlabel('Time (sec)')
set(gca,'fontsize',25)
[t, FRate] = FiringRate(data.MSO.LeftSpks{2},'bin',1,stim_per,0);
subplot(3,2,4), plot(t,FRate)
set(gca,'fontsize',25)
[t, FRate] = FiringRate(data.MSO.RightSpks{2},'bin',1,stim_per,0);
subplot(3,2,6), plot(t,FRate)
set(gca,'fontsize',25)

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 22 16];
print('Example_MSO','-dpng','-r0')

H_imp = data.MSO.H_imp{2};
H_NF = data.MSO.H_NF{2};
Mseq_StepDur = data.Stimuli.MSeq_StepDuration;
Mseq_N = data.Stimuli.MSeq_N;
Mseq_samps = round(Mseq_StepDur*1e-3*fs)*(2^Mseq_N-1);

CF = data.tc.CFinterp;
Time_Impulse = 0.100;
H_imp = H_imp(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
phase = unwrap(angle(fft(H_imp)));
f_phase = fs*(0:length(phase)-1)/length(phase);
[H_f, f] = pmtm(H_imp,2.5,[],fs);
Roll_off_6dB_index = find(pow2db(H_f) <= max(pow2db(H_f)) - 6,1,'first');

t=(0:length(H_imp)-1)./fs*1000; %time in ms

figure, subplot(3,1,1), hold on, plot(t, H_imp,'k', 'linewidth',3), title('H(t)'), xlabel('Time (ms)'), ylabel('Firing Rate / IAC')
for m =1:numel(H_NF)
    H_NF{m} = H_NF{m}(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
    plot(t,H_NF{m},'r')
end
xlim([0 20])
hold off
set(gca,'fontsize',25)


subplot(3,1,2), hold on, semilogx(f,pow2db(H_f),'k','linewidth',3), title('H(f)'), xlabel('Frequency (Hz)'), xlim([0 500]), ylabel('Power (dB/Hz)'), ylim([30,60])
semilogx(f(Roll_off_6dB_index),pow2db(H_f(Roll_off_6dB_index)),'bx','MarkerSize',20,'linewidth',3);
for m=1:numel(H_NF)
    [p_NF,f] = pmtm(H_NF{m},2.5,[],fs);
    semilogx(f,pow2db(p_NF),'r')
end
hold off, set(gca, 'XScale','log')
set(gca,'fontsize',25)

subplot(3,1,3),plot(f_phase,phase,'k','linewidth',3), title('Phase(f)'), ylabel('Radians'), xlim([0 300]), xlabel('Frequency (Hz)')
set(gca,'fontsize',25)

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 16];
print('IAC_Hex','-dpng','-r0')


t2 = 0:1/fs:Mseq_samps/fs-1/fs;
figure,plot(t2,data.Stimuli.IACt(1:Mseq_samps)), ylim([-1.1 1.1]), xlim([0 t2(end)+0.02]), xlabel('Time (ms)'), ylabel('IAC')
set(gca,'fontsize',25)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
print('Mseq','-dpng','-r0')


