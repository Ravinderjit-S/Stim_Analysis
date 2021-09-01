%This script generates an example of a binaural transfer function from ITD
%and IAC from the nerve

clear
load('/media/ravinderjit/Data_Drive/Data/AuditoryNerve/DynBin/MseqAnalyzed_10.12.18.mat')
addpath('../../Neuron Analysis Functions')

data_IAC = data_NOSCOR{4};

fs = 48828.125;

H_imp = data_IAC.MSO.H_imp{2};
H_NF = data_IAC.MSO.H_NF{2};
Mseq_StepDur = data_IAC.Stimuli.MSeq_StepDuration;
Mseq_N = data_IAC.Stimuli.MSeq_N;
Mseq_samps = round(Mseq_StepDur*1e-3*fs)*(2^Mseq_N-1);

CF = round(data_IAC.tc.CFinterp);
Time_Impulse = 0.050;
H_imp = H_imp(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
phase = unwrap(angle(fft(H_imp)));
f_phase = fs*(0:length(phase)-1)/length(phase);
[H_f, f] = pmtm(H_imp,2.5,[],fs);

t=(0:length(H_imp)-1)./fs*1000; %time in ms

for m =1:numel(H_NF)
    H_NF{m} = H_NF{m}(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
end

t=(0:length(H_imp)-1)./fs*1000; %time in ms

grey = [126,126,126]/255;
fsz = 9;
figure, subplot(3,2,1), hold on, 
plot(t, H_imp,'k', 'linewidth',2), title(['IAC H(t) CF: ' num2str(CF) ' Hz']), xlabel('Time (ms)'), ylabel('Firing Rate / IAC')


for m =1:numel(H_NF)
    plot(t,H_NF{m},'Color',grey)
end
legend('Response','Noise Floor')
xlim([0,25])
yticks([0,2e4,4e4,6e4])
xticks([0,10,20])
set(gca,'fontsize',fsz)
set(gca, 'FontName','Arial')

subplot(3,2,3), hold on
semilogx(f,pow2db(H_f),'k','linewidth',2), 
title('H(f)'), xlabel('Frequency (Hz)'), xlim([0 600]), ylabel('Power (dB/Hz)'), ylim([10,70])
for m=1:numel(H_NF)
    [p_NF,f] = pmtm(H_NF{m},2.5,[],fs);
    semilogx(f,pow2db(p_NF),'Color',grey)
end
yticks([20,40,60])
hold off
set(gca,'fontsize',fsz)
set(gca, 'FontName','Arial')

subplot(3,2,5),
plot(f_phase,phase,'k','linewidth',2), title('Phase(f)'),
ylabel('Radians'), xlim([0 400]), xlabel('Frequency (Hz)')
xticks([0 100,200,300])
yticks([-20,-15,-10,-5,0])

set(gca,'fontsize',fsz)
set(gca, 'FontName','Arial')

data_ITD = data_MOVN{5};

H_imp = data_ITD.MSO.H_imp{2};
H_NF = data_ITD.MSO.H_NF{2};
Mseq_StepDur = data_ITD.Stimuli.MSeq_StepDuration;
Mseq_N = data_ITD.Stimuli.MSeq_N;
Mseq_samps = round(Mseq_StepDur*1e-3*fs)*(2^Mseq_N-1);

CF = round(data_ITD.tc.CFinterp);
Time_Impulse = 0.050;
H_imp = H_imp(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
phase = unwrap(angle(fft(H_imp)));
f_phase = fs*(0:length(phase)-1)/length(phase);
[H_f, f] = pmtm(H_imp,2.5,[],fs);

for m =1:numel(H_NF)
    H_NF{m} = H_NF{m}(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
end

subplot(3,2,2), hold on,
plot(t,H_imp,'k','linewidth',2), title(['ITD H(t) CF: ' num2str(CF) ' Hz']), xlabel('Time (ms)'), ylabel('Firing Rate / ITD')
yticks([0,2e4,4e4])
xticks([0,10,20])

for m =1:numel(H_NF)
    plot(t,H_NF{m},'Color',grey)
end
xlim([0,25])

set(gca,'fontsize',fsz)
set(gca, 'FontName','Arial')

subplot(3,2,4), hold on
semilogx(f,pow2db(H_f),'k','linewidth',2), 
title('H(f)'), xlabel('Frequency (Hz)'), xlim([0 600]), ylabel('Power (dB/Hz)'), ylim([10,70])
for m=1:numel(H_NF)
    [p_NF,f] = pmtm(H_NF{m},2.5,[],fs);
    semilogx(f,pow2db(p_NF),'Color',grey)
end
yticks([20,40,60])
hold off

set(gca,'fontsize',fsz)
set(gca, 'FontName','Arial')

subplot(3,2,6),
plot(f_phase,phase,'k','linewidth',2), title('Phase(f)'),
ylabel('Radians'), xlim([0 400]), xlabel('Frequency (Hz)')
xticks([0 100,200,300])
yticks([-10,-5,0])

set(gca,'fontsize',fsz)
set(gca, 'FontName','Arial')

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 8 8];
print('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/NerveExTF','-dsvg','-r0')
print('/media/ravinderjit/Data_Drive/Data/Figures/DynBin/NerveExTF','-depsc','-r0')









