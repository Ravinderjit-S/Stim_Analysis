clearvars 
dur = 1;
fs = 48828.125;

BPfilt = designfilt('bandpassfir', 'StopbandFrequency1', 100, 'PassbandFrequency1', 200, 'PassbandFrequency2', 1500, 'StopbandFrequency2', 2000, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60, 'SampleRate', fs);

fm = 20;
stim = IACsinAFC3(dur,fs,fm,BPfilt);

FigPath = '../../../../Figures/DynBin/';

Grpdly = grpdelay(BPfilt);Grpdly = round(Grpdly(1));
StimOscor = stim{3};
t = 0:1/fs:length(stim{1}(1,:))/fs-1/fs;
A = cos(2*pi*fm.*t);
t = t*1000;
fig =figure; 
left_color = [0 0 0];
right_color = [0.4 0.4 0.4];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);
FlatIAC = zeros(1,length(A));

yyaxis left, ylim([-1 1]), ylabel('IAC'), yticks([-1 0 1])
plot(t(Grpdly:end)-t(Grpdly),A(1:end-Grpdly+1),'k','linewidth',1)
yyaxis right, ylim([-.9 .9]), ylabel('Signal Amplitude')
hold on
plot(t(Grpdly:end)-t(Grpdly),StimOscor(1,Grpdly:end),'b','linewidth',1)
plot(t(Grpdly:end)-t(Grpdly),StimOscor(2,Grpdly:end),'r-','linewidth',1)
hold off
xlim([0 1/fm]*1000)
legend('IAC','Left Stim', 'Right Stim','location','Northeast')
xlabel('Time (ms)')
%set(gca,'fontsize',25)
fig = gcf;
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 11 8];
print([FigPath 'OSCOR'],'-depsc')