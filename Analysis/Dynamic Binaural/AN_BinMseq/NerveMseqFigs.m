clear


datapath = '/media/ravinderjit/Data_Drive/Data/AuditoryNerve/DynBin/';
figpath = '/media/ravinderjit/Data_Drive/Data/Figures/DynBin/';


ITDdata = load([datapath 'IACnerveMseq_figData']);
IACdata = load([datapath 'ITDnerveMseq_figData']);


IAC_cf = IACdata.All_CF_interp2;
IAC_gd = IACdata.All_GroupDelay2*1000;
IAC_rolloff = IACdata.All_Roll_off_Freq2;

ITD_cf = ITDdata.All_CF_interp2;
ITD_gd = ITDdata.All_GroupDelay2*1000;
ITD_rolloff = ITDdata.All_Roll_off_Freq2;


figure, hold on
plot(IAC_cf,IAC_rolloff,'ko','MarkerSize',10,'linewidth',2)
plot(ITD_cf,ITD_rolloff,'kx','MarkerSize',10,'linewidth',2)
ylabel('Roll off Frequency (-6dB)')
xlabel('CF (Hz)')
ylim([100 275])
yticks([100:50:275])
xticks([500:500:2500])
set(gca,'fontsize',15)
print([figpath 'CFvsRolloff'],'-depsc')

figure, hold on
plot(IAC_cf, IAC_gd,'ko','MarkerSize',10,'linewidth',2)
plot(ITD_cf,ITD_gd,'kx','MarkerSize',10,'linewidth',2)
ylabel('Group Delay (ms)')
xlabel('CF (Hz)')
xticks([500:500:2500])
legend('IAC','ITD')
legend('boxoff')
set(gca,'fontsize',15)
print([figpath 'CFvsGD'],'-depsc')

