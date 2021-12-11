clear

Data_path = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/';
Fig_path = '/media/ravinderjit/Data_Drive/Data/Figures/DynBin/';

sqtone = load([Data_path 'IACsquareTone_Processed.mat']);
oscor = load([Data_path 'OSCORfmThresh_processed.mat']);
oscor_white = load([Data_path 'OSCORwhite_processed.mat']);
phys_behMod = load('/media/ravinderjit/Data_Drive/Stim_Analysis/Analysis/Dynamic Binaural/Physio_behModel.mat');

sq_f = 1./(sqtone.WindowSizes(2:end));
SNRs = sqtone.AcrossSubjectsSNR(2:end) - sqtone.AcrossSubjectsSNR(1);
sq_SEM = sqtone.AcrossSubjectsSEM(2:end);

os_FM = oscor.FM_played;
os_sem = oscor.sem;
os_acc = mean(oscor.Accuracy');

os_acc_white = oscor_white.Accuracy_white; 

sq_color = [0.4940 0.1840, 0.5560];%[117,112,179]/255;
os_color = [27,158,119]/255;

os_Limit = 9.3;

fig = figure;
hold on
yyaxis left
sq = errorbar(sq_f, SNRs, sq_SEM,'Marker','o','MarkerSize',10, ...
    'Color',sq_color,'LineWidth',2);
ylim([0 10.4])
ylabel('Detection Improvement (dB)')
set(gca,'YColor', sq_color)
yyaxis right
ylabel('OSCOR Accuracy')
ylim([0.3,1.02])
os = errorbar(os_FM, os_acc,os_sem,'Marker','square','MarkerSize',10, ...
    'Color',os_color,'LineWidth',2);
os_white = plot(os_FM, os_acc_white, 'LineStyle',':','Color', os_color, ...
    'LineWidth',3);
plot([os_Limit os_Limit],[0.3 1.1],'--','Linewidth',2,'Color',sq_color)
%os_MOL = plot(os_Limit, 0.8,'p','MarkerSize',10,'Color',[0.4940 0.1840 0.5560],'LineWidth',2);
% os_MOL = line([os_Limit, os_Limit],[0.3,1],'LineStyle','--','Color',[0.4940 0.1840 0.5560], ...
%     'LineWidth',3);
xlabel('Frequency (Hz)')
set(gca,'XScale','log')
set(gca,'YColor', os_color)
set(gca,'XTick',[1,10,100])
set(gca,'XTickLabel',{'1','10','100'})
%set(gca,'fontsize',15)
xlim([0.5 350])
legend({'Bin Unmask','OSCOR', 'OSCOR white', 'OSCOR MOL'},'location','SouthWest')
legend('boxoff')
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 3.3 3.3];
print([Fig_path 'IACbehavior'],'-depsc')
print([Fig_path 'IACbehavior'],'-dsvg')

%% S211 figure

s211_oscor = oscor.Accuracy(:,1);
s211_sqtone = sqtone.S211_SNR;
s211_sqtone = s211_sqtone - s211_sqtone(1);
s211_sqtone = s211_sqtone(2:end);
figure
hold on
yyaxis left
plot(sq_f,s211_sqtone,'Color',sq_color,'linewidth',2)
ylim([0 10.4])
ylabel('Detection Improvement (dB)')
set(gca,'Ycolor',sq_color)
yyaxis right
ylabel('Detection Accuracy')
ylim([0.3 1.02])
plot(os_FM,s211_oscor,'linewidth',2,'Color',os_color)
plot(os_FM,os_acc_white,'LineStyle','--','Color',os_color, 'linewidth',2)
plot([os_Limit os_Limit],[0.3 1.1],'--','Linewidth',2,'Color',sq_color)
xlabel('Frequency (Hz)')
set(gca,'XScale','log')
set(gca,'YColor', os_color)
set(gca,'XTick',[1,10,100])
set(gca,'XTickLabel',{'1','10','100'})
xlim([0.5 350])
legend({'Bin Unmask','OSCOR', 'OSCOR white', 'OSCOR MOL'},'location','SouthWest')
legend('boxoff')
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 3.3 3.3];
print([Fig_path 'IACbehavior_s211'],'-dsvg')

%% S211 OSCOR Figure

s211_oscor = oscor.Accuracy(:,1);
figure, hold on
ylabel('Detection Accuracy')
ylim([0.3 1.02])
plot(os_FM,s211_oscor,'linewidth',2,'Color','k')
plot(os_FM,os_acc_white,'LineStyle','--','Color','k', 'linewidth',2)
xlabel('Frequency (Hz)')
set(gca,'XScale','log')
set(gca,'XTick',[10,100,300])
set(gca,'XTickLabel',{'10','100', '300'})
set(gca,'YTick',[0.3, 0.5, 0.75, 1.])
set(gca,'FontSize',14)
xlim([4 400])
legend({'OSCOR', 'OSCOR white'},'location','SouthWest')
legend('boxoff')
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 3.3 3.3];
print([Fig_path 'IACbehavior_OSCORs211'],'-dsvg')
print([Fig_path 'IACbehavior_OSCORs211'],'-depsc')

%% Just OSCOR figure

fig = figure;
hold on
ylabel('Detection Accuracy')
ylim([0.3,1.02])
os = errorbar(os_FM, os_acc,os_sem,'Marker','square','MarkerSize',10, ...
    'Color','k','LineWidth',2);
os_white = plot(os_FM, os_acc_white, 'LineStyle',':','Color', 'k', ...
    'LineWidth',3);
plot([os_Limit os_Limit],[0.3 1.1],'--','Linewidth',2,'Color','k')
xlabel('Frequency (Hz)')
set(gca,'XScale','log')
set(gca,'XTick',[1,10,100])
set(gca,'XTickLabel',{'1','10','100'})
%set(gca,'fontsize',15)
xlim([3 350])
yticks([0.3,0.5,0.75,1])
legend({'OSCOR', 'OSCOR white', 'OSCOR MOL'},'location','South')
legend('boxoff')
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 3.3 3.3];
print([Fig_path 'IACbehavior_oscor'],'-depsc')
print([Fig_path 'IACbehavior_oscor'],'-dsvg')

%% Just Bin Unmask Figure
SNRs = sqtone.AcrossSubjectsSNR(1:end) - sqtone.AcrossSubjectsSNR(1);
sq_SEM = sqtone.AcrossSubjectsSEM(1:end);

fig = figure;
hold on
errorbar(sqtone.WindowSizes(1:end), SNRs, sq_SEM,'Marker','o','MarkerSize',10, ...
    'Color','k','LineWidth',2);
plot(sqtone.WindowSizes(1:end),phys_behMod.PhysBehMod,'linewidth',2)
plot(sqtone.WindowSizes(1:end),phys_behMod.PhysBehMod,'o','color',[0, 0.4470, 0.7410],'MarkerSize',10,'LineWidth',2)
ylim([0 10.4])
ylabel('Detection Improvement (dB)')
xlabel('Window Size (Sec)')
%set(gca,'fontsize',15)
legend({'Beahvior', 'Physiology Fit'},'location','SouthEast')
legend('boxoff')
xlim([0,1.75])
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 3.3 3.3];
% set(gca,'fontsize',15)
print([Fig_path 'IACbehavior_BinUm'],'-depsc')
print([Fig_path 'IACbehavior_BinUm'],'-dsvg')

error = mean(abs(SNRs(2:end-1)' - phys_behMod.PhysBehMod(2:end-1)));







