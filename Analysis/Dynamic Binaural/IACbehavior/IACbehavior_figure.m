clear

Data_path = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/';
Fig_path = '/media/ravinderjit/Data_Drive/Data/Figures/DynBin/';

sqtone = load([Data_path 'IACsquareTone_Processed.mat']);
oscor = load([Data_path 'OSCORfmThresh_processed.mat']);
oscor_white = load([Data_path 'OSCORwhite_processed.mat']);

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
ylabel('Oscor Accuracy')
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










