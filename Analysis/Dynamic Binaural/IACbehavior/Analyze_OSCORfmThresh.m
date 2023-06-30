clear all
data_path = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/';
%Fig_path = '../../../../Figures/DynBin/';
Fig_path = '/media/ravinderjit/Data_Drive/Data/Figures/DynBin/';
%OL_path = '../../../../../Apps/Overleaf/Binaural Bash 2019/Figures/';
addpath(data_path)

Subjects = [{'S211'}, {'S001'}, {'S132'}, {'S205'}, {'S206'},{'S208'},{'S207'},{'S203'},{'S204'}];


FMs = [];
correctList = [];
ntrials = 20; 
respList = [];
for i =1:numel(Subjects)
     ThisData= load([data_path Subjects{i} '_OSCORfmThresh.mat']);
     FMs = horzcat(FMs,ThisData.FMs');
     correctList = horzcat(correctList,ThisData.correctList');
     respList = horzcat(respList,ThisData.respList');
end


if ~all(FMs(:,1) == FMs(:,end)) || ~all(correctList(:,1) == correctList(:,end))
    error('Yo, check this out')
end

FMs = FMs(:,1); %all the same so just taking one column
correctList = correctList(:,1); %all the same so just taking one column
FM_played = unique(FMs);

for j =1:numel(Subjects)
    for i = 1:numel(FM_played)
        Mask = FMs == FM_played(i);
        Accuracy(i,j) = sum(correctList(Mask) == respList(Mask,j)) / ntrials;  
    end
end


horzcat(FM_played, Accuracy)

figure,plot(log2(FM_played), Accuracy,'kx','linewidth',2),xlabel('IAC mod (Hz)') ,ylabel('Accuracy')
hold on,plot((log2(FM_played(1)):.1:log2(FM_played(end)))',1/3,'-or')
ylim([0,1.05])
set(gca,'XTick',log2(FM_played))
set(gca,'XTickLabel',{'5','10','20','40','80','160','320'})
% ax.XTick = log2(FM_played);
% ax.XTicklabel = FM_played;
% ['5','10','20','40','80','160','320'];

sem = std(Accuracy')/sqrt(numel(Subjects));



figure,errorbar(log2(FM_played),mean(Accuracy'),sem,'ko','linewidth',2)
hold on,plot((log2(FM_played(1)):.1:log2(FM_played(end)))',1/3,'*r','linewidth',2), hold off
set(gca,'XTick',log2(FM_played))
set(gca,'XTickLabel',{'5','10','20','40','80','160','320'})
xlabel('OSCOR FM (Hz)')
ylabel('Accuracy')
ylim([0,1.05]), xlim([2.2 8.4])
legend(['Performance ' char(177) ' SEM'],'Chance Level','location','northeast')
%set(gca,'fontsize',25)
% fig = gcf;
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 9 6];
% print('OscorFMbehavior','-dpng','-r0')

figure('Position',[1, 1, 1000, 637]) 
Individ =plot(log2(FM_played), Accuracy,'-x','Color',[0.8,0.8,0.8],'linewidth',2);
xlabel('OSCOR (Hz)') ,ylabel('Accuracy')
hold on, Chance = plot((log2(FM_played(1)):.1:log2(FM_played(end)))',1/3,'*r','linewidth',2);
PerfAvg = errorbar(log2(FM_played),mean(Accuracy'),sem,'k-o','linewidth',2);
set(gca,'XTick',log2(FM_played))
set(gca,'XTickLabel',{'5','10','20','40','80','160','320'})
set(gca,'TickLength',[0 0])
set(gca,'fontsize',15)
ylim([0,1.05]), xlim([2.2 8.4])
fig.PaperPosition = [0 0 9 6];
legend([PerfAvg, Individ(1), Chance(1)], {['Performance ' char(177) ' SEM'], 'Individuals' ,'Chance Level'},'location','southwest')
% print([Fig_path 'OscorFMbehavior'],'-depsc','-r0')
% print([Fig_path 'OscorFMbehavior'],'-dpng','-r0')
%print([OL_path 'OscorFMbehavior'],'-depsc','-r0')


figure('Position',[1, 1, 1000, 637]), hold on,%plot((0:1/fs:1.6)*1000,FittedCurve,'r','linewidth',2), hold on
boxplot(Accuracy',FM_played,'Colors','k', 'Symbol','+k')
xlabel('OSCOR(Hz)'), ylabel('Accuracy')
set(gca,'fontsize',15)
yticks([0,0.25, 0.5,0.75,1])
%set(findobj(gca,'type','line'),'linew',1.5)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 9 6];
print([Fig_path 'OscorFm_bp'],'-dsvg','-r0')

save([data_path 'OSCORfmThresh_processed.mat'],'FM_played','Accuracy','sem')













