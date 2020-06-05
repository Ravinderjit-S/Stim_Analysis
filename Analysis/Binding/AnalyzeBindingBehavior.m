%This script will analyze data from experiment Binding Behavior
clear
data_drive = '/media/ravinderjit/Data_Drive/';
data_loc = 'Data/BehaviorData/Binding/Pilot_Spring20';
FigPath = '/media/ravinderjit/Data_Drive/Data/Figures/BindingPilot/';


subjects = {'S132','S227','S228','S229','S230'};


trials = 20;
accuracy = nan(10,numel(subjects));

for j = 1:numel(subjects)
    load([data_drive data_loc '/' subjects{j} '_BindingBehavior.mat'])
    for i =1:numel(Corr_inds)
        mask_cond = CorrSet ==i;
        accuracy(i,j) = sum(respList(mask_cond)==correctList(mask_cond)) / trials;
    end
end


% 1 = 1,2
% 2 = 1:4
% 3 = 1:6
% 4 = 1:8
% 5 = 15:16
% 6 = 13:16
% 7 = 11:16
% 8 = 9:16
% 9 = [1,6,11,16]
% 10 = [1,4,7,10,13,16]

Low_f = [1,2,3,4];
High_f = [5,6,7,8];
lh_diff = abs(accuracy(Low_f,:) - accuracy(High_f,:));
lh_avg = (accuracy(Low_f,:) + accuracy(High_f,:)) / 2;

% figure, hold on
% plot(accuracy(Low_f,:),'r','linewidth',2)
% plot(accuracy(High_f,:),'b','linewidth',2)
% ylabel('Accuracy')
% xlabel('# of Comodulated Tones')
% xticks([1, 2, 3, 4])
% xticklabels({'2','4','6','8'})

figure, hold on
l = plot(lh_avg,'k','linewidth',2);
h = plot(lh_diff,'k--','linewidth',2);
ylabel('Accuracy')
xlabel('# of Comodulated Tones')
legend([l(1),h(1)], 'Low High Avg', 'Low High Diff','location','Best')
legend boxoff
xticks([1, 2, 3, 4])
xticklabels({'2','4','6','8'})
yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
set(gca,'fontname','Arial')
set(gca,'fontsize',11)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 3.5 3];
print([FigPath 'NumTones'],'-dpng','-r0')

figure, hold on
barplt = [mean(lh_avg(2,:)), mean(accuracy(9,:)); ...
    mean(lh_avg(3,:)), mean(accuracy(10,:)) ];
stderr = [std(lh_avg(2,:)), std(accuracy(9,:)); ...
    std(lh_avg(3,:)), std(accuracy(10,:)) ] / sqrt(size(accuracy,2));
bar(barplt)
errorbar([0.85 1.15; 1.85 2.15],barplt,stderr,'.k')
yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
xticks([1,2])
ylim([0,1])
xticklabels({'4','6'})
xlabel('# of Comodulated Tones')
ylabel('Accuracy')
set(gca,'fontname','Arial')
set(gca,'fontsize',11)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 3.5 2];
print([FigPath 'ERBspacing'],'-dpng','-r0')









