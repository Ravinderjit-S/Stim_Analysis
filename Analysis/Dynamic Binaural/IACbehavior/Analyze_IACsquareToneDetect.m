clear all
Fig_path = '../../../../Figures/DynBin/';
Subjects = [{'S001'},{'S132'},{'S205'},{'S206'},{'S207'},{'S208'},{'S203'},{'S204'},{'S211'}];
Data_path = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/';

responseTracks = {};
for i =1:numel(Subjects)      
    ThisData = load([Data_path Subjects{i} '_IACtSquareToneDetect.mat']);
    if strcmp(Subjects{i},'S132') %ran several extra conditions on this participant so removing those 
        getRid = [find(ThisData.WindowSizes==1000), find(ThisData.WindowSizes==1.2), find(ThisData.WindowSizes==2.4), find(ThisData.WindowSizes==3.2)];
        ThisData.WindowSizes(getRid) = [];
        ThisData.responseTracks(getRid) = [];
    end 
    [WindowSizes, sortedInds] = sort(ThisData.WindowSizes); ThisData.responseTracks = ThisData.responseTracks(sortedInds);
    responseTracks = horzcat(responseTracks, ThisData.responseTracks');
end

% for j = 1:numel(Subjects)
%     figure, hold on
%     plot(responseTracks{1,j},'b','linewidth',2)
%     plot(responseTracks{2,j},'r')
%     plot(responseTracks{3,j},'--b')
%     plot(responseTracks{4,j},'k')
%     plot(responseTracks{5,j},'g')
%     plot(responseTracks{6,j},'m')
%     plot(responseTracks{7,j},'c')
%     plot(responseTracks{8,j},'--k')
%     plot(responseTracks{9,j},'--b')
%     plot(responseTracks{10,j},'r','linewidth',2)
%     hold off
%     legend(num2str(WindowSizes(1)), num2str(WindowSizes(2)),num2str(WindowSizes(3)),num2str(WindowSizes(4)),num2str(WindowSizes(5)),num2str(WindowSizes(6)),num2str(WindowSizes(7)), ...
%         num2str(WindowSizes(8)),num2str(WindowSizes(9)),num2str(WindowSizes(10)))
%     title(Subjects{j}), xlabel('ResponseTracks')
% end

for j = 1:numel(Subjects)
    All_RevSNRs = [];
    for i = 1:size(responseTracks,1)
        changes = diff(responseTracks{i,j});
        change_indexes = find(changes~=0);
        Reversal_spots = find(diff(sign(changes(changes~=0))));
        Reversal_SNRs = responseTracks{i,j}(change_indexes(Reversal_spots+1));
        Reversal_SNRs = [Reversal_SNRs responseTracks{i,j}(end)]; %adding last reversal
        Reversal_indexes(:,i) = [change_indexes(Reversal_spots+1), length(responseTracks{i,j})];
        All_RevSNRs(:,i) = Reversal_SNRs; 
    end
    MedianSNR(:,j) = -median(All_RevSNRs(end-5:end,:)); % look at last 5 reversals
end


S211_SNR = MedianSNR(:,end);
AcrossSubjectsSNR = mean(MedianSNR,2);
AcrossSubjectsSEM = std(MedianSNR,[],2) ./ sqrt(numel(Subjects)); 

% Deal with Outlier Point
Outlier = isoutlier(MedianSNR(9,:),'grubbs'); %S132 was tired and gave up on this trial 
AcrossSubjectsSNR(9) = mean(MedianSNR(9,~strcmp(Subjects,'S132'))); %ignoring one data point from a subject
AcrossSubjectsSEM(9) = std(MedianSNR(9,~strcmp(Subjects,'S132'))) ./ sqrt(numel(Subjects)-1);


figure, hold on,%plot((0:1/fs:1.6)*1000,FittedCurve,'r','linewidth',2), hold on
errorbar(WindowSizes*1000, AcrossSubjectsSNR - AcrossSubjectsSNR(1), AcrossSubjectsSEM,'ko','linewidth',3), xlim([-0.1,1.63]*1000), ylim([-0.5 11])
plot(WindowSizes(9)*1000,MedianSNR(9,strcmp(Subjects,'S132')),'xr','MarkerSize',15,'linewidth',4)
set(gca,'XDir','reverse')

xlabel('Window Size (ms) '), ylabel('Detection Improvement (dB)')
legend(['Performance ' char(177) ' SEM'], 'outlier','location','SouthWest')
set(gca,'fontsize',25)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
% print('IACtsquareToneFig','-dpng','-r0')

figure('Position',[1, 1, 1000, 637]), hold on,%plot((0:1/fs:1.6)*1000,FittedCurve,'r','linewidth',2), hold on
Individ = plot(WindowSizes(2:end).^-1, MedianSNR(2:end,:)-repmat(MedianSNR(1,:),9,1),'Color',[0.8,0.8,0.8],'linewidth',1);
PerfAvg = errorbar(WindowSizes(2:end).^-1, AcrossSubjectsSNR(2:end) - AcrossSubjectsSNR(1), AcrossSubjectsSEM(2:end),'k-o','linewidth',3); ylim([0 13]), xlim([0.5 25])
%Outlier = plot(WindowSizes(9).^-1,MedianSNR(9,strcmp(Subjects,'S132'))-MedianSNR(1,strcmp(Subjects,'S132')),'xr','MarkerSize',15,'linewidth',4);
set(gca,'XScale','log')
set(gca,'XTick', [1 5 10 20])
xlabel('(1/WindowSize) Hz '), ylabel('Detection Improvement (dB)')
legend([PerfAvg,Individ(1)],{ ['Performance ' char(177) ' SEM'], 'Individuals'},'location','NorthEast')
set(gca,'fontsize',15)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 9 6];
% print([Fig_path 'IACtsquareToneFigfreq'],'-dpng','-r0')

% figure,plot(f_all2,WindowSizes,FitAcrossSubjects)

% figure, hold on
% plot(responseTracks{1,3},'r','linewidth',2)
% plot(responseTracks{10,3},'k','linewidth',2)
% ylabel('SNR (dB)'), xlabel('Trial')
% legend('Window Size = 0', 'Window Size = 1600 ms')
% set(gca,'fontsize',25)
% fig = gcf;
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 8 5];
% print('ExRespTracks','-dpng','-r0')


%save([Data_path 'IACsquareTone_Processed.mat'],'WindowSizes','AcrossSubjectsSNR','AcrossSubjectsSEM','S211_SNR')





