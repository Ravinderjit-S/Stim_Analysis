clear all

Subjects = [{'S001'},{'S132'},{'S205'},{'S206'},{'S207'},{'S208'},{'S203'},{'S204'},{'S211'}];

responseTracks = {};
for i =1:numel(Subjects)      
    ThisData = load([Subjects{i} '_IACtSquareToneDetect.mat']);
    if strcmp(Subjects{i},'S132')
        getRid = [find(ThisData.WindowSizes==1000), find(ThisData.WindowSizes==1.2), find(ThisData.WindowSizes==2.4), find(ThisData.WindowSizes==3.2)];
        ThisData.WindowSizes(getRid) = [];
        ThisData.responseTracks(getRid) = [];
    end 
    [WindowSizes, sortedInds] = sort(ThisData.WindowSizes); ThisData.responseTracks = ThisData.responseTracks(sortedInds);
    responseTracks = horzcat(responseTracks, ThisData.responseTracks');
end

for j = 1:numel(Subjects)
    figure, hold on
    plot(responseTracks{1,j},'b','linewidth',2)
    plot(responseTracks{2,j},'r')
    plot(responseTracks{3,j},'--b')
    plot(responseTracks{4,j},'k')
    plot(responseTracks{5,j},'g')
    plot(responseTracks{6,j},'m')
    plot(responseTracks{7,j},'c')
    plot(responseTracks{8,j},'--k')
    plot(responseTracks{9,j},'--b')
    plot(responseTracks{10,j},'r','linewidth',2)
    hold off
    legend(num2str(WindowSizes(1)), num2str(WindowSizes(2)),num2str(WindowSizes(3)),num2str(WindowSizes(4)),num2str(WindowSizes(5)),num2str(WindowSizes(6)),num2str(WindowSizes(7)), ...
        num2str(WindowSizes(8)),num2str(WindowSizes(9)),num2str(WindowSizes(10)))
    title(Subjects{j}), xlabel('ResponseTracks')
end

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


figure,plot(WindowSizes,MedianSNR,'x'), xlim([0,1.8])
legend(Subjects)


ft = fittype('Integrator(x,A,tau)');

for j = 1:numel(Subjects)
    FitInt = MedianSNR(:,j) - MedianSNR(1,j); %subtract windowsize = 0 for values to be amount benefit

    if strcmp(Subjects{j}, 'S132')
        WindowSizes2 = WindowSizes; WindowSizes2(WindowSizes==0.8) = []; FitInt(WindowSizes==0.8) = [];A_FitInt{j} =FitInt;
        A_FitInt{j} = FitInt;
        [f{j},gf{j},o{j}] = fit(WindowSizes2',FitInt,ft,'StartPoint',[1,0.1]);
    else
        A_FitInt{j} = FitInt;
        [f{j},gf{j},o{j}] = fit(WindowSizes',FitInt,ft,'StartPoint',[1,0.1]);
    end
    A(j) = f{j}.A;
    Taus(j) = f{j}.tau;
end

figure,plot(f{1},WindowSizes,A_FitInt{1})
figure,plot(f{2},WindowSizes2,A_FitInt{2})
figure,plot(f{3},WindowSizes,A_FitInt{3})
figure,plot(f{4},WindowSizes,A_FitInt{4})


AcrossSubjectsSNR = mean(MedianSNR,2);
AcrossSubjectsSEM = std(MedianSNR,[],2) ./ sqrt(numel(Subjects)); 

% Deal with Outlier Point
Outlier = isoutlier(MedianSNR(9,:),'grubbs'); %S132 was tired and gave up on this trial 
AcrossSubjectsSNR(9) = mean(MedianSNR(9,~strcmp(Subjects,'S132'))); %ignoring one data point from a subject
AcrossSubjectsSEM(9) = std(MedianSNR(9,~strcmp(Subjects,'S132'))) ./ sqrt(numel(Subjects)-1);


FitAcrossSubjects = AcrossSubjectsSNR - AcrossSubjectsSNR(1);

[f_all, gf_all, o_all] = fit(WindowSizes',FitAcrossSubjects,ft,'StartPoint', [FitAcrossSubjects(end),0.1]);


figure,plot(f_all,WindowSizes,FitAcrossSubjects), title('Across Subjects'), hold on
errorbar(WindowSizes, AcrossSubjectsSNR - AcrossSubjectsSNR(1), AcrossSubjectsSEM,'b.'), xlim([0,1.63])
legend('Data','Exp Fit')


ft2 = fittype('BMLDFunc(x,TnuTno,T)');
ft3 = fittype('Integrator2(x,A,B,C,tau1,tau2)');

TnuTno = db2pow(AcrossSubjectsSNR(end)-AcrossSubjectsSNR(1));
[f_all, gf_all, o_all] = fit(WindowSizes', FitAcrossSubjects,ft2,'lower',[0.01,TnuTno],'upper',[1,TnuTno],'StartPoint', [TnuTno,0.1]);

fs = 4096;
FittedCurve = BMLDFunc(0:1/fs:1.6,f_all.TnuTno,f_all.T);


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
print('IACtsquareToneFig','-dpng','-r0')

figure, hold on,%plot((0:1/fs:1.6)*1000,FittedCurve,'r','linewidth',2), hold on
errorbar(WindowSizes(2:end).^-1, AcrossSubjectsSNR(2:end) - AcrossSubjectsSNR(1), AcrossSubjectsSEM(2:end),'ko','linewidth',3), ylim([-0.5 11]), xlim([0.5 25])
plot(WindowSizes(9).^-1,MedianSNR(9,strcmp(Subjects,'S132')),'xr','MarkerSize',15,'linewidth',4)
set(gca,'XScale','log')
set(gca,'XTick', [1 5 10 20])
xlabel('(1/WindowSize) Hz '), ylabel('Detection Improvement (dB)')
legend(['Performance ' char(177) ' SEM'], 'outlier','location','SouthWest')
set(gca,'fontsize',25)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
print('IACtsquareToneFigfreq','-dpng','-r0')

% [f_all2, gf_all2, o_all2] = fit(WindowSizes',FitAcrossSubjects,ft3,'lower',[0.1,0,0,0,0],'upper',[20,1,1,1,1],'StartPoint',[FitAcrossSubjects(end),0.5,0.5,0.1,0.1]);

% figure,plot(f_all2,WindowSizes,FitAcrossSubjects)

figure, hold on
plot(responseTracks{1,3},'r','linewidth',2)
plot(responseTracks{10,3},'k','linewidth',2)
ylabel('SNR (dB)'), xlabel('Trial')
legend('Window Size = 0', 'Window Size = 1600 ms')
set(gca,'fontsize',25)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 8 5];
print('ExRespTracks','-dpng','-r0')









