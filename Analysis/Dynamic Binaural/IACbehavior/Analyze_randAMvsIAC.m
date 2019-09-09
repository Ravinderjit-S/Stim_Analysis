% analyze 3AFC SAM noise, OSCOR 
clear all
subjects = {'BM'};
All_Accuracy = [];
figure, hold on
for subjs = 1:numel(subjects)
    load([subjects{subjs} '_randAMvsIAC.mat'])
    uFMs = unique(FMs);
    
    Accuracy = [];
    
    for i = 1:length(uFMs)
        ThisCondition = FMs == uFMs(i);
        CondAcc = sum(correctList(ThisCondition) == respList(ThisCondition)) ./ ntrials;
        Accuracy = [Accuracy CondAcc];
    end
    plot(uFMs, Accuracy, 'x','Color',[rand rand rand]), ylim([0 1.1])
    All_Accuracy = [All_Accuracy; Accuracy];
end



