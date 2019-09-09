% analyze 3AFC SAM noise, OSCOR 
clear all
% load('test2_AMvsIAC.mat')
% load('test_longrisetime_AMvsIAC.mat')
load('S211_AMvsIAC.mat')
uDepths = unique(Depths);
uFMs = unique(FMs);
Accuracy = []; 

for i = 1:length(uFMs)
    for j =1:length(uDepths)
        ThisCondition = Depths == uDepths(j) & FMs == uFMs(i);
        CondAcc = sum(correctList(ThisCondition) == respList(ThisCondition)) ./ ntrials;
        Accuracy = [Accuracy CondAcc];
    end
end


figure, plot(repmat(uFMs,4,1)',reshape(Accuracy,4,4)'), xlim([0,45])
hold on, plot(repmat(uFMs,4,1),reshape(Accuracy,4,4),'kx')
legend(num2str(uDepths'))
xlabel('OSCOR/AM Freq (Hz)')
ylabel('Accuracy (20 trials)')



