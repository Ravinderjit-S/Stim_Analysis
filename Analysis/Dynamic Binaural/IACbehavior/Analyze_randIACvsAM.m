% analyze 3AFC SAM noise, OSCOR 
clear all

load('S211_randAMvsIAC.mat')
uFMs = unique(FMs);
Accuracy = []; 

for i = 1:length(uFMs)
    Accuracy(i) = sum(correctList(FMs == uFMs(i)) == respList(FMs == uFMs(i))) / ntrials;
end

figure,plot(uFMs,Accuracy,'x')
hold on, plot(uFMs,Accuracy)
xlabel('OSCOR/AM freq')
ylabel('Accuracy (20Trials)')
title('Randomish Depth')






