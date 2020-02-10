% compare white noise oscor to filtere noise oscor
clear
Data_path = '/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/';
oscor_white = load('/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/OSCOR_white/S211_whitenoise_OSCORfmThresh.mat');
oscor_filt = load('/media/ravinderjit/Data_Drive/Data/BehaviorData/IACbehavior/S211_OSCORfmThresh.mat');


FMs = oscor_white.FMs;
ntrials = 20; 

FM_unq = unique(FMs);

for i = 1:numel(FM_unq)
    Accuracy_white(i) = sum(oscor_white.correctList(FMs==FM_unq(i)) == oscor_white.respList(FMs==FM_unq(i))) / ntrials;
    Accuracy_filt(i) = sum(oscor_filt.correctList(FMs==FM_unq(i)) == oscor_filt.respList(FMs==FM_unq(i))) / ntrials;
end

figure, hold on
plot(log2(FM_unq), Accuracy_white, 'b',log2(FM_unq), Accuracy_filt,'r', 'linewidth',2)
plot(log2(FM_unq(1)):.1:log2(FM_unq(end)),1/3,'*k','linewidth',2), hold off
set(gca,'XTick',log2(FM_unq))
set(gca,'XTickLabel',{'5','10','20','40','80','160','320'})
xlabel('OSCOR FM (Hz)')
ylabel('Accuracy')
ylim([0,1.05]), xlim([2.2 8.4])
legend('White','Filt','Chance','location','northeast')


save([Data_path 'OSCORwhite_processed.mat'],'Accuracy_white')
