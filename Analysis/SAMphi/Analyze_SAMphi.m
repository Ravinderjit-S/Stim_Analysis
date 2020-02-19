%Analyze data from 3AFC SAM, phi difference experiment 
clear
data_loc = '/media/ravinderjit/Data_Drive/Data/BehaviorData/SAMphi/';
subj = 'S211_di_SAM_phi';
load([data_loc, subj, '.mat'])


phis_unique = unique(phis); 
AMs_unique = unique(AMs);

accuracy = nan(numel(phis_unique),numel(AMs_unique));

for j = 1:length(AMs_unique)
    for k = 1:length(phis_unique)
        cond = AMs == AMs_unique(j) & phis == phis_unique(k);
        accuracy(k,j) = sum(respList(cond) == correctList(cond)) / ntrials;
    end
end

for i = 1:length(phis_unique)
    phaseLabel{i} = num2str(phis_unique(i));
end


figure()
plot(1:numel(phis_unique),accuracy,'linewidth',2)
hold on
plot(1:.1:numel(phis_unique), 1/3, '*r','linewidth',2), hold off
legend([num2str(AMs_unique(1)) ' hz'], [num2str(AMs_unique(2)) ' hz'], 'Chance','location','NorthWest')
set(gca,'XTickLabel',phaseLabel)
xlabel('phase (degrees)')
ylabel('accuracy')
xlim([0.2, numel(phis_unique) + 0.2])
ylim([0, 1])
