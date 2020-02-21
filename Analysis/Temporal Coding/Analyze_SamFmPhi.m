%Analyze data from 3AFC SAM, phi difference experiment 
clear
data_loc = '/media/ravinderjit/Data_Drive/Data/BehaviorData/TemporalCoding/';
subj = 'S211_SamFm_phi_block3';
load([data_loc, subj, '.mat'])

mods = params.mod;
phis = params.phi;

phis_unique = unique(phis); 
mods_unique = unique(mods);

accuracy = nan(numel(phis_unique),numel(mods_unique));

mods = mods(1:length(respList)); %if have not run all blocks only look at part that have run
phis = phis(1:length(respList));

for j = 1:length(mods_unique)
    for k = 1:length(phis_unique)
        cond = mods == mods_unique(j) & phis == phis_unique(k);
        accuracy(k,j) = sum(respList(cond) == correctList(cond)) / sum(cond);
    end
end

for i = 1:length(phis_unique)
    phaseLabel{i} = num2str(phis_unique(i));
end


figure()
plot(1:numel(phis_unique),accuracy,'linewidth',2)
hold on
plot(1:.1:numel(phis_unique), 1/3, '*r','linewidth',2), hold off
legend([num2str(mods_unique(1)) ' hz'], [num2str(mods_unique(2)) ' hz'], ...
    [num2str(mods_unique(3)) ' hz'], [num2str(mods_unique(4)) ' hz'], ...
    [num2str(mods_unique(5)) ' hz'], 'Chance','location','NorthWest')
set(gca,'XTickLabel',phaseLabel)
xlabel('phase (degrees)')
ylabel('accuracy')
xlim([0.2, numel(phis_unique) + 0.2])
ylim([0, 1])
