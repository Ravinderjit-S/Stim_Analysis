%Analyze data from 3AFC SAM, phi difference experiment 
clear
data_loc = '/media/ravinderjit/Data_Drive/Data/BehaviorData/TemporalCoding/';
subj = 'S211_SamFm_phi_aBlocks';
load([data_loc, subj, '.mat'])

mods = params.mod;
phis = params.phi;

phis_unique = unique(phis); 
mods_unique = unique(mods);

Amods_mask = zeros(1,length(mods));
Fmods_mask = zeros(1,length(mods));
for i = 1:length(params.modType)
    if params.modType(i) == 'A'
        Amods_mask((i-1)*100+1:(i)*100) = 1;
    else
        Fmods_mask((i-1)*100+1:(i)*100) = 1;
    end
end

accuracy_AM = nan(numel(phis_unique),numel(mods_unique));
accuracy_FM = nan(numel(phis_unique),numel(mods_unique));

%if have not run all blocks only look at part that have run
mods = mods(1:length(respList)); 
phis = phis(1:length(respList));
Amods_mask = Amods_mask(1:length(respList));
Fmods_mask = Fmods_mask(1:length(respList));

for j = 1:length(mods_unique)
    for k = 1:length(phis_unique)
        cond_AM = mods == mods_unique(j) & Amods_mask & phis == phis_unique(k);
        cond_FM = mods == mods_unique(j) & Fmods_mask & phis == phis_unique(k);
        accuracy_AM(k,j) = sum(respList(cond_AM) == correctList(cond_AM)) / sum(cond_AM);
        accuracy_FM(k,j) = sum(respList(cond_FM) == correctList(cond_FM)) / sum(cond_FM);
    end
end

for i = 1:length(phis_unique)
    phaseLabel{i} = num2str(phis_unique(i));
end


figure()
plot(1:numel(phis_unique),accuracy_AM,'linewidth',2)
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
title('AM')

figure()
plot(1:numel(phis_unique),accuracy_FM,'linewidth',2)
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
title('FM')







