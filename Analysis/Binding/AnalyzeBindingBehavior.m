%This script will analyze data from experiment Binding Behavior
clear
data_drive = '/media/ravinderjit/Data_Drive/';
data_loc = 'Data/BehaviorData/Binding/Pilot_Spring20';

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



    
    




