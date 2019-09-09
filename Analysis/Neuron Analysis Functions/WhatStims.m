function [stim_descrps] = WhatStims(data)
%This functions takes in data in cell array format where each cell is a NEL
%picture and returns what stims were run in this set of data. 

stim_descrps = {};
for i = 1:length(data)
    data_i = data{i};
    if isfield(data_i.Stimuli, 'short_description')
        stim_descrps{numel(stim_descrps) +1} = data_i.Stimuli.short_description;
    end
end
stim_descrps = unique(stim_descrps);

end
