function [data_stim] = Extract_StimData(data, descriptor)

data_stim = [];
for i = 1:length(data)
    data_i = data{i};
    if isfield(data_i, 'Stimuli') && isfield(data_i.Stimuli, 'short_description') && strcmp(data_i.Stimuli.short_description, descriptor)
        data_stim = [data_stim {data_i}];
    end
end

if isempty(data_stim)
    error('Check descriptor')
end
end
    
    