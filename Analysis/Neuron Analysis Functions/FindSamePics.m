function [SamePics, SameStim, SameUnit] = FindSamePics(data)
% this function will find data (pics) takn on the same fiber with the same
% stimulus. sometimes one may run a certain stimuli on a fiber and then
% stop, run a different stimuli and come back and run the other stimuli. To
% analyze all that data together, this function will find those cases where
% stimuli were run in seperate pics on a fiber
% SamePics: pictures that are the same stimuli
% SameStim: Stimuli that was repeated
% SameUnit: Unit on which it happened

SameStim = {}; SamePics = {}; SameUnit = [];
units = [];
StimDescs = {};
PicNums = [];

for i = 1:numel(data)
    if isfield(data{i},'General') && isfield(data{i},'Stimuli')
        if isfield(data{i},'CalibData')
            continue
        end
        unit = data{i}.General.track + data{i}.General.unit/100;
        units = [units unit];
        if isfield(data{i},'TcData') % check for tuning curves
            StimDesc = 'TC';
        else
            StimDesc = data{i}.Stimuli.short_description;
        end
        StimDescs = [StimDescs {StimDesc}];
        PicNums = [PicNums data{i}.General.picture_number];
    end
end

Unique_units = unique(units);
for j = 1:numel(Unique_units)
    Unit_stims = StimDescs(units==Unique_units(j));
    Unit_pics = PicNums(units==Unique_units(j));
    Unique_unit_stims = unique(Unit_stims);
    for k = 1:numel(Unique_unit_stims)
        if sum(strcmp(Unique_unit_stims{k},Unit_stims))>1 % see if there are any duplicate stims 
            SameStim = [SameStim Unique_unit_stims(k)];
            SamePics = [SamePics {Unit_pics(strcmp(Unique_unit_stims{k},Unit_stims))}];
            SameUnit = [SameUnit Unique_units(j)];
        end
    end
end

end