function [spk_cond] = ExtractConditions(datai)
%This function will extract conditions based on the length of attens so #of
%conditions should be = to length(attens). Data assumed to be in NEL format
%and all conditions are run before they are repeated
%datai = structrue in NEL format
%spk_cond = cell array with each cell containing the spikes for each condition

attens = datai.Stimuli.attens;
conds = length(attens);
Tot_Lines = datai.Stimuli.fully_presented_lines;
spikes = datai.spikes{:};
spk_cond = cell(1,conds);
for i =1:conds
    cond_lines = i:conds:Tot_Lines;
    lines_cond = spikes(ismember(spikes(:,1),cond_lines),1);
    for j =1:numel(cond_lines) %change line numbers to go from 1 to number of lines of that condition
        lines_cond(lines_cond==cond_lines(j),:) = j;
    end
    spikes_cond = spikes(ismember(spikes(:,1),cond_lines),2);
    
    spk_cond{i} = horzcat(lines_cond,spikes_cond);
end
end
