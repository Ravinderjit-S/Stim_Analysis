function [dataOUT]=correct_bad_lines(dataIN)
%%R.S: Modified on 09/25/18
if isempty(dataIN.Stimuli.bad_lines) || isfield(dataIN.Stimuli, 'bl_status')
    dataOUT = dataIN;
else
    bad_lines = dataIN.Stimuli.bad_lines(end:-1:1);
    dataOUT=dataIN;
    %Remove the bad lines
    lines=dataIN.spikes{1}(~ismember(dataIN.spikes{1}(:,1),bad_lines),1);
    spikes=dataIN.spikes{1}(~ismember(dataIN.spikes{1}(:,1),bad_lines),2);
    dataOUT.spikes{1}=[lines spikes];
    %Correct the remaining line numbers
    for i=1:length(bad_lines)
        dataOUT.spikes{1}(dataOUT.spikes{1}(:,1)>=bad_lines(i)+1,1)=...
            dataOUT.spikes{1}(dataOUT.spikes{1}(:,1)>=bad_lines(i)+1,1)-1;
    end
    dataOUT.Stimuli.fully_presented_lines = dataIN.Stimuli.fully_presented_lines -length(bad_lines); %correcting fully presented lines
    dataOUT.Stimuli.fully_presented_stimuli = dataIN.Stimuli.fully_presented_stimuli - length(bad_lines); %correcting fully presented stimuli
    dataOUT.Stimuli.bl_status = 'fixed'; 
end
return