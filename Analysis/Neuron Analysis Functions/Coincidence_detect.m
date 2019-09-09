function [MSO] = Coincidence_detect(spkL, spkR, maxISI)
%spkL and spkR are trials of spikes in NEL format (2 columns, first has
%trial # and second has spike times
%maxISI = max interspike interval for conincidence detection (in seconds)
%MSO = output of coincidence detector (MSO simulation) in NEL format


%This 'if' statement ensures spkL contains the longer number of trials 
if spkR(end,1) > spkL(end,1) 
    temp = spkL;
    spkL = spkR;
    spkR = temp;
end

MSO = [];
for i = 1:spkL(end,1)
    spikeL = spkL(spkL(:,1)==i,2);
    for j = 1:spkR(end,1)
         spikeR = spkR(spkR(:,1)==j,2);
         ISI = spikeL - spikeR';
         [Lind, Rind] = find(abs(ISI) <= maxISI);
         MSOij_spikes = max(horzcat(spikeL(Lind),spikeR(Rind)),[],2);
         MSOij_NEL = horzcat(((i-1)*spkR(end,1)+j)*ones(length(MSOij_spikes),1),MSOij_spikes); %NEL Format
         MSO = vertcat(MSO,MSOij_NEL);
%          figure,plot(MSOij_spikes, 3*ones(1,length(MSOij_spikes)),'r.'), hold on
%          plot(spikeL, ones(1,length(spikeL)),'b.')
%          plot(spikeR, 2*ones(1,length(spikeR)),'b.')
%          
         
    end
end


