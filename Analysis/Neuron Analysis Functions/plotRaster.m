function [] = plotRaster(spikes)
%spikes in NEL format (2 columns, 1st column is trial # and second is spike
%times)

figure()
plot(spikes(:,2),spikes(:,1),'.', 'MarkerEdgeColor',[0 0.5 0])
ylabel('Trial #'), xlabel('Time (seconds)')


end