function [MEMs, MEM, elicitor, n] = analyzeMEMR(MEMfile)


load(MEMfile);

endsamps = ceil(stim.clickwin*stim.Fs*1e-3);

freq = linspace(200, 8000, 1024);
MEMband = [500, 2000];
ind = (freq >= MEMband(1)) & (freq <= MEMband(2));

n = stim.nLevels;
for k = 1:stim.nLevels
    fprintf(1, 'Analyzing level # %d / %d ...\n', k, stim.nLevels);
    temp = reshape(squeeze(stim.resp(k, :, 2:end, 1:endsamps)),...
        (stim.nreps-1)*stim.Averages, endsamps);
    tempf = pmtm(temp', 4, freq, stim.Fs)';
    resp_freq(k, :) = median(tempf, 1); %#ok<*SAGROW>
    

    blevs = k; % Which levels to use as baseline (consider 1:k)
    temp2 = squeeze(stim.resp(blevs, :, 1, 1:endsamps));
    
    if(numel(blevs) > 1)
        temp2 = reshape(temp2, size(temp2, 2)*numel(blevs), endsamps);
    end
    
    temp2f = pmtm(temp2', 4, freq, stim.Fs)';
    bline_freq(k, :) = median(temp2f, 1);
end



if(min(stim.noiseatt) == 6)
    elicitor = 94 - (stim.noiseatt - 6);
else
    elicitor = 94 - stim.noiseatt;
end
elicitor = elicitor(1:n);

MEM = pow2db(resp_freq ./ bline_freq);

% figure;
% 
% cols = getDivergentColors(n);
% cols(cols > 1) = 1;
% cols (cols < 0 ) = 1;

% axes('NextPlot','replacechildren', 'ColorOrder',cols);

smoothmem = true;
plotorig = false;
if smoothmem
    for k = 1:n
        MEMs(k, :) = sgolayfilt(MEM(k, :), 2, 65);
    end
else
    MEMs = MEM;
end

% semilogx(freq / 1e3, MEMs, 'linew', 2);
% xlim([0.25, 4]);
% ylim([-2, 1.5]);
% ticks = [0.25, 0.5, 1, 2, 4];
% set(gca, 'XTick', ticks, 'XTickLabel', num2str(ticks'),...
%     'box', 'off', 'FontSize', 24);
% legend(num2str(elicitor'), 'location', 'best');
% 
% 
% if plotorig
%     hold on; 
%     semilogx(freq / 1e3, MEM, '--', 'linew', 2);
% end
% xlabel('Frequency (kHz)', 'FontSize', 28);
% ylabel('\Delta Ear canal pressure (dB)', 'FontSize', 28);



% figure;
% plot(elicitor, mean(abs(MEM(:, ind)), 2) , 'ok-', 'linew', 2);
% hold on;
% xlabel('Elicitor Level (dB SPL)', 'FontSize', 28);
% ylabel('\Delta Ear Canal Pressure (dB)', 'FontSize', 28);
% set(gca,'FontSize', 28);











