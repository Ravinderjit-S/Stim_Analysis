clear

path = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/MEMR/';
Subject = 'S270';
file = 'MEMR_S270_R_04-Nov-2021_18_28_22.mat';
MEMfile = [path, Subject, '/', file];


freq = linspace(200, 8000, 1024);
MEMband = [500, 2000];
[MEMs, MEM, elicitor, n] = analyzeMEMR(MEMfile);
ind = (freq >= MEMband(1)) & (freq <= MEMband(2));

figure;

cols = getDivergentColors(n);
cols(cols > 1) = 1;
cols (cols < 0 ) = 1;

axes('NextPlot','replacechildren', 'ColorOrder',cols);


semilogx(freq / 1e3, MEMs, 'linew', 2);
xlim([0.25, 4]);
ylim([-2, 1.5]);
ticks = [0.25, 0.5, 1, 2, 4];
set(gca, 'XTick', ticks, 'XTickLabel', num2str(ticks'),...
    'box', 'off', 'FontSize', 12);
legend(num2str(elicitor'), 'location', 'best');

xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('\Delta Ear canal pressure (dB)', 'FontSize', 12);



figure;
plot(elicitor, mean(abs(MEM(:, ind)), 2) , 'ok-', 'linew', 2);
hold on;
xlabel('Elicitor Level (dB SPL)', 'FontSize', 12);
ylabel('\Delta Ear Canal Pressure (dB)', 'FontSize', 12);
set(gca,'FontSize', 12);




