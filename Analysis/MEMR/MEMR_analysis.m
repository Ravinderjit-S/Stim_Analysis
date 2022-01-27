clear

fig_loc = '/media/ravinderjit/Data_Drive/Data/Figures/MTBproj/MEMR/';

path = '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/MEMR/';
Subjects = {'S072','S078','S088','S207','S259','S260','S270', ...
    'S271','S273','S274','S277','S281','S282','S285','S291', ...
    'S305', 'S308', 'S309', 'S310'};

freq = linspace(200, 8000, 1024);
MEMband = [500, 2000];
ind = (freq >= MEMband(1)) & (freq <= MEMband(2));

for s = 1:length(Subjects)
    fold = dir([path Subjects{s}]);
    for fl = 1:2
        file = fold(2+fl).name;
        MEMfile = [path, Subjects{s}, '/', file];
        [MEMs, MEM, elicitor, n] = analyzeMEMR(MEMfile);
        f1 = figure;
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
        
        saveas(f1, [fig_loc, Subjects{s} '_Run_' num2str(fl) '.png'])
       


        f2 = figure;
        plot(elicitor, mean(abs(MEM(:, ind)), 2) , 'ok-', 'linew', 2);
        hold on;
        xlabel('Elicitor Level (dB SPL)', 'FontSize', 12);
        ylabel('\Delta Ear Canal Pressure (dB)', 'FontSize', 12);
        set(gca,'FontSize', 12);
        
        saveas(gcf, [fig_loc, Subjects{s} '_CanalP_Run_' num2str(fl) '.png'])
        
        close([f1,f2])
        
        
    end
    
    
end











