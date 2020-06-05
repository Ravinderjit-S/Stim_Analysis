% This code is to analyze auditory nerve data with the binding stimulus
% collected in Summer, 17
clear
addpath('../Neuron Analysis Functions')
addpath('/media/ravinderjit/Data_Drive/Chindata_Summer17/Chin Stim')
load('/media/ravinderjit/Data_Drive/Chindata_Summer17/MS-2017_06_07-M17015_normal_MTB.mat'); %loads variable data
FigPath = '/media/ravinderjit/Data_Drive/Data/Figures/BindingPilot/';

data2 = load('/media/ravinderjit/Data_Drive/Chindata_Summer17/MS-2017_06_20-M17016_normal_MTB.mat');
data2 = data2.data;

% dataInds = [1,2,3,6,11,13,14,16,17,18,21,23]; % [-2:2], useCF = yes
% dataInds2 = [9]; %ERBspace = 1
% dataInds3 = [22]; ERBspace = 2 
% dataInds4 = [36];

Cond_CorrInds = {};
Cond_UseCf = {};
Cond_atten = [];
Cond_CF = [];
Cond_Aavg = [];
Cond_Bavg = [];
Cond_ind = [];
Cond_ERBspace = [];
Cond_lines = [];

for j = 1:length(data)
    fprintf('%d \n',j)
    pic = data{j};
    ERB_space = pic.Stimuli.ERB_space;
    Corr_inds = pic.Stimuli.Corr_inds;
    ncoh = pic.Stimuli.ncoh;
    Use_CF = pic.Stimuli.Use_CF_Tone;
    nstim = pic.Stimuli.nstim;
    attens = pic.Stimuli.attens;
    NoiseSeed = pic.Stimuli.NoiseSeed;
    Corr_Tones_inds = pic.Stimuli.Corr_Tones_inds;
    full_lines = pic.Stimuli.fully_presented_lines;
    stims = pic.Stimuli.list_RP1;
    spike_times = pic.spikes{:};
    cf_ind = pic.Stimuli.cfInd;
    Tone_f = pic.Stimuli.TonesUsed;
    Tones = length(Tone_f);
    CF = Tone_f(cf_ind);
    
    if isempty(Corr_inds) || length(Corr_inds) ~=5 || length(stims)>1 || ~strcmpi(Use_CF, 'Yes ')
        continue
    end
    
    fs = 48828;
    [s,fr,tmod] = spectrogram(stims{:},fs/100,[],[],fs,'yaxis');

    mod = abs(s(find(fr>=Tone_f(cf_ind),1),:));
%     figure,plot(tmod,mod)
%     hold on, plot(tmod,abs(s(find(fr>=Tone_f(cf_ind),1),:)),'r')

    [spk_cond] = ExtractConditions(pic);

    % plotRaster(spk_cond{2})
    Amax = [];
    A2max = [];
    Bmax = [];
    B2max = [];
    for i = 1:length(spk_cond)
        Cond_CorrInds = [Cond_CorrInds {Corr_inds}];
        Cond_UseCf = [Cond_UseCf {Use_CF}];
        Cond_atten = [Cond_atten attens(i)];
        Cond_CF = [Cond_CF CF];
        Cond_ind = [Cond_ind j];
        Cond_ERBspace = [Cond_ERBspace ERB_space];
        
        spks = spk_cond{i};
        Cond_lines = [Cond_lines spks(end,1)];
        [t, A_fr] = FiringRate(spks, [], 5, 5000, 1);
        hold on
        plot(tmod,mod*25+50,'r','linewidth',2)
        legend('Firing Rate','Envelope')
    %     ylim([50 300])
    %     xticks([0,1,2,3,4,5])
    %     yticks([100, 200, 300])
    %     set(gca,'fontname','Arial');
    %     set(gca,'fontsize',11);
    %     fig = gcf;
    %     fig.PaperUnits = 'inches';
    %     fig.PaperPosition = [0 0 6 6];
    %     print([FigPath 'FiringRate'],'-dpng','-r0')


    %     figure,plot(tmod, A_fr(1:length(tmod)))
    %     hold on
    %     plot(tmod, mod*30+40,'r')

        onesec = find(tmod>=1,1)-1;
        twosec = find(tmod>=2,1)-1;
        thrsec = find(tmod>=3,1)-1;
        A = xcorr(A_fr(1:onesec),mod(1:onesec),'coeff');
        B = xcorr(A_fr(onesec+1:twosec),mod(onesec+1:twosec),'coeff');
        A2 = xcorr(A_fr(twosec+1:thrsec),mod(twosec+1:thrsec),'coeff');
        B2 = xcorr(A_fr(thrsec+1:length(tmod)),mod(thrsec+1:end),'coeff');

        Amax = [Amax max(A)];
        A2max = [A2max max(A2)]; 
        Bmax = [Bmax max(B)];
        B2max = [B2max max(B2)];
    end
    Aavg = (Amax + A2max) / 2;
    Bavg = (Bmax + B2max) / 2;
    Cond_Aavg = [Cond_Aavg Aavg];
    Cond_Bavg = [Cond_Bavg Bavg];

end

horzcat(Cond_Aavg', Cond_Bavg',Cond_ERBspace', Cond_lines' ,Cond_CF'/1000)
horzcat(Cond_Bavg' -Cond_Aavg',Cond_ERBspace', Cond_lines' ,Cond_CF'/1000)

Cond_diff = Cond_Bavg' - Cond_Aavg'; 
Cond_diff = 100 * Cond_diff ./ Cond_Aavg'; %This is now percent increase
mask = Cond_atten == 40 & Cond_lines >= 20;

mn1 = mean(Cond_diff(Cond_ERBspace ==1 & mask));
mn15 = mean(Cond_diff(Cond_ERBspace ==1.5 & mask));
mn2 = mean(Cond_diff(Cond_ERBspace ==2 & mask));

sem1 = std(Cond_diff(Cond_ERBspace ==1 & mask)) / sqrt(length(Cond_diff(Cond_ERBspace ==1 & mask)))
sem15 = std(Cond_diff(Cond_ERBspace ==1.5 & mask)) / sqrt(length(Cond_diff(Cond_ERBspace ==1.5 & mask)))
sem2 = std(Cond_diff(Cond_ERBspace ==2 & mask)) / sqrt(length(Cond_diff(Cond_ERBspace ==2 & mask)))

figure, hold on
barplt = [mn1, mn15, mn2];
stder = [sem1, sem15, sem2];
bar(barplt)
errorbar([1,3],barplt([1,3]),stder([1,3]), '.k')
yticks([-2, 0, 2, 4, 6, 8])
xticks([1,2,3])
xticklabels({'1','1.5','2'})
xlabel('ERB spacing')
ylabel('% Increase in ENVcoh')
set(gca,'fontname','Arial')
set(gca,'fontsize',11)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 4.5 3];
print([FigPath 'Nerve_ERBspacing'],'-dpng','-r0')





