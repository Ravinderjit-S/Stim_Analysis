% look at CF trends
clear all
Analyze_dates = [{'09.16.18'},{'09.20.18'},{'09.21.18'},{'10.01.18'},{'10.12.18'}];
%Analyze_dates = [{'09.20.18'},{'09.21.18'},{'10.01.18'},{'10.12.18'}];


addpath('../../Neuron Analysis Functions')

Screen_pixelLocs = [1986 1 1855 1001]; %makes figure the size of my second screen at Lyle ... saves the figure as a larger figure
All_CF_RT = [];
All_CF_interp = [];
All_atten = [];
All_Roll_off_Freq = [];
All_GroupDelay = [];
All_MSOspks = [];
All_MSOlns = [];
All_Mseq_stepDur = [];
Used = [];
All_Himpt = [];
All_Himpf = [];
All_phasef = [];
All_spontRate = [];
All_Roll_off_Freq10dB = [];
All_Roll_off_Freq6dB = [];
All_Breathing = [];

Time_Impulse = .050; %time of H_imp to keep
fs = 48828.125;

for TheDate = 1:numel(Analyze_dates)
    load(['MseqAnalyzed_' Analyze_dates{TheDate} '.mat'])
    
    for getPics = 1:numel(data_NOSCOR)
        NOSCOR_pics(getPics) = data_NOSCOR{getPics}.General.picture_number;
    end
    Already_analyzed = []; %if there are repeat stimuli ... they will be skipped here
    [SamePics, SameStim, SameUnit] = FindSamePics(data_NOSCOR);
    for i = 1:numel(data_NOSCOR)
        if any(i==Already_analyzed)
            continue
        end
        NOSCOR_i = data_NOSCOR{i};
        
        Mseq_StepDur = NOSCOR_i.Stimuli.MSeq_StepDuration;
        Mseq_N = NOSCOR_i.Stimuli.MSeq_N;
        Mseq_samps = round(Mseq_StepDur*1e-3*fs)*(2^Mseq_N -1);
        Mseq_i = NOSCOR_i.Stimuli.IACt(1:Mseq_samps);
        stim_dur = NOSCOR_i.Stimuli.Gating.Duration;
        stim_per = NOSCOR_i.Stimuli.Gating.Period;
        IACt = NOSCOR_i.Stimuli.IACt;
        Mseq_repeats = length(IACt)/Mseq_samps;
        unit_i = NOSCOR_i.General.track + NOSCOR_i.General.unit/100;
        attens = NOSCOR_i.Stimuli.attens;
        if Mseq_N ~=9
            error('Check Mseq_N')
        end
%         if Mseq_StepDur ~= 2
%             error('Check Mseq_StepDur')
%         end

        % Getting Spont Rate


%         Calcu spontRate on every cond and stuff maybe
%         L_spks = NOSCOR_i.MSO.LeftSpks{1};
%         R_spks = NOSCOR_i.MSO.RightSpks{1};
        [spontRate, drivenRate, spontSTD, drivenSTD] = SpontaneousRate(NOSCOR_i.MSO.LeftSpks{1},stim_dur,stim_per);    
                       

            
        
        for k =1:numel(NOSCOR_i.MSO.H_imp)
            H_NFt = [];
            H_NFf = [];
            
            H_dummy = NOSCOR_i.MSO.H_NF{k};
            H_imp = NOSCOR_i.MSO.H_imp{k};
            
%             if any(unit_i == SameUnit) % combine same stimuli in diff pics ----------!!!!!!!!!!!!!!!!!! This needs to be fixed
%                 warning('Need to fix Unit combining ... not weighted properly currently')
%                 PicsSame = SamePics{SameUnit==unit_i};
%                 SameUnit(SameUnit==unit_i) = [];
%                 NOSCOR_Inds = [];
%                 for thePics = 1:numel(PicsSame)
%                     NOSCOR_Inds = [NOSCOR_Inds find(NOSCOR_pics == PicsSame(thePics))];
%                 end
%                 if sum(i==NOSCOR_Inds) ~=1
%                     error('Something is off')
%                 end
%                 NOSCOR_Inds = setdiff(NOSCOR_Inds,i); %the other indexes beside the current one to include
%                 Already_analyzed = [Already_analyzed NOSCOR_Inds];
%                 MSO_spks = size(NOSCOR_i.MSO.simulated{k},1);
%                 MSO_lines = NOSCOR_i.MSO.simulated{k}(end,1);
%                 for together = 1:numel(NOSCOR_Inds)
%                     MSO_spks = MSO_spks + size(data_NOSCOR{NOSCOR_Inds(together)}.MSO.simulated{k},1);
%                     MSO_lines = MSO_lines + data_NOSCOR{NOSCOR_Inds(together)}.MSO.simulated{k}(end,1);
%                     H_imp = H_imp+data_NOSCOR{NOSCOR_Inds(together)}.MSO.H_imp{k};
%                     H_dummy2 = data_NOSCOR{NOSCOR_Inds(together)}.MSO.H_NF{k};
%                     for dumComb = 1:numel(H_dummy) % don't need this variable to be a cell array and just made my life more difficult but whatever ...
%                         H_dummy{dumComb} = H_dummy{dumComb} + H_dummy2{dumComb};
%                     end
%                 end
%                 H_imp = H_imp ./ numel(PicsSame);
%                 for dumComb = 1:numel(H_dummy)
%                     H_dummy{dumComb} = H_dummy{dumComb} ./ numel(PicsSame);
%                 end
%             else
%                 MSO_spks = size(NOSCOR_i.MSO.simulated{k},1);
%                 MSO_lines = NOSCOR_i.MSO.simulated{k}(end,1);
%             end
            MSO_spks = size(NOSCOR_i.MSO.simulated{k},1);
            MSO_lines = NOSCOR_i.MSO.simulated{k}(end,1);
            for j = 1:numel(H_dummy)
                H_imp_NF = H_dummy{j}(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
                H_NFt = vertcat(H_NFt, H_imp_NF);
                H_NFf = vertcat(H_NFf, pmtm(H_imp_NF,2.5,[],fs)');
            end
            H_impt = H_imp(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
            [H_impf,f_H] = pmtm(H_impt,2.5,[],fs);
            Z_Ht = (H_impt - mean(H_NFt)) ./ std(H_NFt);
            Z_Hf = (pow2db(H_impf') - mean(pow2db(H_NFf))) ./ std(pow2db(H_NFf));
            Z_score_cutoff = 2;
            Roll_off = f_H(find(Z_Hf<Z_score_cutoff,1,'first'));
            H_phase = unwrap(angle(fft(H_impt)));
            f_phase = fs*(0:length(H_phase)-1)/length(H_phase);
            
            Roll_off_10dB = f_H(find(pow2db(H_impf) <= max(pow2db(H_impf))-10,1,'first')); %10 dB drop off in power point 
            Roll_off_6dB = f_H(find(pow2db(H_impf) <= max(pow2db(H_impf))-6,1,'first')); %6 dB dropp off
            
            
            attenuation = attens(k*2);
            CF = NOSCOR_i.tc.CFinterp;
            
            
            fig_Tit = sprintf('Atten: %d, CF: %d MSOsimLines: %d MSOsimspks: %d, MSeqStepDur: %d',round(attenuation), round(CF), MSO_lines, MSO_spks,Mseq_StepDur);
            figure('visible','off','Position',Screen_pixelLocs)
            subplot(3,1,1),plot((0:length(H_impt)-1)/fs .*1000, H_impt,'linewidth',2),title(['H(t) ' fig_Tit]),xlabel('Time (msec)'), ylabel('Firing Rate / IAC')  
            hold on
            for m = 1:size(H_NFt,1)
                plot((0:length(H_impt)-1)/fs .*1000,H_NFt(m,:),'r')
            end
            hold off
            subplot(3,1,2),semilogx(f_H,pow2db(H_impf),'linewidth',2), title('H(f)'), xlabel('Frequency (Hz)'), xlim([0 600]), ylim([max(pow2db(H_impf))-50, max(pow2db(H_impf))+10]), ylabel('Power (dB/Hz)')
            hold on
            for m = 1:size(H_NFf,1)
                semilogx(f_H,pow2db(H_NFf(m,:)),'r')
            end
            semilogx(Roll_off*ones(1,10), [max(pow2db(H_impf)):-10:max(pow2db(H_impf))-90],'k','linewidth',2)
            semilogx(Roll_off_10dB*ones(1,10), [max(pow2db(H_impf)):-10:max(pow2db(H_impf))-90],'m','linewidth',2)
            hold off
            subplot(3,1,3),plot(f_phase,H_phase),xlim([0 500]), xlabel('Frequency (Hz)'), ylabel('Radians'), title('Phase(f)')

            
            F_GD_line = f_phase(1:find(f_phase>Roll_off_10dB,1,'first')-1);
            PhaseFit = polyfit(F_GD_line,H_phase(1:find(f_phase>Roll_off_10dB,1,'first')-1),1);
            GD_line = PhaseFit(1)*F_GD_line + PhaseFit(2);
            hold on, plot(F_GD_line,GD_line,'k')
            
            if MSO_lines <100
                Used = [Used 0];
                saveas(gcf,['Not_Used/' Analyze_dates{TheDate} '_Fig' num2str(i) '.' num2str(k) '.png'])
            else
                Used = [Used 1];
                saveas(gcf,['Used/' Analyze_dates{TheDate} '_Fig' num2str(i) '.' num2str(k) '.png'])
            end
                
            
%             figure, semilogx(f_H,Z_Hf),title('Z_Hf')
            
            All_CF_RT = [All_CF_RT NOSCOR_i.tc.CFrealtime];
            All_CF_interp = [All_CF_interp CF];
            All_atten = [All_atten attenuation];
            All_MSOspks = [All_MSOspks MSO_spks];
            All_MSOlns = [All_MSOlns MSO_lines];
            All_Mseq_stepDur = [All_Mseq_stepDur Mseq_StepDur];
            All_Roll_off_Freq = [All_Roll_off_Freq, Roll_off];
            All_Roll_off_Freq10dB = [All_Roll_off_Freq10dB, Roll_off_10dB];
            All_Roll_off_Freq6dB = [All_Roll_off_Freq6dB, Roll_off_6dB];
            All_Himpt = [All_Himpt; H_impt];
            All_Himpf = [All_Himpf; H_impf'];
            All_phasef = [All_phasef; H_phase];
            All_spontRate = [All_spontRate spontRate];
            
            if strcmp(Analyze_dates{TheDate},'10.12.18')
                All_Breathing = [All_Breathing, 1];
            else
                All_Breathing = [All_Breathing, 0];
            end
            
            
            GroupDelay = PhaseFit(1);
            All_GroupDelay = [All_GroupDelay,GroupDelay./(2*pi)];
            
        end
        
    end
end

All_GroupDelay = abs(All_GroupDelay);

% figure,semilogx(sort(unique(All_CF_interp(Used==1))), ones(1,length(unique(All_CF_interp(Used==1)))),'x')
% xlabel('CF')
% 
% figure,
% subplot(2,1,1), semilogx(All_CF_interp,All_Roll_off_Freq,'x'), xlabel('CF (Hz)'),ylabel('Roll off Frequency (Hz)'), title('All')
% subplot(2,1,2), semilogx(All_CF_interp,abs(All_GroupDelay),'x'),xlabel('CF(Hz)'), ylabel('Group Delay (sec)')
% 
% 
% 
% Mseq1 = Used==1 & All_Mseq_stepDur ==1;
% Mseq2 = Used==1 & All_Mseq_stepDur ==2;
% 
% figure
% Mask = Mseq1;
% Mask2 = Mseq2;
% subplot(2,1,1), semilogx(All_CF_interp(Mask),All_Roll_off_Freq(Mask),'rx'), xlabel('CF (Hz)'),ylabel('Roll off Frequency (Hz)'), title('Used IAC')
% hold on, semilogx(All_CF_interp(Mask2),All_Roll_off_Freq(Mask2),'bx'), hold off, legend('MseqDur: 1','MseqDur:2')
% subplot(2,1,2), semilogx(All_CF_interp(Mask),abs(All_GroupDelay(Mask)),'rx'),xlabel('CF(Hz)'), ylabel('Group Delay (sec)')
% hold on, semilogx(All_CF_interp(Mask2),abs(All_GroupDelay(Mask2)),'bx'), hold off
% 
% 
% 
% un_Atten = unique(All_atten);
% Colors = ['rcbkm'];
% figure
% for j =1:length(un_Atten)
%     Mask = Mseq1 & All_atten == un_Atten(j);
%     color_j = Colors(j);
%     subplot(2,1,1), hold on, semilogx(All_CF_interp(Mask),All_Roll_off_Freq(Mask),'x','Color',color_j), xlabel('CF (Hz)'),ylabel('Roll off Frequency (Hz)'), title(['Used, MseqDur: 1, atten: ' num2str(un_Atten(j))])
%     subplot(2,1,2), hold on, semilogx(All_CF_interp(Mask),abs(All_GroupDelay(Mask)),'x','Color',color_j),xlabel('CF(Hz)'), ylabel('Group Delay (sec)')
% end
% legend(num2str(un_Atten(1)),num2str(un_Atten(2)),num2str(un_Atten(3)),num2str(un_Atten(4)),num2str(un_Atten(5)))
% 
% figure
% for j =1:length(un_Atten)
%     Mask = Mseq2 & All_atten == un_Atten(j);
%     color_j = Colors(j);
%     subplot(2,1,1), hold on, semilogx(All_CF_interp(Mask),All_Roll_off_Freq(Mask),'x','Color',color_j), xlabel('CF (Hz)'),ylabel('Roll off Frequency (Hz)'), title(['Used, MseqDur: 2, atten: ' num2str(un_Atten(j))])
%     subplot(2,1,2), hold on,semilogx(All_CF_interp(Mask),abs(All_GroupDelay(Mask)),'x','Color',color_j),xlabel('CF(Hz)'), ylabel('Group Delay (sec)')
% end
% legend(num2str(un_Atten(1)),num2str(un_Atten(2)),num2str(un_Atten(3)),num2str(un_Atten(4)),num2str(un_Atten(5)))

Mask = Used==1 & All_Roll_off_Freq>0 & All_Roll_off_Freq > All_Roll_off_Freq10dB & (All_Mseq_stepDur == 1 | All_Mseq_stepDur ==2);
Mask2 = Mask & All_spontRate <= 0.1 & All_atten ==40;
Mask3 = Mask & (All_spontRate>0.1 & All_spontRate <=18)  & All_atten ==40;
Mask4 = Mask & All_spontRate >18 & All_atten==40;
Mask5 = Mask & All_Breathing ==1 & All_atten ==40;

figure, plot(All_CF_interp(Mask2), All_spontRate(Mask2),'rx'), xlabel('CF'), ylabel('SpontRate')
hold on, plot(All_CF_interp(Mask3), All_spontRate(Mask3),'bx')
plot(All_CF_interp(Mask4), All_spontRate(Mask4),'gx')
plot(All_CF_interp(Mask5), All_spontRate(Mask5),'ko')
%legend('Low Spont', 'Med Spont', 'High Spont','Breathing Inflated')
legend('Med Spont', 'High Spont', 'Breathing Inflated')
set(gca,'fontsize',14)

figure, plot(All_CF_interp(Mask), All_Roll_off_Freq(Mask),'x'), xlabel('CF') , ylabel('Roll off Frequency (Hz)'),set(gca,'fontsize',14)

figure, plot(All_CF_interp(Mask2), All_Roll_off_Freq(Mask2),'rx'), xlabel('CF') , ylabel('Roll off Frequency (Hz)'),set(gca,'fontsize',14)
hold on, plot(All_CF_interp(Mask3), All_Roll_off_Freq(Mask3),'bx')
plot(All_CF_interp(Mask4), All_Roll_off_Freq(Mask4),'gx')
plot(All_CF_interp(Mask5), All_Roll_off_Freq(Mask5),'ko')
%legend('Low Spont', 'Med Spont','High Spont','Breathing Inflated')
legend('Med Spont', 'High Spont', 'Breathing Inflated')

figure, plot(All_CF_interp(Mask2), All_Roll_off_Freq10dB(Mask2),'rx'), xlabel('CF'), ylabel('Roll Off -10 dB'), set(gca,'fontsize',14)
hold on, plot(All_CF_interp(Mask3), All_Roll_off_Freq10dB(Mask3),'bx')
plot(All_CF_interp(Mask4), All_Roll_off_Freq10dB(Mask4),'gx')
plot(All_CF_interp(Mask5), All_Roll_off_Freq10dB(Mask5),'ko')
%legend('Low Spont', 'Med Spont','High Spont','Breathing Inflated')
legend('Med Spont', 'High Spont', 'Breathing Inflated')

Mask_20 = Mask & All_atten ==20;
Mask_30 = Mask & All_atten ==30;
Mask_40 = Mask & All_atten ==40;
Mask_50 = Mask & All_atten ==50; 
Mask_60 = Mask & All_atten ==60;

Spectral_Level = round(110 - unique(All_atten) - 10*log10(20e3-10));

figure, hold on, xlabel('CF'), ylabel('Roll off -10dB'), set(gca,'fontsize',14)
plot(All_CF_interp(Mask_60), All_Roll_off_Freq10dB(Mask_60),'gx','linewidth',2)
plot(All_CF_interp(Mask_50), All_Roll_off_Freq10dB(Mask_50),'ro','linewidth',2)
plot(All_CF_interp(Mask_40), All_Roll_off_Freq10dB(Mask_40),'bx','linewidth',2)
plot(All_CF_interp(Mask_30), All_Roll_off_Freq10dB(Mask_30),'mo','linewidth',2)
plot(All_CF_interp(Mask_20), All_Roll_off_Freq10dB(Mask_20),'cx','linewidth',2)
%legend('60','50','40','30','20')
legend('7 dB SPL/Hz','17','27','37','47')

figure, plot(All_CF_interp(Mask_40), All_Roll_off_Freq10dB(Mask_40),'bx','linewidth',2), xlabel('CF'), ylabel('Roll off -10dB'), legend('70 db SPL'),set(gca,'fontsize',14)
% figure, plot(All_CF_interp(Mask_40), All_Roll_off_Freq3dB(Mask_40),'bx'), xlabel('CF'), ylabel('Roll off -3dB'), legend('70 db SPL')


figure
Mask2_40 = Mask_40;
% Mask3_temp = boolean([1,0,0,1,0,0,1,1,1,1,1,1]);
All_Himpf2 = All_Himpf(Mask2_40',:);
All_Roll_off_Freq2 = All_Roll_off_Freq10dB(Mask2_40);
All_Roll_off_Freq2 = All_Roll_off_Freq6dB(Mask2_40);
All_phasef2 = All_phasef(Mask2_40',:);
All_CF_interp2 = All_CF_interp(Mask2_40);
All_GroupDelay2 = All_GroupDelay(Mask2_40);
% All_Himpf2 = All_Himpf2(Mask3_temp',:);
% All_Roll_off_Freq2 = All_Roll_off_Freq2(Mask3_temp);
% All_CF_interp2 = All_CF_interp2(Mask3_temp);
% All_phasef2 = All_phasef2(Mask3_temp',:);
% All_GroupDelay2 = All_GroupDelay2(Mask3_temp);

[~, SortedCFinds] = sort(All_CF_interp2);
ColorScheme = jet(size(All_Himpf2,1));

FontSize = 25;

for i = 1:size(All_Himpf2,1)
    Himpf_i = All_Himpf2(i,:);
    RollFreq_i = All_Roll_off_Freq2(i);
    CF_i = All_CF_interp2(i);
    Himpf_i = Himpf_i(f_H <= RollFreq_i);
    semilogx(f_H(f_H <= RollFreq_i), pow2db(Himpf_i),'color',ColorScheme(SortedCFinds==i,:),'linewidth',2), xlim([10 500])
    hold on
end
xticks([10, 100])
xticklabels({'10', '100'})
hold off
xlabel('Frequency (Hz)','FontSize',FontSize)
ylabel('Power (dB/Hz)','FontSize',FontSize)
colormap(jet)
% c = colorbar;
% c.Label.String = 'CF (Hz)';
% caxis([min(All_CF_interp2), max(All_CF_interp2)])
set(gca,'fontsize',25)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
print('IAC_PowerFreq','-dpng','-r0')

figure, hold on
for i =1:size(All_Himpf2,1)
    plot(All_CF_interp2(i),All_Roll_off_Freq2(i),'o','color',ColorScheme(SortedCFinds==i,:),'linewidth',5,'MarkerSize',10)
end
RF_fit = polyfit(All_CF_interp2,All_Roll_off_Freq2,1);
RF_fitLine = RF_fit(1).*All_CF_interp2+RF_fit(2);
% plot(All_CF_interp2,RF_fitLine,'k')
ylim([0 max(All_Roll_off_Freq2)+50])
hold off
xlabel('CF (Hz)','fontsize',20), ylabel('Roll off frequency (Hz)'), ylim([100 270]), xlim([min(All_CF_interp2)-50 max(All_CF_interp2)+50])
yticks([100:50:275])
xticks([500:500:2500])
set(gca,'fontsize',25)
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
print('IAC_CFvsRollOff_6dB','-dpng','-r0')

figure, hold on
for i =1:size(All_phasef2)
    Phasef_i = All_phasef2(i,:);
    RollFreq_i = All_Roll_off_Freq2(i);
    CF_i = All_CF_interp2(i);
    Phasef_i = Phasef_i(f_phase <= RollFreq_i);
    plot(f_phase(f_phase <= RollFreq_i),Phasef_i./(2*pi),'color',ColorScheme(SortedCFinds==i,:),'linewidth',2), xlim([0 400])
end
hold off
xlabel('Frequency (Hz)'), ylabel('Cycles')
set(gca,'fontsize',25)
% c = colorbar;
% colormap(jet)
% c.Label.String = 'CF (Hz)';
% caxis([min(All_CF_interp2), max(All_CF_interp2)])
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
print('IAC_PhaseFreq','-dpng','-r0')

figure, hold on
for i=1:size(All_Himpf2,1)
    plot(All_CF_interp2(i),All_GroupDelay2(i)*1000,'o','color',ColorScheme(SortedCFinds==i,:),'linewidth',5,'MarkerSize',10) 
end
hold off
xlabel('CF (Hz)'), ylabel ('Group Delay (ms)'), xlim([min(All_CF_interp2)-50 max(All_CF_interp2)+50]), ylim([3, max(All_GroupDelay2*1000) + 0.5])
yticks([3:8])
set(gca,'fontsize',25)    
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 11 8];
print('IAC_GroupDelay','-dpng','-r0')
   



