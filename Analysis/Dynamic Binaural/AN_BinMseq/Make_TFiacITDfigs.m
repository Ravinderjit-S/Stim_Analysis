%make figures



dates = [{'09.16.18'},{'09.20.18'},{'09.21.18'},{'10.01.18'},{'10.12.18'}];
dates = [{'09.21.18'},{'10.01.18'},{'10.12.18'}];

for TheDate = 1:numel(dates)
    
    date = dates{TheDate};
    
    load(['MseqAnalyzed_' date '.mat'])
    
    Time_Impulse = .050; %amount of time of impulse to keep
    Screen_pixelLocs = [1986 1 1855 1001];
    fs = 48828.125;
    
    parfor i = 1:numel(data_MOVN)
        MOVN_i = data_MOVN{i};
        
        
        Mseq_StepDur = MOVN_i.Stimuli.MSeq_StepDuration;
        Mseq_N = MOVN_i.Stimuli.MSeq_N;
        Mseq_samps = round(Mseq_StepDur*1e-3*fs)*(2^Mseq_N -1);
        Mseq_i = MOVN_i.Stimuli.ITDt(1:Mseq_samps);Mseq_i_noDC = Mseq_i;
        Mseq_i_noDC(Mseq_i ==0) = 1; Mseq_i_noDC(abs(Mseq_i) > 0) = -1;
        Mseq_i = Mseq_i_noDC; %fixing DC offset in the ITDt
        stim_dur = MOVN_i.Stimuli.Gating.Duration;
        stim_per = MOVN_i.Stimuli.Gating.Period;
        ITDt = MOVN_i.Stimuli.ITDt;
        Mseq_repeats = length(ITDt)/Mseq_samps;
        attens = MOVN_i.Stimuli.attens;
        picNum = MOVN_i.General.picture_number;
        
        spksL = MOVN_i.MSO.LeftSpks;
        spksR = MOVN_i.MSO.RightSpks;
        msosim = MOVN_i.MSO.simulated;
        for k = 1:numel(MOVN_i.MSO.simulated)
            if strcmp(msosim{k}, 'Not Enough Data')
                continue
            end
            H_imp = MOVN_i.MSO.H_imp{k};
            H_NFs = MOVN_i.MSO.H_NF{k};
            
            H_imp = H_imp(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
            phase_cycles = unwrap(angle(fft(H_imp))) ./ (2.*pi) ; %phase in cycles
            f_phase = fs*(0:length(phase_cycles)-1)/length(phase_cycles);
            [Hf f_H] = pmtm(H_imp,2.5,[],fs);
            
            attenuation = attens(k*2);
            CF = MOVN_i.tc.CFinterp;
            num_fibs = msosim{k}(end,1);
            MSO_spks = size(msosim{k},1);
            
            ThisFig = figure('Position',Screen_pixelLocs)
            fig_Tit = sprintf('Atten: %d, CF: %d MSOsimLines: %d MSOsimspks: %d, MSeqStepDur: %d, Pic:%d',round(attenuation), round(CF), num_fibs, MSO_spks,Mseq_StepDur,picNum);
            subplot(3,1,1), plot((0:length(H_imp)-1)/fs,H_imp,'linewidth',2), title(['H(t) ' fig_Tit]), xlabel('Time (sec)')
            hold on
            for m=1:numel(H_NFs)
                H_imp_dumb = H_NFs{m}(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
                plot((0:length(H_imp)-1)/fs,H_imp_dumb,'r')
            end
            hold off
            subplot(3,1,2),semilogx(f_H,pow2db(Hf),'linewidth',2), title('H(f)'),xlabel('Freq'), xlim([0 2000]),ylim([max(pow2db(Hf))-60 max(pow2db(Hf))])
            hold on
            for m=1:numel(H_NFs)
                H_imp_dumb = H_NFs{m}(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
                [fH_dumb, ~] = pmtm(H_imp_dumb,2.5,[],fs);
                semilogx(f_H, pow2db(fH_dumb),'r')
            end
            hold off
            subplot(3,1,3), plot(f_phase,phase_cycles,'linewidth',2), title('Phase(f)'), ylabel('Phase (cycles)'), xlim([0 600])
            saveas(gcf,['Mseq_ITDt_figs2/' date '/Fig' num2str(i) '.' num2str(k) '.png'])
            close(ThisFig)
            
        end
    end
    
    parfor i = 1:numel(data_NOSCOR)
        NOSCOR_i = data_NOSCOR{i};
        
        
        Mseq_StepDur = NOSCOR_i.Stimuli.MSeq_StepDuration;
        Mseq_N = NOSCOR_i.Stimuli.MSeq_N;
        Mseq_samps = round(Mseq_StepDur*1e-3*fs)*(2^Mseq_N -1);
        Mseq_i = NOSCOR_i.Stimuli.IACt(1:Mseq_samps);
        stim_dur = NOSCOR_i.Stimuli.Gating.Duration;
        stim_per = NOSCOR_i.Stimuli.Gating.Period;
        IACt = NOSCOR_i.Stimuli.IACt;
        Mseq_repeats = length(IACt)/Mseq_samps;
        attens = NOSCOR_i.Stimuli.attens;
        picNum = NOSCOR_i.General.picture_number;
        
        spksL = NOSCOR_i.MSO.LeftSpks;
        spksR = NOSCOR_i.MSO.RightSpks;
        msosim = NOSCOR_i.MSO.simulated;
        for k = 1:numel(NOSCOR_i.MSO.simulated)
            if strcmp(msosim{k}, 'Not Enough Data')
                continue
            end
            H_imp = NOSCOR_i.MSO.H_imp{k};
            H_NFs = NOSCOR_i.MSO.H_NF{k};
            
            H_imp = H_imp(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
            phase_cycles = unwrap(angle(fft(H_imp))) ./ (2.*pi) ; %phase in cycles
            f_phase = fs*(0:length(phase_cycles)-1)/length(phase_cycles);
            [Hf f_H] = pmtm(H_imp,2.5,[],fs);
            
            attenuation = attens(k*2);
            CF = NOSCOR_i.tc.CFinterp;
            num_fibs = msosim{k}(end,1);
            MSO_spks = size(msosim{k},1);
            
            ThisFig = figure('Position',Screen_pixelLocs)
            fig_Tit = sprintf('Atten: %d, CF: %d MSOsimLines: %d MSOsimspks: %d, MSeqStepDur: %d, Pic:%d',round(attenuation), round(CF), num_fibs, MSO_spks,Mseq_StepDur,picNum);
            subplot(3,1,1), plot((0:length(H_imp)-1)/fs,H_imp,'linewidth',2), title(['H(t) ' fig_Tit]), xlabel('Time (sec)')
            hold on
            for m=1:numel(H_NFs)
                H_imp_dumb = H_NFs{m}(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
                plot((0:length(H_imp)-1)/fs,H_imp_dumb,'r')
            end
            hold off
            subplot(3,1,2),semilogx(f_H,pow2db(Hf),'linewidth',2), title('H(f)'),xlabel('Freq'), xlim([0 2000]),ylim([max(pow2db(Hf))-60 max(pow2db(Hf))])
            hold on
            for m=1:numel(H_NFs)
                H_imp_dumb = H_NFs{m}(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
                [fH_dumb, ~] = pmtm(H_imp_dumb,2.5,[],fs);
                semilogx(f_H, pow2db(fH_dumb),'r')
            end
            hold off
            subplot(3,1,3), plot(f_phase,phase_cycles,'linewidth',2), title('Phase(f)'), ylabel('Phase (cycles)'), xlim([0 600])
            saveas(gcf,['Mseq_IACt_figs2/' date '/Fig' num2str(i) '.' num2str(k) '.png'])
            close(ThisFig)
            
        end
    end
end




