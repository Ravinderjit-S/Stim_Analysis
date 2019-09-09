function [Converged] = NerveMseq_ConvergenceAnalysis(data)

addpath('../../Neuron Analysis Functions')
Screen_pixelLocs = [1986 1 1855 1001]; %makes figure the size of my second screen at Lyle ... saves the figure as a larger figure

Time_Impulse = .050; %time of H_imp to keep
fs = 48828.125;

for i = 1:numel(data)
    PicNums_data(i) = data{i}.General.picture_number; %#ok
end

if isfield(data{1}.Stimuli,'IACt')
    NOSCOR = 1;
else
    NOSCOR = 0;
end
    
Converged = [];

for i = 1:numel(data)
    %% Stim Parameters
    data_i = data{i};
    Mseq_StepDur = data_i.Stimuli.MSeq_StepDuration;
    Mseq_N = data_i.Stimuli.MSeq_N;
    Mseq_samps = round(Mseq_StepDur*1e-3*fs)*(2^Mseq_N -1);
    stim_dur = data_i.Stimuli.Gating.Duration;
    stim_per = data_i.Stimuli.Gating.Period;
    unit_i = data_i.General.track + data_i.General.unit/100;
    attens = data_i.Stimuli.attens;
    
    if NOSCOR
        Mseq_i = data_i.Stimuli.IACt(1:Mseq_samps);
        Mseqt = data_i.Stimuli.IACt;
    else
        Mseq_i = data_i.Stimuli.ITDt(1:Mseq_samps);Mseq_i_noDC = Mseq_i;
        Mseq_i_noDC(Mseq_i ==0) = 1; Mseq_i_noDC(abs(Mseq_i) > 0) = -1;
        Mseq_i = Mseq_i_noDC; %fixing DC offset in the ITDt
        Mseqt = data_i.Stimuli.ITDt;
    end

    Mseq_repeats = length(Mseqt)/Mseq_samps;
    
    if Mseq_N ~=9
        error('Check Mseq_N')
    end
    
    
    %% Convergence Analysis
   
    for k =1:numel(data_i.MSO.simulated)

        if strcmp(data_i.MSO.simulated{k}, 'Not Enough Data')
            continue
        end
        spikes = data_i.MSO.simulated{k};
        Roll_off_10dB_All = []; Roll_off_3dB_All = []; H_f_m_All = [];
        total_trials = spikes(end,1);
        if total_trials > 50
            start = total_trials-50;
        else
            start = 1;
        end
        Check_converge = start:5:total_trials;
        if Check_converge(end)~=total_trials
            Check_converge(end) = total_trials;
        end
        for m = Check_converge
            spikes_m = spikes(find(spikes(:,1) <= m),:);
            [~, FRate_m] = FiringRate(spikes_m,'bin',(1/fs)*1e3,stim_dur,0);
            Wrapped_m = nan(Mseq_repeats,Mseq_samps);
            for n=1:Mseq_repeats
                Wrapped_m(n,:) = FRate_m((n-1)*Mseq_samps+1:n*Mseq_samps);
            end
            FRate_wrapd_m = mean(Wrapped_m);
            H_imp_m = cconv(FRate_wrapd_m,Mseq_i(end:-1:1));
            H_imp_m = H_imp_m(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
            [H_f_m, f_H] = pmtm(H_imp_m,2.5,[],fs);
            H_f_m_All = [H_f_m_All;H_f_m'];
            Roll_off_10dB = f_H(find(pow2db(H_f_m) <= pow2db(H_f_m(2))-10,1,'first')); %10 dB drop off in power point 
            Roll_off_3dB = f_H(find(pow2db(H_f_m) <= pow2db(H_f_m(2))-3,1,'first')); %3 dB dropp off
            Roll_off_10dB_All = [Roll_off_10dB_All, Roll_off_10dB];
            Roll_off_3dB_All = [Roll_off_3dB_All, Roll_off_3dB];
            
        end
        Var_H_f = var(pow2db(H_f_m_All));
        Converged = [Converged all(Var_H_f(1:find(f_H<=500,1,'last')) <1)];
        attenuation = attens(k*2);
        picNum = data_i.General.picture_number;
        Fig_Tit = sprintf('Atten: %d, picNum: %d', attenuation, picNum);
        figure, subplot(2,1,1), plot(Check_converge,Roll_off_10dB_All,'x-'), title(Fig_Tit), ylabel('Roll off 10dB'), xlabel('MSO Lines')
        subplot(2,1,2), plot(Check_converge,Roll_off_3dB_All,'x-'), ylabel('Roll off 3 dB')
        figure,semilogx(f_H,pow2db(H_f_m_All(1:end-1,:))','r'), xlim([0 1000])
        hold on, semilogx(f_H,pow2db(H_f_m_All(end,:)),'b','linewidth',3), xlabel('Frequency (Hz)'), ylabel('Power (db/Hz)'), title('H_f'), hold off
        figure, semilogx(f_H, Var_H_f), ylabel('Variance (db/Hz)'), xlabel('Frequency (Hz)'), title('Variance vs frequency')
        
    end
    
end

end
        
           
            
            
            
        
        
    
