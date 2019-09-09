function [] = AddNFMSO(date,Add_dummies)
%adds [Add_dummies] # of NFs to simulated MSO data

addpath('../../Neuron Analysis Functions')

load(['MseqAnalyzed_' date '.mat'])

fs = 48828.125;

for i = 1:numel(data_MOVN)
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
    
    
    for k = 1:numel(MOVN_i.MSO.simulated)
        msosim_k = MOVN_i.MSO.simulated{k};
        if strcmp(msosim_k, 'Not Enough Data')
            continue
        end
        for m = 1:Add_dummies
            Jumbled_MSO = Jumble_spikes(msosim_k,stim_dur);
            Dummy_MSO{m} = Jumbled_MSO;
        end
        
        for m=1:Add_dummies
            [t, P_spk_dumb] = FiringRate(Dummy_MSO{m},'bin', (1/fs)*1e3, stim_dur,0);
            Wrapped_dumb = nan(Mseq_repeats,Mseq_samps);
            for n=1:Mseq_repeats
                Wrapped_dumb(n,:) = P_spk_dumb((n-1)*Mseq_samps+1:n*Mseq_samps);
            end
            P_spk_wrapd_dumb = mean(Wrapped_dumb);
            H_imp_dumb{m} = cconv(P_spk_wrapd_dumb,Mseq_i(end:-1:1));
        end
        
        MOVN_i.MSO.H_NF{k} = [MOVN_i.MSO.H_NF{k}, H_imp_dumb];
    end
    data_MOVN{i} = MOVN_i;
    

end

for i = 1:numel(data_NOSCOR)
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
        
      for k = 1:numel(NOSCOR_i.MSO.simulated)
        msosim_k = NOSCOR_i.MSO.simulated{k};
        if strcmp(msosim_k, 'Not Enough Data')
            continue
        end
        for m = 1:Add_dummies
            Jumbled_MSO = Jumble_spikes(msosim_k,stim_dur);
            Dummy_MSO{m} = Jumbled_MSO;
        end
        
        for m=1:Add_dummies
            [t, P_spk_dumb] = FiringRate(Dummy_MSO{m},'bin', (1/fs)*1e3, stim_dur,0);
            Wrapped_dumb = nan(Mseq_repeats,Mseq_samps);
            for n=1:Mseq_repeats
                Wrapped_dumb(n,:) = P_spk_dumb((n-1)*Mseq_samps+1:n*Mseq_samps);
            end
            P_spk_wrapd_dumb = mean(Wrapped_dumb);
            H_imp_dumb{m} = cconv(P_spk_wrapd_dumb,Mseq_i(end:-1:1));
        end
        
        NOSCOR_i.MSO.H_NF{k} = [NOSCOR_i.MSO.H_NF{k}, H_imp_dumb];
    end
    data_NOSCOR{i} = NOSCOR_i;     
end


lastwarn('')
save(['MseqAnalyzed_' date '.mat'],'data_NOSCOR','data_MOVN')
[warnMsg, warnID] = lastwarn;
if ~isempty(warnMsg)
    save(['MseqAnalyzed_' date '.mat'],'data_NOSCOR','data_MOVN','-v7.3')
end
end



             
        
    
    
    



