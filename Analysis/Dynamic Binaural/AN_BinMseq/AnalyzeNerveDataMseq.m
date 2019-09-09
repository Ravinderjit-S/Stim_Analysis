function [data_Processed] = AnalyzeNerveDataMseq(data,Num_dummies,MaxISI,TC_stuff,data_date)
% this function will estimate system functions for the NOSCOR mseq stim and
% MOVN mseq stim with noise floors. This is data from the nerve,so does a
% coincidence detection b/t left and right stim
%data should be only OSCOR or MOVN pictures

fs = 48828.125; %sampling rate of audio system
Screen_pixelLocs = [1986 1 1855 1001]; %makes figure the size of my second screen at Lyle ... saves the figure as a larger figure


if isfield(data{1}.Stimuli,'IACt')
    NOSCOR = 1;
else
    NOSCOR = 0;
end
    

for i = 1:numel(data)
    PicNums_data(i) = data{i}.General.picture_number; %#ok
end

%% Extract TC_stuff
BFs = TC_stuff{1}; BF_RT = TC_stuff{2}; thresh = TC_stuff{3};
Q10 = TC_stuff{4}; unit = TC_stuff{5}; A_freqs = TC_stuff{6}; A_thresh = TC_stuff{7};


%% Pic combining setup
[SamePics, SameStim, SameUnit] = FindSamePics(data);
for i = 1:numel(SamePics) % pic combining stuff
    Spics_i = SamePics{i};
    if length(Spics_i) >2
        error('Not accounting for more than 2 pics being same right now')
    else
        if data{Spics_i(1)==PicNums_data}.Stimuli.MSeq_StepDuration ~= data{Spics_i(2)==PicNums_data}.Stimuli.MSeq_StepDuration
            SamePics(i) = []; SameStim(i) = []; SameUnit(i) = [];
        end
    end
end

SameTrig = []; SameSkip = [];
for i = 1:numel(SamePics) % this is for pic combining
    SameTrig(i) = find(PicNums_data == SamePics{i}(1)); %#ok
    SameSkip(i) = find(PicNums_data == SamePics{i}(2)); %#ok
end

%% Data analysis Loop
for i =1:numel(data)
    if any(i == SameSkip) %skip these cause they were combnied with another picture
        data{i} = [];
        continue
    end
    
    data_i = data{i}; data_i = correct_bad_lines(data_i);
    spk_conds = ExtractConditions(data_i);
    %% Pic Combining
    if any(i == SameTrig) % doing pic combining
        data2_i = data{SameSkip(i==SameTrig)}; data2_i = correct_bad_lines(data2_i);
        spk_conds2 = ExtractConditions(data2_i);
        if numel(spk_conds) == numel(spk_conds2) && all(data_i.Stimuli.attens == data2_i.Stimuli.attens)
            for j =1:numel(spk_conds)
                spk_conds2{j}(:,1) = spk_conds2{j}(:,1) + spk_conds{j}(end,1);
                spk_conds{j} = vertcat(spk_conds{j},spk_conds2{j});
                data_i.fully_presented_lines = data_i.Stimuli.fully_presented_lines + data2_i.Stimuli.fully_presented_lines;
            end
        else %Could not be exactly the same so not combining 
            SameSkip(SameTrig==i) = [];
        end
    end
    %% Coincidence Detection to simulate MSO
    MSO_sim = cell(numel(spk_conds)/2,1);
    spksL = cell(numel(spk_conds)/2,1);
    spksR = cell(numel(spk_conds)/2,1);
    for j =1:numel(spk_conds)/2
        spkL = spk_conds{2*j-1};
        spkR = spk_conds{2*j};
        spksL{j} = spkL;
        spksR{j} = spkR;
        if isempty(spkL) || isempty(spkR)
            MSO_sim{j} = 'Not Enough Data';
        else
            MSO_sim{j} = Coincidence_detect(spkL,spkR, MaxISI);
        end
    end
    data_i.MSO.ISI = MaxISI;
    data_i.MSO.simulated = MSO_sim;
    data_i.MSO.LeftSpks = spksL;
    data_i.MSO.RightSpks = spksR;
    
    %% Stim Parameters
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

    %% Estimate Transfer Function
    for k = 1:numel(MSO_sim)
        if strcmp(MSO_sim{k}, 'Not Enough Data')
            continue
        end
        
        %% Noise Floors 
        for m=1:Num_dummies
            Dummy_MSO{m} = Jumble_spikes(MSO_sim{k},stim_dur/1000); %#ok
            [~, FRate_dumb] = FiringRate(Dummy_MSO{m},'bin', (1/fs)*1e3, stim_dur,0);
            Wrapped_dumb = nan(Mseq_repeats,Mseq_samps);
            for n=1:Mseq_repeats
                Wrapped_dumb(n,:) = FRate_dumb((n-1)*Mseq_samps+1:n*Mseq_samps);
            end
            FRate_wrapd_dumb = mean(Wrapped_dumb);
            H_imp_dumb{m} = cconv(FRate_wrapd_dumb,Mseq_i(end:-1:1)); %#ok
        end
       
        %% Real Transfer function
        [~,FRate] = FiringRate(MSO_sim{k}, 'bin', (1/fs)*1e3, stim_dur,0);
        Wrapped = nan(Mseq_repeats,Mseq_samps);
        for m=1:Mseq_repeats
            Wrapped(m,:) = FRate((m-1)*Mseq_samps+1:m*Mseq_samps);
        end
        FRate_wrapd = mean(Wrapped);
        [H_imp] = cconv(FRate_wrapd, Mseq_i(end:-1:1));
        
        data_i.MSO.H_imp{k} = H_imp;
        data_i.MSO.H_NF{k} = H_imp_dumb;

        Time_Impulse = 0.100;  %time of impulse to keep in seconds
        H_imp = H_imp(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
        phase = unwrap(angle(fft(H_imp)));
        f_phase = fs*(0:length(phase)-1)/length(phase);
        [H_f, f] = pmtm(H_imp,2.5,[],fs);
        
        %% Extract TC Parameters
        CF = BFs(unit == unit_i) *1000; %Hz
        data_i.tc.CFrealtime = BF_RT(unit==unit_i);
        data_i.tc.CFinterp = CF;
        data_i.tc.ThreshInterp = thresh(unit==unit_i);
        data_i.tc.rawFreqs = A_freqs{unit==unit_i};
        data_i.tc.rawThresh = A_thresh{unit==unit_i};
        
        %% Plotting
        attenuation = attens(k*2);
        num_fibs = MSO_sim{k}(end,1);
        MSO_spks = size(MSO_sim{k},1);
        picNum = data_i.General.picture_number;
        if NOSCOR
            fig_Tit = sprintf('Atten: %d, CF: %d MSOsimLines: %d MSOsimspks: %d, MSeqStepDur: %d, pic: %d',round(attenuation), round(CF), num_fibs, MSO_spks,Mseq_StepDur,picNum);
        else
            fig_Tit = sprintf('Atten: %d, CF: %d MSOsimLines: %d MSOsimspks: %d, MSeqStepDur: %d, ITD: %d, Pic: %d',round(attenuation), round(CF), num_fibs, MSO_spks,Mseq_StepDur, max(abs(Mseqt)),picNum);
        end
        
        ThisFig = figure('visible','off','Position',Screen_pixelLocs);
        subplot(3,1,1),plot((0:length(H_imp)-1)/fs * 1000, H_imp,'linewidth',2),title(['H(t)... ' fig_Tit]),xlabel('Time (ms)')
        hold on
        for m = 1:Num_dummies
            H_imp_dumb{m} = H_imp_dumb{m}(Mseq_samps+1:Mseq_samps+round(Time_Impulse*fs));
            plot((0:length(H_imp)-1)/fs * 1000,H_imp_dumb{m},'r')
        end
        hold off
        subplot(3,1,2),semilogx(f,pow2db(H_f),'linewidth',2), title('H(f)'), xlabel('Freq'), xlim([0 10000]), ylim([max(pow2db(H_f))-60, max(pow2db(H_f))])
        hold on
        for m = 1:Num_dummies
            [p_dumb,f] = pmtm(H_imp_dumb{m},2.5,[],fs);
            semilogx(f,pow2db(p_dumb),'r')
        end
        hold off
        subplot(3,1,3),plot(f_phase,phase,'linewidth',2),title('Phase(f)'), ylabel('phase (radians)'), xlim([0 500])
        
        if NOSCOR
            saveas(gcf,['Mseq_IACt_figs/' data_date '/Fig' num2str(i) '.' num2str(k) '.png'])
        else
            saveas(gcf,['Mseq_ITDt_figs/' data_date '/Fig' num2str(i) '.' num2str(k) '.png'])
        end
        close(ThisFig)
        
        ThisFig = figure('visible','off','Position',Screen_pixelLocs);
        subplot(3,2,1), plot(MSO_sim{k}(:,2),MSO_sim{k}(:,1),'.', 'MarkerEdgeColor',[0 0.5 0]), title('MSO simulated'), ylabel('Trial'), xlabel('Time (sec)')
        hold on, plot(stim_dur/1000*ones(1,length(MSO_sim{k}(:,1))),MSO_sim{k}(:,1),'r'), hold off %red vertical line indicating when stim stops
        subplot(3,2,3), plot(spksL{k}(:,2),spksL{k}(:,1),'.', 'MarkerEdgeColor',[0 0.5 0]), title('Nerve Left')
        hold on, plot(stim_dur/1000*ones(1,length(spksL{k}(:,1))),spksL{k}(:,1),'r'), hold off %red vertical line indicating when stim stops
        subplot(3,2,5), plot(spksR{k}(:,2),spksR{k}(:,1),'.', 'MarkerEdgeColor',[0 0.5 0]), title('Nerve Right')
        hold on, plot(stim_dur/1000*ones(1,length(spksR{k}(:,1))),spksR{k}(:,1),'r'), hold off %red vertical line indicating when stim stops
        
        [t, FRate] = FiringRate(MSO_sim{k}, 'bin', 3, stim_per,0);
        subplot(3,2,2), plot(t,FRate), ylabel('FiringRate'), xlabel('Time (sec)'), xlim([-0.1 8])
        [t, FRate] = FiringRate(spksL{k}, 'bin', 0.2, stim_per,0);
        subplot(3,2,4), plot(t, FRate),xlim([-0.1 8])
        [t, FRate] = FiringRate(spksR{k}, 'bin', 0.2, stim_per,0);
        subplot(3,2,6), plot(t, FRate),xlim([-0.1 8])
        
        if NOSCOR
            saveas(gcf,['Mseq_IACt_figs/' data_date '/Fig' num2str(i) '.' num2str(k) 'Raster.png'])
        else
            saveas(gcf,['Mseq_ITDt_figs/' data_date '/Fig' num2str(i) '.' num2str(k) 'Raster.png'])
        end
        close(ThisFig)
    end
    data_Processed{i} = data_i; %#ok
    
end
    
    
    
    
    
    
