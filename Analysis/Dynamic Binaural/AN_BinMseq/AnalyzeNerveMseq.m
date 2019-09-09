%function [] = AnalyzeNerveMseq(date)
clear all
addpath('../../Neuron Analysis Functions')
addpath('StimFuncs')

data_date = '10.12.18';

load(['AN_' data_date '.mat']); %loads 'data'

if strcmp(data_date,'10.01.18')
    data{35} = []; %garbage tuning curve, getting rid of on front end for now
    data{37} = []; %also couldn't get any data from this unit so just removing for now
end
if strcmp(data_date,'10.12.18')
    data{6} = []; data{7} = []; data{8} = []; data{9} = []; data{10} = []; data{11} = []; %CN unit
    data{37} = []; %irrelevant tuning curves (no extra data on them) ... probably was just searching or losing and getting fiber... looks like pic 39 is a stable recording of the attempt at these 2 pics
    data{38} = [];
end

fs = 48828.125;
[BFs, BF_RT, thresh, Q10, unit,A_freqs,A_thresh] = plotTCs(data);
TC_stuff = {BFs, BF_RT, thresh, Q10, unit, A_freqs, A_thresh};

Num_dummies = 25;
MaxISI = 0.00005; %50 us

[data_NOSCOR] = Extract_StimData(data, 'OSCOR ');
less_ind = 0;
for i = 1:numel(data_NOSCOR)
    i = i - less_ind;
    if strcmp(data_NOSCOR{i}.Stimuli.OSCORtype, 'M-Sequence ')
        continue
    else
        data_NOSCOR(i) = [];
        less_ind = less_ind+1;
    end
end

[data_NOSCOR] = AnalyzeNerveDataMseq(data_NOSCOR,Num_dummies,MaxISI,TC_stuff,data_date);

[data_MOVN] = Extract_StimData(data, 'MOVN ');
[data_MOVN] = AnalyzeNerveDataMseq(data_MOVN,Num_dummies,MaxISI, TC_stuff,data_date);

%NerveMseq_ConvergenceAnalysis(data_NOSCOR)

lessI = 0;
for i =1:numel(data_NOSCOR)
    i = i-lessI;
    if ~isstruct(data_NOSCOR{i})
        data_NOSCOR(i) = [];
        lessI = lessI+1;
    end
end

lessI = 0;
for i =1:numel(data_MOVN)
    i = i - lessI;
    if ~isstruct(data_MOVN{i})
        data_MOVN(i) = [];
        lessI = lessI+1;
    end
end      

save(['MseqAnalyzed_' data_date '.mat'], 'data_NOSCOR', 'data_MOVN')


 
