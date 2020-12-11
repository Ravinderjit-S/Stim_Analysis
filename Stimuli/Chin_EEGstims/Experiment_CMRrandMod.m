clear all; close all hidden; clc;
addpath(genpath('CMR'))



[RP1, RP2, RX8, PA1,PA2,PA3,PA4,tdt_info] = tdt_init(99);

%% stim params
fs =48828.125;
mod40_coh0 = load('CMRrandmod_tpsd_40_coh_0.mat');
mod40_coh1 = load('CMRrandmod_tpsd_40_coh_1.mat');
mod223_coh0 = load('CMRrandmod_tpsd_223_coh_0.mat');
mod223_coh1 = load('CMRrandmod_tpsd_223_coh_1.mat');

stim_dur = size(mod40_coh1.Sig,2)/fs; %4 sec
ISI = 1.15;
jit = 0.100;
trials = size(mod40_coh1.Sig,1); %300
risetime = .050;

    

%% tdt setup
circuit_RP1 = 'Play_EEG.rcx';
circuit_RP2 = 'Play_EEG_RP2_2.rcx';
circuit_RX8 = 'RX8_triggers.rcx';

invoke(RP1,'Halt');
invoke(RP1,'ClearCOF');
invoke(RP1,'LoadCOF',circuit_RP1);

invoke(RP2,'Halt');
invoke(RP2,'ClearCOF');
invoke(RP2,'LoadCOF',circuit_RP2);

invoke(RX8,'Halt');
invoke(RX8,'ClearCOF');
invoke(RX8,'LoadCOF',circuit_RX8);

invoke(RP1,'Run');
invoke(RP2,'Run');
invoke(RX8,'Run');

%mixer selector
invoke(RP1,'SetTagVal','Select_L',1);
invoke(RP1,'SetTagVal','Connect_L',2);
invoke(RP2,'SetTagVal','Select_R',5);
invoke(RP2,'SetTagVal','Connect_R',3);

invoke(PA1,'SetAtten',0);
invoke(PA2,'SetAtten',0);
invoke(PA3,'SetAtten',25);
invoke(PA4,'SetAtten',25);

%% Play Stimuli

for i=1:trials
    stims{1} = mod40_coh0.Sig(i,:);
    stims{2} = mod40_coh1.Sig(i,:);
    stims{3} = mod223_coh0.Sig(i,:);
    stims{4} = mod223_coh1.Sig(i,:);
    
    order = randperm(length(stims));
    for j =1:length(order)
        stim = stims{order(j)}; 
        stim = rampsound(stim,fs,risetime);
        stim = scaleSound(stim);
        invoke(RP1, 'SetTagVal', 'nsamps', length(stim));
        invoke(RP1,'WriteTagVEX','datainR',0,'F32',stim);
        pause(0.1);
        invoke(RX8, 'SetTagVal', 'TrigVal', order(j));
        invoke(RP1,'SoftTrg',1);
        fprintf(1,'Trig %d \n',order(j))
        pause(stim_dur + ISI);
        if j == length(order)
            pause(jit*rand)
        end
    end
    fprintf(1,'Trial #%d/%d \n',i,trials)
end


