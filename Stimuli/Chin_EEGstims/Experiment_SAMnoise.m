clear all; close all hidden; clc;

[RP1, RP2, RX8, PA1,PA2,PA3,PA4,tdt_info] = tdt_init(99);

%% stim params
fs =48828.125;
tlen = 0.2;
AMf = [20, 30, 40, 55, 70, 90, 110, 170, 250, 400, 600, 800, 1000, 3000];
t = 0:1/fs:tlen-1/fs;
ISI = 0.10;
jit = 0.050;
trials = 300;
risetime = .005;

for s = 1:length(AMf)
    stim_AMf{s} = SAM_noise(AMf(s),tlen,fs);
end
    

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
    order = randperm(length(stim_AMf));
    for j =1:length(order)
        stim = stim_AMf{order(j)}; 
        stim = rampsound(stim,fs,risetime);
        stim = scaleSound(stim);
        invoke(RP1, 'SetTagVal', 'nsamps', length(stim));
        invoke(RP1,'WriteTagVEX','datainR',0,'F32',stim);
        invoke(RX8, 'SetTagVal', 'TrigVal', order(j));
        invoke(RP1,'SoftTrg',1);
        fprintf(1,'Trig %d \n',order(j))
        pause(tlen + ISI);
        if j == length(order)
            pause(jit*rand)
        end
    end
    fprintf(1,'Trial #%d/%d \n',i,trials)
end


