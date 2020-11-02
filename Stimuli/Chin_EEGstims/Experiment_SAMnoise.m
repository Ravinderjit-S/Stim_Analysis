clear all; close all hidden; clc;

[RP1, RP2, PA1,PA2,PA3,PA4,tdt_info] = tdt_init(99);

%% stim params
fs =48828.125;
tlen = 1;
AMf = 5;
t = 0:1/fs:tlen-1/fs;
ISI = 0.5;
jit = 0.1;
trials = 300;

%% tdt setup
circuit_RP1 = 'Play_EEG.rcx';
circuit_RP2 = 'Play_EEG_RP2_2.rcx';

invoke(RP1,'Halt');
invoke(RP1,'ClearCOF');
invoke(RP1,'LoadCOF',circuit_RP1);

invoke(RP2,'Halt');
invoke(RP2,'ClearCOF');
invoke(RP2,'LoadCOF',circuit_RP2);

invoke(RP1,'Run');
invoke(RP2,'Run');

%mixer selector
invoke(RP1,'SetTagVal','Select_L',0);
invoke(RP1,'SetTagVal','Connect_L',0);
invoke(RP2,'SetTagVal','Select_R',5);
invoke(RP2,'SetTagVal','Connect_R',3);

invoke(PA1,'SetAtten',0);
invoke(PA2,'SetAtten',120);
invoke(PA3,'SetAtten',0);
invoke(PA4,'SetAtten',25);

%% Play Stimuli

for i=1:trials
    stim = SAM_noise(AMF,tlen,fs);
    invoke(RP1, 'SetTagVal', 'nsamps', length(stim));
    invoke(RP1,'WriteTagVEX','datainR',0,'F32',stim);
    invoke(RP1,'SoftTrg',1);
    fprintf(1,'Trial #%d/%d \n',i,trials)
    pause(tlen + ISI + jit*rand());
end