clear all; close all hidden; clc;
[RP1, RP2,RX8,PA1,PA2,PA3,PA4,tdt_info] = tdt_init(99);

fs =48828.125;
t = 0:1/fs:1-1/fs;
stim = sin(2*pi*750.*t);
stim = rampsound(stim,fs,.050);
stim = scaleSound(stim);

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
invoke(RX8,'LoadCOF',circuit_RX8)

invoke(RP1,'Run');
invoke(RP2,'Run');
invoke(RX8,'Run');



%mixer selector
invoke(RP1,'SetTagVal','Select_L',1);
invoke(RP1,'SetTagVal','Connect_L',2);
invoke(RP2,'SetTagVal','Select_R',5);
invoke(RP2,'SetTagVal','Connect_R',3);

invoke(PA1,'SetAtten',0); %keep 0
invoke(PA2,'SetAtten',0); % keep 0
invoke(PA3,'SetAtten',10);
invoke(PA4,'SetAtten',0);

invoke(RP1, 'SetTagVal', 'nsamps', length(stim));
invoke(RP1,'WriteTagVEX','datainR',0,'F32',stim);

for i=1:100
    invoke(RX8, 'SetTagVal', 'TrigVal', i);
    invoke(RP1,'SoftTrg',1);
    fprintf(1,'Trial #%d/%d \n',i,100)
    pause(1.5)
end