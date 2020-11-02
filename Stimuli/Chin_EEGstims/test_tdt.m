clear all; close all hidden; clc;
[RP1, RP2, PA1,PA2,PA3,PA4,tdt_info] = tdt_init(99);

fs =48828.125;
t = 0:1/fs:1-1/fs;
y = sin(2*pi*750.*t);
y = rampsound(y,fs,.050);
y = scaleSound(y);

circuit_RP1 = 'Play_EEG.rcx';
circuit_RP2 = 'Play_EEG_RP2_2.rcx';

invoke(RP1,'Halt');
invoke(RP1,'ClearCOF');
invoke(RP1,'LoadCOF',circuit_RP1);

invoke(RP2,'Halt');
invoke(RP2,'ClearCOF');
invoke(RP2,'LoadCOF',circuit_RP2);


% invoke(RP1, 'SetTagVal', 'StmOn', 1000);
% invoke(RP1, 'SetTagVal', 'StmOff', 500);
%invoke(RP1, 'SetTagVal', 'RiseFall', 10);
invoke(RP1, 'SetTagVal', 'nsamps', length(y));
invoke(RP1,'WriteTagVEX','datainR',0,'F32',y);
invoke(RP1,'Run');
invoke(RP2,'Run');

%mixer selector
invoke(RP1,'SetTagVal','Select_L',0);
invoke(RP1,'SetTagVal','Connect_L',2);
invoke(RP2,'SetTagVal','Select_R',5);
invoke(RP2,'SetTagVal','Connect_R',1);

invoke(PA1,'SetAtten',0);
invoke(PA2,'SetAtten',120);
invoke(PA3,'SetAtten',0);
invoke(PA4,'SetAtten',25);

for i=1:100
    invoke(RP1,'SoftTrg',1);
    fprintf(1,'Trial #%d/%d \n',i,10)
    pause(1.5)
end
