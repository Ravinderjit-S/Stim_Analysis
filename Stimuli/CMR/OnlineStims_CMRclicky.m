% Generate Stimuli for onilne experiment on Prolific
clear

path = '../../CommonExperiment';
p = genpath(path);
addpath(p);

addpath('../StimDev')

SNR_0 = [-15, -28, -32, -36, -40, -44, -48];
SNR_1 = [-25, -36, -40, -44, -48, -52, -56];

fs = 44100;
tlen = 1;
t = 0:1/fs:tlen-1/fs;


