load('PST_pic.mat')

ADdata_fs = 15e3; 
Stim_dur = PST_pic.Stimuli.Gating.Duration / 1000; %sec
Stim_per = PST_pic.Stimuli.Gating.Period / 1000; %sec
ADdata = PST_pic.ADdata.RawTrace;
ADdataTrigs = PST_pic.ADdata.triggerSamples;
t = 0:1/ADdata_fs:length(ADdata)/ADdata_fs - 1/ADdata_fs;
BPfilt = designfilt('bandpassiir', 'StopbandFrequency1', 200, 'PassbandFrequency1', 300, 'PassbandFrequency2', 2900, 'StopbandFrequency2', 3000, 'StopbandAttenuation1', 30, 'PassbandRipple', 1, 'StopbandAttenuation2', 30, 'SampleRate', ADdata_fs);
lenAD = length(ADdata);
ADdata = vertcat(zeros(size(ADdata)), double(ADdata), zeros(size(ADdata)));
ADdata = filtfilt(BPfilt, ADdata);
ADdata = ADdata(lenAD+1:2*lenAD);

Per_samps = Stim_per * ADdata_fs; 

figure,plot(t,ADdata)
hold on
plot(t(ADdataTrigs),ADdata(ADdataTrigs),'rx')
plot(t(ADdataTrigs+Per_samps), ADdata(ADdataTrigs+Per_samps),'kx')



