% Create Sine-Wave speech from .wav file
clear
p = genpath('../../Stim_Dev');
addpath(p)

   
%% randomly grab wav file

path = '../../../../../CI_hackathon/CI-Hackathon-Samples-master/Audio samples for training/';
folds = split(ls(path)); folds = folds(1:end-1);
type_num = randi(length(folds)); 
type_num =4;
path_type = [path folds{type_num} '/'];
if type_num > 2
    ptype2 = split(ls(path_type)); ptype2 = ptype2(1:end-1);
    path_type = [path_type ptype2{randi(length(ptype2))} '/'];
end
wav_files = dir(path_type);
wav_file = [path_type wav_files(randi(length(wav_files)-2)+3).name];

[raw_audio, fs] = audioread(wav_file);
raw_audio = raw_audio ./max(abs(raw_audio));


t = 0:1/fs:length(raw_audio)/fs - 1/fs;

%% Make Tones

Tones_num = 12;
f_start = 500;
f_end = 8000;
ERB_spacing = [];

%[Tones_f, ERBspace] = Get_Tones(Tones_num, ERB_spacing, f_start, f_end); %returns tone frequencies 
Tones_f = [ 381.5       ,  453.1691644 ,  538.30220594,  639.42846884, ...
        759.55246374,  902.2431332 , 1071.73988665, 1273.07855542, ...
       1512.24100964, 1796.33288261, 2133.79468257, 2534.65256437, ...
       3010.81621139, 3576.43267808, 4248.30670581, 5046.4];

for i =1:length(Tones_f)
    sines(:,i) = sin(2*pi*Tones_f(i).*t);
end

%% Get envelopes
[bm, env, delay] = gammatoneFast(raw_audio,Tones_f,fs,true);
env = env/max(max(env)) + .0001;

bw_lp = 40;
lp_filt = dpss(floor(2*fs/bw_lp),1,1); %length = 2/bw (second) ... make time-bandwidth product a minimum 
lp_filt = lp_filt - lp_filt(1);
lp_filt = lp_filt / sum(lp_filt);

lp_env = filtfilt(lp_filt,1,max(raw_audio,0));
lp_env = lp_env ./ max(lp_env); 

lp_gam_env = filtfilt(lp_filt,1,env);
% lp_gam_env = lp_gam_env ./ max(lp_gam_env);

hilbert_env = abs(hilbert(max(raw_audio,0)));
hilbert_env = hilbert_env ./ max(hilbert_env);

env_ratio = lp_gam_env ./ sum(lp_gam_env,2);
env_ratio = env ./ sum(env,2);

sws1 = sum(env.*sines,2);
sws2 = sum(lp_gam_env.*sines,2);
sws3 = sum(hilbert_env.*sines.*env,2);
sws4 = sum(lp_env.*sines.*env,2);



%% Plot Stuff
figure,plot(t,hilbert_env), hold on
plot(t,lp_env)
title('Envelopes')

figure,plot(t,raw_audio/max(abs(raw_audio))), hold on
plot(t,hilbert_env)
plot(t,lp_env)
title('Signal and  envelopes')

figure()
ax1 = subplot(3,1,1); hold on
plot(t,sws1 / max(abs(sws1)))
plot(t,raw_audio / max(abs(raw_audio)),'color',[0.5,0.5,0.5]),title('SWS 1')

ax2 = subplot(3,1,2);hold on
plot(t,sws2 / max(abs(sws2)))
plot(t,raw_audio / max(abs(raw_audio)),'color',[0.5,0.5,0.5]),title('SWS 2');

ax3 = subplot(3,1,3);hold on
plot(t,sws1 / max(abs(sws1)))
plot(t,sws2 / max(abs(sws2))),title('SWS 1vs2');

linkaxes([ax1,ax2,ax3],'x');

figure()
ax1 = subplot(3,1,1); hold on
plot(t,sws3 / max(abs(sws3)))
plot(t,raw_audio / max(abs(raw_audio)),'color',[0.5,0.5,0.5]),title('SWS 3')

ax2 = subplot(3,1,2);hold on
plot(t,sws4 / max(abs(sws4)))
plot(t,raw_audio / max(abs(raw_audio)),'color',[0.5,0.5,0.5]),title('SWS 4');

ax3 = subplot(3,1,3);hold on
plot(t,sws3 / max(abs(sws3)))
plot(t,sws4 / max(abs(sws4))),title('SWS 3vs4');

linkaxes([ax1,ax2,ax3],'x');


figure()
ax1 = subplot(2,1,1);plot(t,env),title('gamma envs');
ax2 = subplot(2,1,2);plot(t,lp_gam_env),title('lp gamma envs');
linkaxes([ax1,ax2],'x');

figure,
subplot(2,1,1), spectrogram(sws1,round(.02*fs),round(.02*0.9*fs),1:10000,fs,'yaxis'), title('SWS 1')
subplot(2,1,2), spectrogram(sws2,round(.02*fs),round(.02*0.9*fs),1:10000,fs,'yaxis'), title('SWS 2')
figure, 
subplot(2,1,1),spectrogram(sws3,round(.02*fs),round(.02*0.9*fs),1:10000,fs,'yaxis'), title('SWS 3')
subplot(2,1,2), spectrogram(sws4,round(.02*fs),round(.02*0.9*fs),1:10000,fs,'yaxis'), title('SWS 4')
figure, spectrogram(raw_audio,round(.02*fs),round(.02*0.9*fs),1:10000,fs,'yaxis'), title('Raw Audio')


%% Play Sounds
soundsc(sws1,fs)
pause(t(end)+0.2)
soundsc(sws2,fs)
pause(t(end)+0.2)
soundsc(sws3,fs)
pause(t(end)+0.2)
soundsc(sws4,fs)
pause(t(end)+0.2)
soundsc(raw_audio,fs)









