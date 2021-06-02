clear
load('Sentences_48828.mat')

lpf = fir1(round(4/150 * fs),150/(fs/2));


for j=1:length(sentences)
    x = hilbert(sentences{j});
    x = filtfilt(lpf,1,abs(x));
    envs{j} = x;
    envs_4096{j} = resample(x,4096,fs);
end

t = 0:1/fs:length(envs{1})/fs-1/fs;
t2 = 0:1/4096:length(envs_4096{1})/4096-1/4096;

figure, hold on
plot(t,envs{1})
plot(t2,envs_4096{1})

save('sentEnv.mat','envs_4096')


