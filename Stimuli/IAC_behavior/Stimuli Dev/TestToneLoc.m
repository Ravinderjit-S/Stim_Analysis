
fs = 48828.125;
tone_dur = 0.02; %ms
t = 0:1/fs:tone_dur;
tone = sin(2*pi*850*t); 
tone = rampsound(tone,fs,.004);
tone = tone + 0.8*randn(1,length(tone));
tone = [tone;-tone]; % tone has IAC=-1


tone = horzcat(zeros(2, round(fs/2)),tone,zeros(2,round(fs/2))); 
sound(tone,fs)

figure,plot(tone')