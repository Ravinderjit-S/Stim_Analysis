%Stim to test binding using sentences
%First 2 letters of speaker is dialect region of talker, 3rd character is
%gender
clear 

pnncv2_loc = '/media/ravinderjit/Data_Drive/pnnc-v2/speakers';

speakers = {'NCF011', 'NCF012', 'NCF013', 'NCF014', 'NCF015', 'NCF017', ...
    'NCM012','NCM013', 'NCM014', 'NCM015', 'NCM016', 'NCM017', 'NCM018', ...
    'PNF133', 'PNF135', 'PNF136', 'PNF137', 'PNF139', 'PNF140', 'PNF142', ...
    'PNF143', 'PNF144', 'PNM055', 'PNM077', 'PNM078', 'PNM079', 'PNM080', ...
    'PNM081', 'PNM082', 'PNM083', 'PNM084', 'PNM085' 'PNM086'};


[sen, fs] = audioread('/media/ravinderjit/Data_Drive/pnnc-v2/speakers/NCF011/audio/NCF011_01-01.wav');

resample_sen = resample(sen,48828/4,44100/4);

Tones_num = 16;
ERB_spacing = [];
f_start = 100;
f_end = 8000;
[Tones_f, ERBspace] = Get_Tones(Tones_num, ERB_spacing, f_start, f_end);

[bm, env] = gammatoneFast(sen,Tones_f,fs);

[pxx,f] = pmtm(sen,4,2^17,fs);
[pxx2,f2] = pmtm(resample_sen,4,2^17,48828);

figure,plot(f,pxx,'b',f2,pxx2,'r')
xlabel('Freq (Hz)')
ylabel('Power')
legend('Original','UpSampled')


soundsc(sum(bm(:,10:16),2),fs)

