fs = 44100;
f1 = 1000;
f2 = 4000;
tlen = 1;
mod_cuts = [30 40];
bp_fo = 1/2 * 5 *fs; %filter order for slowest modulation instance ... keep same sharpness for all modulation filters so setting here
tinit = .2;
tincoh = .100;


[stim, mods] = RandMod_Coherence(f1,f2,mod_cuts,bp_fo,fs,tlen,tinit,tincoh);

t=0:1/fs:1-1/fs;
figure,plot(t,mods{3}')
figure,plot(t,mods{2}')

soundsc(stim{1},fs)
pause(tlen+0.5*tlen)
soundsc(stim{2},fs)
pause(tlen+0.5*tlen)
soundsc(stim{3},fs)







