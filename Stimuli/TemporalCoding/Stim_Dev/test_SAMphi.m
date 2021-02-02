f1 = 1000;
f2 = 4000;
fs = 44100;
tlen = 1.0;
fm = 32;
base_phi = 90;
phi_deg = 50;
dichotic =0;
ref =1;

[stim] = SAM_phi_incoh(f1,f2,fs,tlen,fm,base_phi,phi_deg,dichotic,ref);

figure,
plot(stim{1}')

figure,
plot(stim{2}')

figure,
plot(stim{4}')


for j = 1:4
    soundsc(stim{j},fs)
    pause(1.3)
end


