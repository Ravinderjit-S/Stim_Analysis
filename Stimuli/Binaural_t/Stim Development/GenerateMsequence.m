clear all
load('Seed156.mat'); %loads variable 's' which will set rng
rng(s);


Mseq = mls2(8,0,s);

save('Mseq_IAC_ITD.mat','Mseq')


Mseq_bs = mls2(9,0,2);
save('Mseq_IAC_ITD_bs.mat','Mseq_bs')
