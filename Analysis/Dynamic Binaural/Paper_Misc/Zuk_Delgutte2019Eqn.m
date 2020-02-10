% See Zuk & Delgutte 2019 J of Neurophys
% Want detection threshold (50% point). Paper gives 75% point, slope, and
% model, so using that to get 50% point)

clear all;

% d75 = 43.4;
% slope = 20;
% tau = slope * (4/50);
% 
% d = 51 : 1: 99;
% PC = 50 / (1 + exp( - (log2(d) - log2(d75)) / tau)) + 50;
% figure, plot(d,PC)


slope = 20;
tau = slope * (4/50);
PC = 60;
d75 = 43.4;

PC = 51:99;
d1 = d75 ./ 2 .^(log((50./(PC-50))-1)*tau)
figure,plot(d1,PC)
set ( gca, 'xdir', 'reverse' )

slope = 24.9;
tau = slope * (4/50);
PC = 60;
d75 = 232.3;

d2 = d75 / 2 ^(log((50/(PC-50))-1)*tau)
