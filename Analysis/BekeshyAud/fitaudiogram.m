function [L_fitted, f_fitted] = fitaudiogram(f, L)

f_fitted = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, ...
    6000, 8000, 9000, 10000];

ord = 3;
framelen = 99;
[u, ~, ~] = getReversals(L);

i1 = (f > 250);
i2 = (f >= f(u(2)));
inds = i1 & i2;

Lsmooth = sgolayfilt(L(inds), ord, framelen);
L_fitted = interp1(log(f(inds)), Lsmooth, log(f_fitted), 'spline');
L_fitted(1) = L_fitted(2);

% Round to nearest 5 dB as in clinic
L_fitted = 5 * round(L_fitted/5);

%% Print fit results
% fprintf(1, '\n----------------------------\n');
% fprintf(1, 'Bekesy Research Audiogram Results:\n');
% fprintf(1, '----------------------------\n');
% fprintf(1, 'Freq(kHz)\tThresh(dB)\n');
% fprintf(1, '----------------------------\n');
% for kf = 1:numel(f_fitted)
%     fprintf(1, '%0.3f\t\t%d\n', f_fitted(kf)/1e3, L_fitted(kf));
% end
% fprintf(1, '----------------------------\n');

