close all;
TW = 10;
ntaps = 2*TW - 1;
w = dpss(size(x, 2), TW, ntaps);
size(w)

ntrials = size(x, 1);
nfft = 2^ceil(log2(size(x, 2)));

x = x - repmat(mean(x, 2), 1, size(x, 2));
PLV = zeros(ntaps, nfft);
for k = 1:ntaps
    tap = repmat(w(:, k)', ntrials, 1);
    X = fft(x.*tap, nfft, 2);
    M = fft(Mseq_sig.*tap, nfft, 2);
    R = median(real(X .* conj(M)), 1);
    I = median(imag(X .* conj(M)), 1);
    PLV(k, :) = abs(R + 1j*I) ./ median(abs(X.*conj(M)), 1); %cross spectrum
end


nscramble = 8;

PLVscramble = zeros(nscramble, nfft);

for rep = 1:nscramble
    order = randperm(ntrials);
    odds = order(1:2:end);
    evens = order(2:2:end);
    
    y = zeros(size(x));
    y(1:numel(odds), :) = x(odds, :);
    y( (numel(odds)+1): (numel(odds) + numel(evens)), :) = -1*x(evens, :);
    
    PLVtemp = zeros(ntaps, nfft);
    for k = 1:ntaps
        fprintf(1, 'Doing taper # %d for scramble # %d\n', k, rep);
        tap = repmat(w(:, k)', ntrials, 1);
        X = fft(y.*tap, nfft, 2);
        M = fft(Mseq_sig.*tap, nfft, 2);
        R = median(real(X .* conj(M)), 1);
        I = median(imag(X .* conj(M)), 1);
        PLVtemp(k, :) = abs(R + 1j*I) ./ median(abs(X.*conj(M)), 1);
    end
    PLVscramble(rep, :) = mean(PLVtemp, 1);
    
end

sPLV = smooth(mean(PLV, 1), 40, 'sgolay');
figure; plot(f, sPLV); xlim([0, 50]);
hold on;
for rep = 1:nscramble
    sPLVscramble = smooth(PLVscramble(rep, :), 40, 'sgolay');
    plot(f, sPLVscramble, 'r'); xlim([0, 50]);
end
