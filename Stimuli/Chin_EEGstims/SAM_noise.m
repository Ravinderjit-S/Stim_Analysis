function stim = SAM_noise(AMf,tlen,fs)

t = 0:1/fs:tlen-1/fs;
noise = randn(1,length(t));

AM = 0.5 + 0.5*sin(2*pi.*AMf.*t);

stim = noise.*AM;

end

