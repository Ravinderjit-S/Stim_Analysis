function [envelope] = create_envelope_ifft(bw,Tlen,fs)
%This function creates a random envelope in the frequency domain. More
%efficient than time domain. Also not doing half wave rectification because
%that can add low frequency energy which is undesirable here.

    noise_mod = makeNBNoiseFFT(diff(bw),mean(bw),Tlen,fs,0,0);
    noise_mod = noise_mod - min(noise_mod);
    envelope = noise_mod / max(noise_mod);
    envelope = envelope';

end