function [bands] = CMRbands(center_f, ERB_halfwidth, ERBspacing)
%ERBwidth is bandwidth of noise
%ERBspacing is spacing between edges of noise bands

bands = zeros(3,2);
erbs = ERBs_human(center_f);

bands(2,:) = [FreqERBs_human(erbs-ERB_halfwidth) FreqERBs_human(erbs+ERB_halfwidth)];
bands(1,:) = [FreqERBs_human(ERBs_human(bands(2,1)) - ERBspacing - 2*ERB_halfwidth) FreqERBs_human(ERBs_human(bands(2,1)) - ERBspacing)]; 
bands(3,:) = [FreqERBs_human(ERBs_human(bands(2,2)) + ERBspacing) FreqERBs_human(ERBs_human(bands(2,2)) + ERBspacing + 2*ERB_halfwidth)]; 



end

function [erbs] = ERBs_human(f)
% Smith, Julius O.; Abel, Jonathan S. (10 May 2007). "Equivalent 
%Rectangular Bandwidth". Bark and ERB Bilinear Transforms. Center for
%Computer Research in Music and Acoustics (CCRMA), Stanford University,USA
%f is in hz
    erbs = 21.4 * log10(1 + 0.00437*f);  % (Moore & Glasberg, 1996)
end

function [freq] = FreqERBs_human(ERBs)
%This function will return the frequency for a particular ERBs
%Just the inveresee of function ERBs_human(f)
    freq = (10.^(ERBs/21.4) - 1) / .00437;
end