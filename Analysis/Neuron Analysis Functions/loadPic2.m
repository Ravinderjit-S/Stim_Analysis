function [x fname] = loadPic2(picNum)     % Load picture
picSearchString = sprintf('p%04d*.m', picNum);
picMFile = dir(picSearchString);
if (~isempty(picMFile))
   eval( strcat('x = ',picMFile.name(1:length(picMFile.name)-2),';') );
   fname = picMFile.name(1:length(picMFile.name)-2);
else
   error = sprintf('Picture file p%04d*.m not found.', picNum)
   x = [];
   fname = [];
   return;
end
