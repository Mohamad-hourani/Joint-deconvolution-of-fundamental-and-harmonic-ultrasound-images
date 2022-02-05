%% Computation of the Resolution Gain (RG)
%
%   input: input image
%       x: superresolved image
%
%  ----------------------
% | Code:  Renaud Morin  |
% | Email: morin@irit.fr |
% | Date:  Nov. 2011     |
%  ----------------------
function [RG]=resolution_gain(input,x)

% Autocorr
% xx=xcorr2(log(abs(hilbert(input))));
% yy=xcorr2(log(abs(hilbert(x))));
xx=xcorr2(input);
yy=xcorr2(x);
% Normalisation / max
xx=xx/max(xx(:));
yy=yy/max(yy(:));

% Pixel > seuil
seuil=0.70;
xx_nb=sum(xx(:)>seuil);
yy_nb=sum(yy(:)>seuil);

% Normalisation / nombre de pixel
xx_nb=xx_nb/numel(xx);
yy_nb=yy_nb/numel(yy);

% Resolution gain
RG=xx_nb/yy_nb;
fprintf('Le gain en résolution est de %.3f \n', RG)