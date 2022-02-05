function modeB = rf2bmode(RF, increase)
%% rf2modeb Converts a RF image into a mode B one.
%  modeB = RF2modeB(RF, increase)
%
%         RF: RF image
%   increase: increase factor to adjust contrast
%
%  ----------------------
% | Code:  Renaud Morin  |
% | Email: morin@irit.fr |
% | Date:  June 2011     |
%  ----------------------


clear modeB


%% Initialization
if nargin==1
    increase=0;
elseif nargin<1 || nargin>2
    error('There must be 1 or 2 arguments.')
end


%% Computation
modeB=20*log(abs(hilbert(RF))+increase);
max_modeB=max(max(modeB));
modeB = 255.*modeB/max_modeB;