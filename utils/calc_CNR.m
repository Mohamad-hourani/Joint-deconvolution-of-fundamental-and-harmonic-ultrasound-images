function [CNR, CNR_max, mu_1, mu_2, var_1, var_2] = calc_CNR(R1,R2)
%% CALC_CNR Calculates the contrast-to-noise ratio.
%  [CNR CNR_max mu_in mu_out var_in var_out] = calc_CNR(Xminmax_in,Yminmax_in,Xminmax_out,Yminmax_out,figure_nb)
%
%     R1 : First region of interest
%     R2 : Second region of interest
%
%  ----------------------
% | Code:  Renaud Morin  |
% | Email: morin@irit.fr |
% | Date:  May 2011      |
%  ----------------------


clear CNR


%% Checking arguments validity
if nargin<2
    error('2 two region are needed as first inputs')
elseif nargin > 6
   error('Too many input arguments.')
end

%% Processing
%% mean of regions
mu_1=mean2(R1);
mu_2=mean2(R2);
%% Standard deviation of regions 
var_1=std2(R1)^2;
var_2=std2(R2)^2;
%% Computing CNRs
CNR= abs(mu_1-mu_2)/sqrt(var_1+var_2);
CNR= 20*log10(CNR);
CNR_max=abs(mu_1)/sqrt(var_2);
fprintf('Le CNR est de %.3f \n',CNR)
% keyboard