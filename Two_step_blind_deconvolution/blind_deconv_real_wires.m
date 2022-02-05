%=========================================================================%
%========   Joint deconvolution of fundanmental and harmonic      ========%
%========                ultrasound images                        ========%
%========           Code Author: Mohamad HOURANI                  ========%
%========          Version (Date): Feb,07 2020                    ========%
%========            Email: mohamad.hourani@irit.fr               ========%
%=========================================================================%
%------------------------      COPYRIGHT      ----------------------------%
% Copyright (2020): Mohamad HOURANI, Adrian Basarab, Denis Kouam\'{e}, and 
% Jean-Yves Tourneret;
% Permission to use, copy, modify, and distribute this software for any
% purpose without fee is hereby granted, provided that this entire notice
% is included in all copies of any software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or iplied
% warranty. Inparticular, the authors do not make any representation or
% warranty of any kind concerning the merchantability of this software or
% its fitness for any particular purpose.
%-------------------------------------------------------------------------%
%-----------------------      REFERENCE      -----------------------------%
% This set of MATLAB files contain an implementation of the algorithms
% described in the following paper:
%
% Mohamad Hourani, Adrian Basarab, Denis Kouam\'{e}, Jean-Yves Tourneret, 
% Guilia Matrone, Alessandro Ramalli,"On Ultrasound Image Deconvolution 
% using Fundamental and Harmonic Data"
%-------------------------------------------------------------------------%


%% add paths
addpath ./../data;
addpath ./../utils;
addpath ./../psf_est
%% Data 
load wires;
rf=rf;
%% Attenuation (exponential)
%[SF,SH,r,z]=spectrum_of_cells_gliss(rf./max(rf(:)),110,1,fs);
W=ones(size(rf));
Winv=1./W;
clear wires;  

load hwires;
load fwires;
rff=rff;
rff2=rf2;
y1 = rff /max(rff(:));
y2 = rf2 /max(rf2(:));
% [h_est1,ps,ph]=psf_est(y1(100:1100,100:200),0.86);                                         % Estimation of the fundamental PSF
% [h_est2,ps,ph]=psf_est(y2(100:1100,100:200),0.86);                                         % Estimation of the harmonic PSF
load wires_est_PSF;
H1=psf_est1/max(psf_est1(:));
H2=psf_est2/max(psf_est2(:));

%% Preparation of inputs 
N = numel(y1);
BSNRdb1 = 40;                                                              % Blur signal to noise ratio
BSNRdb2 = 40;                                                              % Blur signal to noise ratio
sigma1 = norm(y1-mean(mean(y1)),'fro')/sqrt(N*10^(BSNRdb1/10));            % Standard deviation of the WGN in fundamental image
sigma2 = norm(y2-mean(mean(y2)),'fro')/sqrt(N*10^(BSNRdb2/10));            % Standard deviation of the WGN in fundamental image
%% PSF estimation and operators
[FB1,FBC1,F2B1,Bpad1] = Operator_PSF(H1,y1);                               % Operator of the fundamental PSF
[FB2,FBC2,F2B2,Bpad2] = Operator_PSF(H2,y2);                               % Operator of the harmonic PSF
% Normalized data
%% Presenting inputs
figure,
subplot(121),imagesc(rf2bmode(y1,1));colormap(gray);title('Fundamental ultrasound image')
subplot(122),imagesc(rf2bmode(y2,1));colormap(gray);title('Harmonic ultrasound image Image (with attenuation)')
figure,subplot(121),imagesc(H1),title('Fundamental PSF')
subplot(122),imagesc(H2),title('Harmonic PSF')
%%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%----Joint Solution----%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1/2*||y1-H1x||^2 +1/2*||y2-WH2x||^2 + mu||x||_1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization of ADMM 
stoppingrule = 1;                                                          % stopping rule (1- difference between two objectives,  2- distances(norm value)
tolA = -inf;                                                               % Tolerance for stopping criterion
sig2n = sigma1^2;                                                          % Variance of noise
p =1;                                                                      % lp normin the prior
beta =5;                                                                   % AL Hyperparameter
taup =5e-2;                                                                    
mu  = taup*sig2n; %mu=[]                                                   % lp norm hyperparameter
nt= numel(mu);                                                             % number of the set of lp norm hyperparameter
maxiter = 50;                                                              % maximum number of iteration                           
objective = zeros(nt,maxiter);                                             % objective's initialization
times = zeros(1,maxiter);                                                  % time of ADMM iterations
% Inputs
X1 = y1;
X2 = y2;
U = y1;
D = 0*y1;
%% ADMM Iterations
tic
[x,U,D,criterion]=joint_deconv_ADMM(FB1,FB2,FBC1,FBC2,F2B1,F2B2,W,Winv,X1,X2,U,D,beta,mu,p,tolA,objective,times,stoppingrule,maxiter);
toc
%figure,plot(criterion);
x_joint= x./max(x(:));
l=rf2bmode(x_joint,1);
lx1=20*log10(abs(hilbert(x_joint)));
    
%%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%----LASSO FUNDAMENTAL----%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1/2*||y1-H1x||^2 + mu||x||_1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y=y1;
beta =5;
taup =4e7;
tau  = taup*sig2n; 
mu=tau;
X1 = y;
U = y;
D = 0*y;
[x,U,D,criterion]=Lasso_fund_deconv_ADMM(FB1,FBC1,F2B1,X1,U,D,beta,mu,p,tolA,objective,times,stoppingrule,maxiter);
x_lasso_fund= x./max(x(:));
%%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%----LASSO HARMONIC----%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1/2*||y2-WH2x||^2 + mu||x||_1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y=y2;
beta = 3;
taup =9e8;
tau  = taup*sig2n; 
mu=tau;
X2 = y;
U = y;
D = 0*y;
[x,U,D,criterion]=Lasso_har_deconv_ADMM(FB2,FBC2,F2B2,X2,U,D,beta,mu,p,tolA,objective,times,stoppingrule,maxiter);
x_lasso_har= x./max(x(:));
%% Results
pos= [1 700 2400 400]
H=figure,
set(H,'position',pos)
subplot(151),imagesc(rf2bmode(y1,0.2)),colormap('gray'),title('Fundamental B-mode image')
subplot(152),imagesc(rf2bmode(y2,0.2)),colormap('gray'),  title( 'Harmonic B-mode image' )
subplot(155),imagesc(rf2bmode(x_joint,0.2)),colormap('gray'),title('Joint solution ')
subplot(153),imagesc(rf2bmode(x_lasso_fund,0.2)),colormap('gray'),title('Lasso fundamental solution')
subplot(154),imagesc(rf2bmode(x_lasso_har,0.2)),colormap('gray'),  title( 'Lasso Harmonic solution' )
