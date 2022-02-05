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
load data_simu3;
refl=refl./max(refl(:));                                                   % Tissue reflectivity (build from MRI of kidney)
H1=H1/max(H1(:));
H2=H2/max(H2(:));
%% Attenuation (exponential)
alpha= 0.8*(7)^1.35;                                                       % Attenuation coefficient 
alpha_l =1*alpha/8.69;                                                     % Amplitude attenuation factor  (cm^(-1))
z = linspace(5,1000, size(refl,1))*1e-3 ;                                  % Depth (5mm)
Attenuation=exp(-alpha_l*z);                                               % Axial attenuation scheme
W=repmat(Attenuation',1,size(refl,2));                                     % Attenuation matrix 
Winv=1./W;                                                                 % Inverse of attenuation matrix 
%% Operator PSF
[FB1,FBC1,F2B1,Bx1,Bpad1] = HXconv(refl,H1);                               % Fundamental PSF operator
[FB2,FBC2,F2B2,Bx2,Bpad2] = HXconv(refl,H2);                               % Harmonic PSF operator
Bx2=Bx2.*W;                                                                % WHx (attenuated harmonic image)
%% Preparation of inputs 
N = numel(refl);
BSNRdb1 = 40;                                                              % Blur signal to noise ratio
BSNRdb2 = 40;                                                               % Blur signal to noise ratio
sigma1 = norm(Bx1-mean(mean(Bx1)),'fro')/sqrt(N*10^(BSNRdb1/10));          % Standard deviation of the WGN in fundamental image
sigma2 = norm(Bx2-mean(mean(Bx2)),'fro')/sqrt(N*10^(BSNRdb2/10));          % Standard deviation of the WGN in fundamental image
y1 = Bx1 + sigma1*randn(size(refl));                                       % Fundamental image+ WGN
y2 = Bx2 + sigma2*randn(size(refl));                                       % Harmonic image+ WGN
y2=y2./max(y2(:));                                                         % Normalized data
y1=y1./max(y1(:)); 
%% PSF estimation and operators
% [h_est1,ps,ph]=psf_est(y1,0.86);                                         % Estimation of the fundamental PSF
% [h_est2,ps,ph]=psf_est(y2,0.86);                                         % Estimation of the harmonic PSF
load h_est;
n=round(size(h_est1,1)/2);
[a,b]=size(H1);
[c,d]=size(H2);
H11=h_est1(n-round(a/2)+1:n+round(a/2)+1,n-round(b/2)+1:n+round(b/2)+1);
H22=h_est2(n-round(c/2)+1:n+round(c/2)+1,n-round(d/2)+1:n+round(d/2)+1);
H11=H11/max(H11(:));                                                       % Normalization of the fundamental PSF
H22=H22/max(H22(:));                                                       % Normalization of the harmonic PSF
[FB1,FBC1,F2B1,Bpad1] = Operator_PSF(H11,refl);                            % Operator of the fundamental PSF
[FB2,FBC2,F2B2,Bpad2] = Operator_PSF(H22,refl);                            % Operator of the harmonic PSF
% Normalized data
%% Presenting inputs
figure,
subplot(131),imagesc(rf2bmode(refl,1));colormap(gray);title('TRF Image'),
subplot(132),imagesc(rf2bmode(y1,1));colormap(gray);title('Fundamental ultrasound image')
subplot(133),imagesc(rf2bmode(y2,1));colormap(gray);title('Harmonic ultrasound image Image (with attenuation)')
figure,subplot(221),imagesc(H1),title('Orginal Fundamental PSF')
subplot(222),imagesc(H11),title('Estimated Fundamental PSF')
subplot(223),imagesc(H2),title('Orginal Harmonic PSF')
subplot(224),imagesc(H22),title('Estimated Harmonic PSF')
%%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%----Joint Solution----%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1/2*||y1-H1x||^2 +1/2*||y2-WH2x||^2 + mu||x||_1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization of ADMM 
stoppingrule = 1;                                                          % stopping rule (1- difference between two objectives,  2- distances(norm value)
tolA = -inf;                                                             % Tolerance for stopping criterion
sig2n = sigma1^2;                                                          % Variance of noise
p =1;                                                                      % lp normin the prior
beta =0.001*10^3;                                                           % AL Hyperparameter
taup =1e3;                                                                    
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
beta =1;
taup =1e8;
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
beta = 1;
taup =1e8;
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
subplot(161),imagesc(rf2bmode(refl,1)),colormap('gray'),title('Tissue reflectivity function')
subplot(162),imagesc(rf2bmode(y1,1)),colormap('gray'),title('Fundamental  B-mode image')
subplot(163),imagesc(rf2bmode(y2,1)),colormap('gray'),  title( 'Harmonic B-mode image' )
subplot(166),imagesc(rf2bmode(x_joint,1)),colormap('gray'),title('Joint solution')
subplot(164),imagesc(rf2bmode(x_lasso_fund,1)),colormap('gray'),title('Lasso Fundamental')
subplot(165),imagesc(rf2bmode(x_lasso_har,1)),colormap('gray'),  title( 'Lasso Harmonic')
