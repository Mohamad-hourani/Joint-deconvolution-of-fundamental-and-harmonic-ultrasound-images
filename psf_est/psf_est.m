function [ h,ps,ph ] = psf_est(rf,ncf)
addpath ./../PSFest_Michailovich;
%|PSF estimation
%|reference: Oleg Michalovich 2005 IEEE Medical Processing
%|input
%      rf: the observated image, rf=conv2(x,h)
%|output
%      h: the estimation of PSF
%      ps: amplitude spectrum
%      ph: phase spectrum
%%  
%rf = filt_rf(rf,0.16);
rf = filt_rf(rf,ncf);
[Nl,Nc]=size(rf); 
%zero padding to get a rectangular RF image
%the number of lines is always larger than the number of columns in RF ims
rf=padarray(rf,[0 Nl-Nc],'post');
N=Nl;

RF=fft2(rf);
logSpec_RF=log(abs(RF)+1e-7);
phase_RF=angle(RF);
%% estimation of the PSF amplitude 
% -Step1: outliers shringkage
amp=outliers_removal(logSpec_RF);
% -Step2: cycle-spinning de-noising / wavelet shringkage denoising is also can
% used here
LogSpecFilt=filter_wavelet(amp,4);
% - estimation
ps=exp(LogSpecFilt);
ps=(ps-min(ps(:)))/(max(ps(:))-min(ps(:)));
%% estimation of the PSF amplitude
[ph]=phase_est(phase_RF,N);
%% 
h = ifft2( ps.*exp(1j*ph) );
h = real(h);
h = fftshift(h);
end


function data_out=outliers_removal(data_in)
N=numel(data_in);
lambda=1;
% - periodic padding and median filtering
data_in_pad = padarray(data_in,[2 2],'symmetric','both');
Y = medfilt2( data_in_pad,[3 3] );
Y = Y(3:end-2,3:end-2);

Diff = data_in-Y;
absDiff = abs(Diff);
R = sort(absDiff(:));
th = R( round( N*lambda) );
Residuals = zeros(size(data_in));
idx = find( absDiff > th );
Residuals(idx) = sign( Diff(idx) ).*( absDiff(idx) - th );
data_out = data_in - Residuals;
debug()
end


function y = filter_wavelet(x,evmom)
J=1;
if ~exist('evmom','var')
    evmom = 2;
end

wname = ['db' num2str(evmom)];

% -- building 2D wavelet filter
[n m] = size(x);
Lo_D = wfilters(wname);
hJ = cascade(Lo_D,J);
vJ = 2^(-J)*conv( hJ,hJ(end:-1:1) );
r = numel(vJ);
vJ = vJ(:);
vJ = vJ*vJ';
 
xpad = padarray(x,[floor(r/2) floor(r/2)] , 'symm' , 'both' );
[npad mpad] = size(xpad);
vJ = padarray(vJ , [npad-r mpad-r] , 'post' );
vJ = circshift(vJ , [-floor(r/2) -floor(r/2)]);

% - filtering
ypad = real( ifft2( fft2(vJ).*fft2(xpad) ) ); 
y = ypad(floor(r/2)+1:floor(r/2)+n,floor(r/2)+1:floor(r/2)+m);
debug()
end


function [phase]=phase_est(phase_RF,N)
% Riesz bases constructed using cubic B-splines
J=2;  %resolution parameter \in -Z
M=2^(J+1);  % equals to # basis functions

PHI=b3splines(N,J); %M^2 basis functions
phi=PHI(:,1);
psi=gradient(phi);
delta=N/M;
power_phi=(abs(fft(phi))).^2;
power_psi=(abs(fft(psi))).^2;
S1=power_phi;
S2=power_psi;
for k=0:delta-1
    power_phi=circshift(power_phi,-M);
    S1=S1+power_phi;
    power_psi=circshift(power_psi,-M);
    S2=S2+power_psi;
end
S1=S1/delta;
S2=S2/delta;
Fw=1./S1.*power_phi;
Fr=1./S2.*fft(psi).*conj(fft(phi));
for i=0:delta-1
    Fr(i*M+1)=0;
end
w=ifft(Fw);
r=ifft(Fr);

[phix,phiy]=gradient(phase_RF);
phix=outliers_removal(phix);
phiy=outliers_removal(phiy);

px=conv(r,sum(phix,1)/N,'same');
pxx=repmat(px',N,1); %identical rows
py=conv(r,sum(phiy,2)/N,'same');
pyy=repmat(py,1,N);%identical columns
etax = ifft2(fft(r)*fft(w)');
etay=ifft2(fft(w)*fft(r)');

psix= conv2(phix,etax,'same');
psiy=conv2(phiy,etay,'same');
phase = (psix+psiy+pxx+pyy)/(2*delta^2);
end



