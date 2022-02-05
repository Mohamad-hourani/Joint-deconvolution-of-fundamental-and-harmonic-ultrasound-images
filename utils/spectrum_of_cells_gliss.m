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
%-------------------------------------------------------------------------
% This function calculate the attneuation matrix by calculating the ratio
% between the energy harmonic and fundamental spectrum over a moving
% window all along the axial direction of the image

function [SF,SH,r,z]  =spectrum_of_cells_gliss(x,vertcellsize,horizcellsize,fs)
% n1 = size(x,1)/vertcellsize;
% if horizcellsize ==1
% n2=size(x,2);
% C = mat2cell(x,n1*ones(1,vertcellsize));
% else 
   n2 = size(x,2)/horizcellsize;
%     C = mat2cell(x,n1*ones(1,vertcellsize),n2*ones(1,horizcellsize));
% end
k=1;
a=vertcellsize/2;
for i =a+1:1:size(x,1)-a
        
        im = x(i-a:i+a-1,:);
        [r1,c1]=size(im); 
        w=window2(r1,c1,@blackman) ; 
        im=real(ifft2(fft2(im.*w)));
        Nfft =  2 .^ nextpow2(size(im));
        imF = fftshift(fft2(im, Nfft(1), Nfft(2)))/numel(im);
        %imF = fftshift(fft2(im', Nfft(1), Nfft(2)));
        fx=linspace(-fs/2,fs/2,Nfft(1));
        fy=linspace(-fs/2,fs/2,Nfft(2));
        f = (0:Nfft(2)/2-1)*fs/Nfft(2);
        [minDH1, inminhx1] = min(abs(fx-9e6));
        [minDH2, inminhx2] = min(abs(fx-11e6));
        [minDH2, inminfx1] = min(abs(fx-4e6));
        [minDH4, inminfx2] = min(abs(fx-6e6));
        [minDH5, inminy1] = min(abs(fy+10e6));
        [minDH6, inminy2] = min(abs(fy-10e6));
        SH(k) = sum(sum(abs(imF(inminhx1:inminhx2,inminy1:inminy2))))/numel(imF(inminhx1:inminhx2,inminy1:inminy2));
        SF(k) = sum(sum(abs(imF(inminfx1:inminfx2,inminy1:inminy2))))/numel(imF(inminfx1:inminfx2,inminy1:inminy2));
        r(k)=SH(k)/(SF(k));
        k=k+1;
       %figure,imagesc(10*log10(abs(imF)));colorbar,title('Spectrum of the fundmental and Harmonic after filtering')
       %figure; imagesc(fx, fy, 10*log10(abs(imF)));colorbar,title('Spectrum of the fundmental and Harmonic after filtering')
%         figure,plot(fx,10*log10(abs(imF)));
%         drawnow;
end
 e= zeros(size(x,1),1);
 e(1:a,:)=r(1);
 e(size(x,1)-a:end)=r(end);
 e(a+1:1:size(x,1)-a)=r;
 r=e;
 z=repmat(r(:),1,n2);
end

% for i = 2000:100:4000
%         im = C{i,j};
%         N =  2 .^ nextpow2(size(im));
%         imF = fftshift(fft(im, N(2)));
%         f = (0:N(2)/2-1)*fs/N(2); 
%         [minDH, inminhx1] = min(abs(f-9e6));
%         [minDH, inminhx2] = min(abs(f-11e6));
%         [minDH, inminfx1] = min(abs(f-4e6));
%         [minDH, inminfx2] = min(abs(f-6e6));
%         SH = sum(sum(abs(imF(1,inminhx1:inminhx2))))/numel(imF(1,inminhx1:inminhx2));
%         SF = sum(sum(abs(imF(1,inminfx1:inminfx2))))/numel(imF(1,inminfx1:inminfx2));
%         r(i)=SH/SF;
%         figure; plot( fy,10*log10(abs(imF)));
%         drawnow
% end
