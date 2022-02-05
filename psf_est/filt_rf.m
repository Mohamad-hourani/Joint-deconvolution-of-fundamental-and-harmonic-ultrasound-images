function y = filt_rf(x,ncf)
[B,A] = butter(3,ncf,'low');
y = filtfilt(B,A,x);

% [H] = filt1D(size(y,2),.6,1e-4);
% H = ones(size(y,1),1)*H';
% y = real(ifft2(fft2(y).*H));