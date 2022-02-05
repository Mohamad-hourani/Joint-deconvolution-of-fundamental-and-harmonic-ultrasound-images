function [ PSNR ] = calcPSNR(x_hat,x ,x_max)

MAX_X = x_max;
N = prod(size(x));
PSNR = 10*log10( (N*MAX_X^2) / sum(sum((x_hat-x).^2)) );

   
end

