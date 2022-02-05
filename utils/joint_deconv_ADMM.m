function [x,U,D,criterion]= joint_deconv_ADMM(FB1,FB2,FBC1,FBC2,F2B1,F2B2,W,Winv,X1,X2,U,D,beta,mu,p,tolA,objective,times,stoppingrule,maxiter)
for i = 1:maxiter
   %%%%%%%%%%%%%%%%%%%%%%%%%% update X %%%%%%%%%%%%%%%%%%%%%%%%%%%
   %  argmin_x  1/2*||y1-H1x||^2 +1/2*||y2-WH2x||^2+ beta/2*||X - U + D/beta||^2 
   % Solution in the Fourier domain
   num= FBC1.*fft2(X1) + FBC2.*fft2(Winv.*X2)+ beta.*fft2(U)-fft2(D);
   denom = F2B1+ F2B2+ beta*ones(size(X1));
   FX = num./denom;
   x =ifft2(num./denom);
   %%%%%%%%%%%%%%%%%%%%%%%%%% update U %%%%%%%%%%%%%%%%%%%%%%%%%%%
   % argmin_u mu||u||_1 + beta/2*||X - u + D/mu||^2 
   % Soft-Thresholding
    U = prox_lp(x+D,p,beta/mu);
   %%%%%%%%%%%%%%%%%%%%%%%%%% update D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %Update of the Lagrangian multiplier
    D = D + beta.*(x-U);
 %%   
%      
    HX1 = ifft2(FB1.*FX);
    HX2 = W.*ifft2(FB2.*FX);
    resid1 =  X1 - HX1;
    resid2 =  X2 - HX2;
    objective(i+1) = (resid1(:)'*resid1(:))+(resid2(:)'*resid2(:))...
        + mu*sum(abs(x(:)).^p);
    distance(i)=norm(x(:)-U(:),2);
    times(i) = toc;
    switch stoppingrule
        case 1      
            criterion(i) = abs(objective(i+1)-objective(i))/objective(i);
        case 2
            criterion(i) = distance(i);
    end

     if ( criterion(i) < tolA )
         break
     end
end
end
