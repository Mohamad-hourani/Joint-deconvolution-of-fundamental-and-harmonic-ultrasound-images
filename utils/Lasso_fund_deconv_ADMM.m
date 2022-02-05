function [x,U,D,criterion]=Lasso_fund_deconv_ADMM(FB1,FBC1,F2B1,X1,U,D,beta,mu,p,tolA,objective,times,stoppingrule,maxiter)
for i = 1:maxiter
%     ISNR_admmSet(t,i) = ISNR_cal(y,refl,X); 
   %%%%%%%%%%%%%%%%%%%%%%%%%% update X %%%%%%%%%%%%%%%%%%%%%%%%%%%
   % argmin_x  1/2*||y1-H1x||^2 + beta/2*||X - U + D/beta||^2 
   num = FBC1.*fft2(X1)+ beta.*fft2(U)-beta.*fft2(D);
   denom= F2B1  + beta;
   FX = num./denom;
   x =real(ifft2(num./denom));

   %%%%%%%%%%%%%%%%%%%%%%%%%% update U %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % argmin_u mu||u||_1 + beta/2*||X - u + D/mu||^2  
  U = prox_lp(x+D,p,mu/beta);
%     U = max(abs(x+D)-beta/mu,0).*sign(x);
   %%%%%%%%%%%%%%%%%%%%%%%%%% update D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = D + (x-U);
 %%   
%    
    HX = ifft2(FB1.*FX);
    resid =  X1 - HX;
    objective(i+1) = 0.5*(resid(:)'*resid(:)) + mu*sum(abs(x(:)).^p);
    obj1(i+1) = (resid(:)'*resid(:));
    obj2(i+1) = mu*sum(abs(x(:)).^p);
    distance(i)=norm(x(:)-U(:),2);
%     err = x-refl;
%     mses(i) =  (err(:)'*err(:))/N;
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