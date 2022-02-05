function [S] = b3splines(N,J)

% B3SPLINES generates a basis of periodically shifted cubic B-splines. In
% particular, the function returns an N-by-M matrix, where M=2^(J+1), each
% column of which contains a shifted version of a cubic B-spline. Thus the
% total number of shifts (or, equivalently, # basis functions) is M.
%
%                         [S] = b3splines(N,J) 
% Input:
%       N - number of data points at which the splines are to be evaluated
%       J - defines the number of basis functions, i.e. M = 2^(J+1).
% 
% Output:
%       S - basis matrix
%
% written by Oleg Michailovich, September 24th, 2008.

x=(1/N)*(0:N-1)';
M=2^(J+1);
dx=1/M;

S=zeros(N,M);
for k=0:M-1,
    S(:,k+1)=evalcubic(M*(x-k*dx));
end
S(:,1)=S(:,1)+evalcubic(M*(x-M*dx));
S(:,2)=S(:,2)+evalcubic(M*(x-(M+1)*dx));
S(:,M)=S(:,M)+evalcubic(M*(x+dx));

%--------------------------------------------------------------------------
function [s] = evalcubic(x)

x=x(:);
X=[x+2 x+1 x x-1 x-2];
s=(abs(X).^3)*([1; -4; 6; -4; 1]/12);