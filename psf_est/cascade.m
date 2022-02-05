function hj = cascade(wfil,J)

hj = wfil; % -- h1 value

for i=1:J - 1
    hj = conv(wfil,up(hj));
end
end


function y=up(x)
% UP inserts 0 between each element of the input vector 
n = numel(x);
y = [x(:)';zeros(1,n)]; 
y = y(:);
y = y(1:end-1)';
end