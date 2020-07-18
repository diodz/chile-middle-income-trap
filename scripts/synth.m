%% We define function to minimize

function f = synth(W, Z1, Z0)

% The function will expect a vector W of length equal to the number of 
% controls and an intercept in the last value
% It will also expect a matrix Z1 with the outcome variable for the treated 
% country for the pre-treatment periods
% Finally, a matrix Z0 with the outcome variable for the un-treated (J-1) 
% countries for the pre-treatment periods

n = length(W);
aux = 1;

f = (Z1 -W(n)*aux- Z0*W(1:(n-1))).'*(Z1 -W(n)*aux- Z0*W(1:(n-1)));
