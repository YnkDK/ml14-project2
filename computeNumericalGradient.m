function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  

numgrad = zeros(size(theta));



eps = 0.0001;
for i=1:length(numgrad)
    tmp = theta;
    tmp(i) = tmp(i)+eps;
    [v1,dum] = J(tmp);
    tmp(i) = tmp(i)-2*eps;
    [v2,dum] = J(tmp);
    numgrad(i) = (v1-v2)/(2*eps);

end
