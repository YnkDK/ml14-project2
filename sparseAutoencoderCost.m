function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
% visibleSize: the number of input units 
% hiddenSize: the number of hidden units  
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units
% (greek letter rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Data store in column (DIFFERENT FROM H1) data(:,i) is the i-th training example.
%
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of neural nets (sparse encoder note). 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% shortcut constants
sp = sparsityParam;
m1 = 1/size(data,2);

% forward pass
a1 = data;
s2 = W1*a1;
s2 = bsxfun(@plus,b1,s2);
a2 = sigmoid(s2);
s3 = bsxfun(@plus,b2,W2*a2);
a3 = sigmoid(s3);

%cost 
cost = (1/2)*m1*sum(sum((data-a3).^2));
% weight decay
regcost = lambda/2 * (sum(sum(W1.^2))+sum(sum(W2.^2)));
cost = cost + regcost;
% sparsity cost
esp = mean(a2,2);%sample sparsity (estimated sparsity)
kbl = sp * log((sp./esp)) + (1-sp)*log((1-sp)./(1-esp));
cost = cost + beta*sum(kbl);

% backwards pass
delta3 = -(data-a3).*sigmoid(s3).*(1-sigmoid(s3));
sp_grad = beta*(-(sp./esp) + (1-sp)./(1-esp));
delta2 = bsxfun(@plus,W2'*delta3,sp_grad).*sigmoid(s2).*(1-sigmoid(s2));
W1grad = m1*delta2*a1' + lambda.* W1;
W2grad = m1*delta3*a2' + lambda.* W2;
b1grad = sum(delta2,2)*m1; 
b2grad = sum(delta3,2)*m1;


% Now we convert the gradients back to a vector format (suitable for minFunc).

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end


function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

