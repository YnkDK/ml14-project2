function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)
% theta: trained weights from the autoencoder
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% Changed Data Matrix

% We first convert theta to the (W1, W2, b1, b2) matrix/vector format

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

a1 = data; %dim x n, 
s2 = bsxfun(@plus,b1,W1*a1);%W1 h1 x dim. W1a1 = h1 x n
activation = sigmoid(s2);% the activation of each neuron for each training point



end


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
