function out = softmaxTrain(X,labels,lambda)
% softmax, Trains a softmax function with data X,y and regularization
% lambda. Data is Nxd and labels is n x  and labels must be in range 1:K 
  ym = full(sparse(1:numel(labels),labels, 1));
  theta = randn(size(X,2),size(ym,2));
  addpath('./minFunc/');
  opt = struct;
  opt.Method = 'lbfgs';
  opt.useMex = 0;
  opt.maxIter = 140;
  [x, fvec, info, output] = minFunc(@(t) soft_cost(X,ym,lambda,t),theta(:),opt);
  out = reshape(x,numel(x)/size(ym,2),size(ym,2));
  
end


function [cost,grad] = soft_cost(X,y,lambda,theta)
 theta = reshape(theta,numel(theta)/size(y,2),size(y,2));
 tmp = softmax(X*theta,2);%n x K
 cost = sum(log(sum(tmp.*y,2))) ;
 cost = -cost + 0.5*lambda*sum(sum(theta(1:end,:).*theta(1:end,:)));
 grad = X'*(y-tmp);
 lg = lambda.*theta;
 grad = -grad+lg;
 grad = [grad(:)];
end

function out = softmax(z,dim)
%very simple softmax function calls logsumexp that handles numerical issues
  out = exp(bsxfun(@minus,z,logsumexp(z,dim)));
end

function s = logsumexp(x, dim)
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
end
