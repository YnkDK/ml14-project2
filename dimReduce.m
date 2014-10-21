function newData = dimReduce( inValues, reduceTo )
%DIMREDUCE reduces dimentionality
%   Gandal's magical wand was here.

% imshow(reshape(inValues(1,:), 28 , 28));  
[coeff, score, variance] = princomp(inValues);
ndim = reduceTo; %new dim size ?
newData = score(:,1:ndim)*coeff(1:ndim,1:ndim)';
% should represent a 5
% imshow(reshape(newData(1,:), sqrt(ndim) , sqrt(ndim)));  
% imshow(reshape(newData(1,:), 24 , 25));  
% imshow(reshape(newData(1,:), 20 ,30));  
% imshow(reshape(newData(1,:), 25 , 24));  
% imshow(reshape(newData(1,:), 24 , 25));  
% imshow(reshape(newData(1,:), 15 , 40));  
% imshow(reshape(newData(1,:), 40 , 15));  
end

