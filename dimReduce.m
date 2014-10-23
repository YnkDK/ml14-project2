function newData = dimReduce( inValues, reduceTo )
%DIMREDUCE reduces dimentionality
%   Gandal's magical wand was here.
[coeff, score, ~] = princomp(inValues);
ndim = reduceTo; %new dim size ?
newData = score(:,1:ndim)*coeff(1:ndim,1:ndim)';
end

