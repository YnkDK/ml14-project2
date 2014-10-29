function [dat, lab] = preprocessSvm(data, labels, reduceTo, newSize)
    dat = data;
    lab = labels;
    if(exist('newSize', 'var') && newSize > size(data, 1))
        [dat, lab] = genData(data, labels, newSize);
    end
    
    if(exist('reduceTo', 'var') && reduceTo ~= size(data, 2))
       dat = dimReduce(data, reduceTo);
    end
end