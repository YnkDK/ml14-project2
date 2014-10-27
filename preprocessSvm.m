function [dat, lab] = preprocessSvm(data, labels, reduceTo, newSize)
    if(exist('newSize', 'var') && newSize > size(data, 1))
       [dat, lab] = genData(data, labels, newSize);
    else
       % TODO: Change to best found
       dat = data;
       lab = labels;
    end
    
    if(exist('reduceTo', 'var') && reduceTo ~= size(data, 2))
       dat = dimReduce(data, reduceTo);
    else
       % TODO: Change to best found
       dat = dat;
    end
end