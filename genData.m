function [newData, newLabel] = genData(data, labels, newSize)
    
    % Figure out how much new data needed
    needed = newSize - length(data);
    if(needed < 0)
        newData = data;
        newLabel = labels;
        warning('data size (%d) smaller than new size (%d), returning input data', length(data), newSize);
        return;
    end
    newData = zeros(newSize, size(data, 2));
    newData(1 : length(data), :) = data;
    newLabel = zeros(newSize, 1);
    newLabel(1 : length(data), :) = labels;
    % Try to distribute the new data, such that the count
    % of each label is (almost) equal
    [count,label] = hist(labels, unique(labels));
    distribution = getDist(count, length(label), needed);
    idx = length(data);
    for i = 1:length(label)
        if(distribution(i) == 0)
            continue;
        end
        nd = genDataForLabel(distribution(i), data(labels(:) == (i-1), :));
        newData(idx + 1 : idx + distribution(i), :) = nd;
        newLabel(idx + 1 : idx + distribution(i), :) = repmat(label(i), distribution(i), 1);
        idx = idx + distribution(i);
    end
end

function res = genDataForLabel(newData, data)
    % Initialize the random number generator
    rng(0,'twister');
    
    res = zeros(newData, size(data, 2));
    % We have 6 image manipulators
    % Calculate how many each should generate
    num = floor(newData / 4);
    
    % noise
    for i = 1:num
        % Pick a random row
        rowIdx = ceil(rand * size(data,1));
        % Reshape it to an 28x28 image
        tmp = reshape(data(rowIdx, :), 28, 28);
        % Add noise and shape it back again
        res(i, :) = reshape(imnoise(tmp, 'localvar', tmp), 1, 784);
    end
    % rotate
    for i = 1:num
        % Pick a random row
        rowIdx = ceil(rand * size(data,1));
        % Reshape it to an 28x28 image
        tmp = reshape(data(rowIdx, :), 28, 28);
        % Pick an angle between -40 and 40 degrees
        deg = 80 * rand - 40;
        % Add rotate and shape it back again
        res(i + num, :) = reshape(imrotate(tmp, deg, 'bilinear', 'crop'), 1, 784);
    end
    % transform
    tform = maketform('affine',[1 0 0; .5 1 0; 0 0 1]);
    for i = 1:num
        % Pick a random row
        rowIdx = ceil(rand * size(data,1));
        % Reshape it to an 28x28 image
        tmp = reshape(data(rowIdx, :), 28, 28);
        % Add transformation       
        tmp = imtransform(tmp,tform,'bicubic','udata',[0 1],...
                              'vdata',[0 1],...
                              'size',size(tmp),...
                              'fill',0);
        % Shape it back again
        res(i + num*2, :) = reshape(tmp, 1, 784);
    end
    % scale
    for i = 1:(newData - 3*num)
        % Pick a random row
        rowIdx = ceil(rand * size(data,1));
        % Reshape it to an 28x28 image
        tmp = reshape(data(rowIdx, :), 28, 28);
        % Pick a scale factor
        scale = (1.2-0.8) * rand + 0.8;
        % Add transformation
        imresize(tmp, 'Scale', scale,...
                      'OutputSize', [28 28],...
                      'method', 'bicubic');
   
        % Shape it back again
        res(i + num*3, :) = reshape(tmp, 1, 784);
    end    
end

function res = getDist(count, numLabels, needed)
    even = (max(count)-count);
    if(sum(even) < needed)
        need = needed - sum(even);
        even = even + floor(need/numLabels);
    elseif(sum(even) > needed)
        need = sum(even) - needed;
        even = even - floor(need/numLabels);
        [m, i] = min(even);
        while(m < 0)
           m = ceil(-m/(numLabels - sum(even == 0)));
           even(i) = 0;
           for n = 1:numLabels
              if(n == i || even(n) == 0)
                  continue;
              end
              even(n) = even(n) - m;
           end
           [m, i] = min(even);
        end
    end
    if(sum(even) < needed) 
        [m, i] = min(even);
        even(i) = m + (needed - sum(even));
    elseif(sum(even) > needed)
        surplus = sum(even) - needed;
        [m, i] = max(even);
        even(i) = m - surplus;
    end
    res = even;
end