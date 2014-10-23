function [newData, newLabel] = genData(data, labels, newSize)
    
    % Figure out how much new data needed
    needed = newSize - length(data);
    if(needed < 0)
        newData = data;
        newLabel = labels;
        warning('data size (%d) smaller than new size (%d), returning input data', length(data), newSize);
        return;
    end
    % Pre-allocate new data
    newData = zeros(newSize, size(data, 2));
    newLabel = zeros(newSize, 1);
    % Insert 'old' data
    newData(1 : length(data), :) = data;
    newLabel(1 : length(data), :) = labels;
    % Try to distribute the new data, such that the count
    % of each label is (almost) equal
    [count,label] = hist(labels, unique(labels));
    % Get the number of each class to be generated
    distribution = getDist(count, length(label), needed);
    % Index for first newly generated data
    idx = length(data);
    for i = 1:length(label)
        % If we should not add any for this class, skip it
        if(distribution(i) == 0)
            continue;
        end
        % Generate new data for class i
        nd = genDataForLabel(distribution(i), data(labels(:) == (i-1), :));
        % Insert new data to result
        newData(idx + 1 : idx + distribution(i), :) = nd;
        % Insert the required number of labels
        newLabel(idx + 1 : idx + distribution(i), :) = repmat(label(i), distribution(i), 1);
        % Update index
        idx = idx + distribution(i);
    end
end

function res = genDataForLabel(newData, data)
    numDim = size(data, 2);
    dim = sqrt(numDim);
    
    % Initialize the random number generator
    rng(0,'twister');
    
    res = zeros(newData, size(data, 2));
    % We have 4 image manipulators
    % Calculate how many each should generate
    num = floor(newData / 4);
    
    % noise
    for i = 1:num
        % Pick a random row
        rowIdx = ceil(rand * size(data,1));
        % Reshape it to an dimxdim image
        tmp = reshape(data(rowIdx, :), dim, dim);
        % Add noise and shape it back again
        noise = imnoise(tmp, 'localvar', tmp);
        res(i, :) = reshape(noise, 1, numDim);
    end
    % rotate
    for i = 1:num
        % Pick a random row
        rowIdx = ceil(rand * size(data,1));
        % Reshape it to an dimxdim image
        tmp = reshape(data(rowIdx, :), dim, dim);
        % Pick an angle between -40 and 40 degrees
        deg = 30 * rand - 15;
        % Add rotate and shape it back again
        res(i + num, :) = reshape(imrotate(tmp, deg, 'bilinear', 'crop'), 1, numDim);
    end
    % transform
    tform = maketform('affine',[1 0 0; .5 1 0; 0 0 1]);
    for i = 1:num
        % Pick a random row
        rowIdx = ceil(rand * size(data,1));
        % Reshape it to an dimxdim image
        tmp = reshape(data(rowIdx, :), dim, dim);
        % Add transformation       
        tmp = imtransform(tmp,tform,'bicubic','udata',[0 1],...
                              'vdata',[0 1],...
                              'size',size(tmp),...
                              'fill',0);
        % Shape it back again
        res(i + num*2, :) = reshape(tmp, 1, numDim);
    end
    % scale
    for i = 1:(newData - 3*num)
        % Pick a random row
        rowIdx = ceil(rand * size(data,1));
        % Reshape it to an dimxdim image
        tmp = reshape(data(rowIdx, :), dim, dim);
        % Pick a scale factor
        scale = (1.1-0.9) * rand + 0.9;
        % Add transformation
        imresize(tmp, 'Scale', scale,...
                      'OutputSize', [dim dim],...
                      'method', 'bicubic');
   
        % Shape it back again
        res(i + num*3, :) = reshape(tmp, 1, numDim);
    end    
end

function res = getDist(count, numLabels, needed)
    % The number of each class to be generated
    % such that they are equally many
    even = (max(count)-count);
    if(sum(even) < needed)
        % We still need more data
        need = needed - sum(even);
        % Evenly distribute the rest among all classes
        even = even + floor(need/numLabels);
    elseif(sum(even) > needed)
        % We cannot make it even distributed
        need = sum(even) - needed;
        % Substract the exess from each class
        even = even - floor(need/numLabels);
        [m, i] = min(even);
        while(m < 0)
           % While we still need to generate negative numbers
           m = ceil(-m/(numLabels - sum(even == 0)));
           % Set the negative value to 0
           even(i) = 0;
           for n = 1:numLabels
              % Do not substract anything if it is already 0
              if(even(n) == 0)
                  continue;
              end
              % Even the fraction of the exess for class n
              even(n) = even(n) - m;
           end
           % Find the new min
           [m, i] = min(even);
        end
    end
    if(sum(even) < needed)
        % We might still need a few more
        [m, i] = min(even);
        even(i) = m + (needed - sum(even));
    elseif(sum(even) > needed)
        % We might have generated to many
        surplus = sum(even) - needed;
        [m, i] = max(even);
        even(i) = m - surplus;
    end
    % Update res
    res = even;
end