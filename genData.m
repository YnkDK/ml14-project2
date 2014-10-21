function out = genData(data, labels, newSize)
    out = data;
    % Figure out how much new data needed
    needed = newSize - length(data);
    if(needed < 0)
        out = data;
        warning('data size (%d) smaller than new size (%d), returning input data', length(data), newSize);
        return;
    end
    % Try to distribute the new data, such that the count
    % of each label is (almost) equal
    [count,label] = hist(labels, unique(labels));
    distribution = getDist(count, length(label), needed);

    for i = 1:length(label)
        newData = genDataForLabel(distribution(i), data(labels(:) == i, :))
    end
    disp(label);
    disp(distribution);
    disp(sum(distribution + count));

end

function res = genDataForLabel(newData, data)
    % Example
    tmp = data(1, :);
    tmp = reshape(tmp, 28, 28);
    tmp = imrotate(tmp, 45, 'crop');
    imshow(tmp);
    res = tmp;
    input('click to see next');
    % noise
    % rotate
    % transform
    % scale
    % translate
    % strech
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