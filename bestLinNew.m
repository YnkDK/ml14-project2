function [ model ] = bestLinNew( data, labels )
fncText ='-t 0 -q ';
%BESTRBF Summary of this function goes here
%   Detailed explanation goes here
    % Use 10 fold cross validation
    folds = 10;
    % Ensure that standard values for SVM are included (C = 1 or C = 2^0
    C = meshgrid(-5:5:15);
    %# grid search, and cross-validation
    coarseAcc = zeros(numel(C),1);
    parfor (i=1:numel(C))
        tic;
        coarseAcc(i) = svmtrain(labels, data, ...
                        sprintf('%s -c %f -v %d -m 512',fncText, 2^C(i),  folds));
        toc;
    end
    %# pair C with best accuracy
    [~,idx] = max(coarseAcc);
    fprintf('--------------------\nBest C-value: 2^%d\n\n--------------------\n', C(idx));
    %# contour plot of paramter selection
    contour(C, coarseAcc, reshape(coarseAcc,size(C))), colorbar
    hold on
    plot(C(idx), coarseAcc(idx), 'rx')
    text(C(idx), coarseAcc(idx), sprintf('Acc = %.2f %%',coarseAcc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('log_2(\Acc)'), title('Cross-Validation Accuracy with coarse grid-search')
    input('graph 1 completed, continue ? ');
    bestC = C(idx);
    Cf = meshgrid((bestC-1):0.5:(bestC+1));
    fineAcc = zeros(numel(Cf),1);
    parfor (i = 1:numel(Cf))
        fineAcc(i) = svmtrain(labels, data, ...
                        sprintf('%s -c %f -v %d -m 512',fncText, 2^Cf(i), folds));
    end
    [~,idx] = max(fineAcc);
    fprintf('--------------------\nBest C-value: 2^%d\n\n--------------------\n', Cf(idx));
    
    %# contour plot of paramter selection
    contour(Cf, fineAcc, reshape(fineAcc,size(Cf))), colorbar
    hold on
    plot(Cf(idx), fineAcc(idx), 'rx')
    text(Cf(idx), fineAcc(idx), sprintf('Acc = %.2f %%',fineAcc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('log_2(\acc)'), title('Cross-Validation Accuracy with fine grid-search')
    % Retrain the model without cross validation - but with the best
    % parameters
    
    model = svmtrain(labels, data, ...
                     sprintf('%s -c %f -g %f -m 512',fncText, 2^Cf(idx), 2^fineAcc(idx))); %to solve the cross fold problem
    fprintf('--------------------\nBest C-value: 2^%d\n\n--------------------\n', Cf(idx));
end