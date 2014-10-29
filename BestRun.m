function [ model ] = BestRun( data, labels, fncText)
%BESTRBF Summary of this function goes here
%   Detailed explanation goes here
    % Use 10 fold cross validation
    folds = 10;
    % Ensure that standard values for SVM are included (C = 1 or C = 2^0
    % and gamma = 1/num_factors or gamma = 2^-lg(num_factors))
    nf = -log2(size(data, 2));
    [C,gamma] = meshgrid(-5:5:15, (nf-12):6:(nf+12));

    %# grid search, and cross-validation
    coarseAcc = zeros(25,1);
    parfor (i=1:numel(C))
        coarseAcc(i) = svmtrain(labels, data, ...
                        sprintf(fncText, 2^C(i), 2^gamma(i), folds));
    end
    %# pair (C,gamma) with best accuracy
    [~,idx] = max(coarseAcc);
    fprintf('--------------------\nBest C-value: 2^%d\nBest gamma-value: 2^%d\n--------------------\n', C(idx), gamma(idx));
    %# contour plot of paramter selection
    contour(C, gamma, reshape(coarseAcc,size(C))), colorbar
    hold on
    plot(C(idx), gamma(idx), 'rx')
    text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',coarseAcc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy with coarse grid-search')
    
    bestC = C(idx);
    bestG = gamma(idx);
    [Cf,gammaf] = meshgrid((bestC-1):0.5:(bestC+1), (bestG-0.75):0.5:(bestG+0.75));
    fineAcc = zeros(20,1);
    parfor (i = 1:numel(Cf), 4)
        fineAcc(i+25) = svmtrain(labels, data, ...
                        sprintf(fncText, 2^Cf(i), 2^gammaf(i), folds));
    end
    [~,idx] = max(fineAcc);
    fprintf('--------------------\nBest C-value: 2^%d\nBest gamma-value: 2^%d\n--------------------\n', Cf(idx), gammaf(idx));
    
    %# contour plot of paramter selection
    contour(Cf, gammaf, reshape(fineAcc,size(Cf))), colorbar
    hold on
    plot(Cf(idx), gammaf(idx), 'rx')
    text(Cf(idx), gammaf(idx), sprintf('Acc = %.2f %%',fineAcc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy with fine grid-search')
    % Retrain the model without cross validation - but with the best
    % parameters
    model = svmtrain(labels, data, ...
                     sprintf(fncText, 2^Cf(idx), 2^gammaf(idx)));
    fprintf('--------------------\nBest C-value: 2^%d\nBest gamma-value: 2^%d\n--------------------\n', Cf(idx), gammaf(idx));
%     fprintf('--------------------\nBest C-value: 2^%d\nBest gamma-value: 2^%d\n--------------------\n', Cf(idx), gammaf(idx));
%     fprintf('--------------------\nBest C-value: 2^%d\nBest gamma-value: 2^%d\n--------------------\n', Cf(idx), gammaf(idx));
end

