function [ model ] = bestPolyN( data, labels)
%BESTRBF Summary of this function goes here
%   Detailed explanation goes here
    % Use 10 fold cross validation
    fncText = '-t 1 -q ';
    folds = 10;
    % Ensure that standard values for SVM are included (C = 1 or C = 2^0
    % and degree = 2 -> 10 (with 2 increase).
    [C,degree] = meshgrid(-5:5:15, 2:1:10);

    %# grid search, and cross-validation
    coarseAcc = zeros(numel(C),1);
    parfor (i=1:numel(C))
        tic;
        coarseAcc(i) = svmtrain(labels, data, ...
                        sprintf('%s -c %f -d %f -v %d -m 512',fncText, 2^C(i), degree(i), folds));
        toc;
    end
    %# pair (C,degree) with best accuracy
    [~,idx] = max(coarseAcc);
     figure('Name','Coarse plot','NumberTitle','On');
    fprintf('--------------------\nBest C-value: 2^%d\nBest degree: %d\n--------------------\n', C(idx), degree(idx));
    %# contour plot of paramter selection
    contour(C, degree, reshape(coarseAcc,size(C))), colorbar
    hold on
    plot(C(idx), degree(idx), 'rx')
    text(C(idx), degree(idx), sprintf('Acc = %.2f %%',coarseAcc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('degree'), title('Cross-Validation Accuracy with coarse grid-search');
    input('graph 1 completed, continue ? ');
    bestC = C(idx);
    bestG = degree(idx);
    [Cf,degreef] = meshgrid((bestC-1):0.5:(bestC+1), (bestG-1):1:(bestG+1));
    fineAcc = zeros(numel(Cf),1);
    parfor (i = 1:numel(Cf))
        fineAcc(i) = svmtrain(labels, data, ...
                        sprintf('%s -c %f -d %f -v %d -m 512',fncText, 2^Cf(i), degreef(i), folds));
    end
     figure('Name','Fine plot','NumberTitle','On');
    [~,idx] = max(fineAcc);
    fprintf('--------------------\nBest C-value: 2^%d\nBest degree: 2^%d\n--------------------\n', Cf(idx), degreef(idx));
    
    %# contour plot of paramter selection
    contour(Cf, degreef, reshape(fineAcc,size(Cf))), colorbar
    hold on
    plot(Cf(idx), degreef(idx), 'rx')
    text(Cf(idx), degreef(idx), sprintf('Acc = %.2f %%',fineAcc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('degree'), title('Cross-Validation Accuracy with fine grid-search')
    % Retrain the model without cross validation - but with the best
    % parameters
    
    model = svmtrain(labels, data, ...
                     sprintf('%s -c %f -g %f -m 512',fncText, 2^Cf(idx), degreef(idx))); %to solve the cross fold problem
    fprintf('--------------------\nBest C-value: 2^%d\nBest degree: 2^%d\n--------------------\n', Cf(idx), degreef(idx));
end

