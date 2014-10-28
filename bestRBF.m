function [ model ] = bestRBF( data, labels )
%BESTRBF Summary of this function goes here
%   Detailed explanation goes here
    folds = 10;
    [C,gamma] = meshgrid(-5:5:15, -15:6:3);

    %# grid search, and cross-validation
    cv_acc = zeros(numel(C),1);
    for i=1:numel(C)
        tic;
        cv_acc(i) = svmtrain(labels, data, ...
                        sprintf('-q -c %f -g %f -v %d -m 512', 2^C(i), 2^gamma(i), folds));
        toc;
        fprintf('Progess: %f\n\n', (100*i/numel(C)));
    end

    %# pair (C,gamma) with best accuracy
    [~,idx] = max(cv_acc);

    %# contour plot of paramter selection
    contour(C, gamma, reshape(cv_acc,size(C))), colorbar
    hold on
    plot(C(idx), gamma(idx), 'rx')
    text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')
    % Retrain the model without cross validation - but with the best
    % parameters
    model = svmtrain(labels, data, ...
                     sprintf('-q -c %f -g %f -m 512', 2^C(idx), 2^gamma(idx)));
    fprintf('Best C-value: 2^%d | Best gamma-value: 2^%d\n', C(idx), gamma(idx));
end

