function [ model ] = bestLin( data, labels )
%BESTRBF Summary of this function goes here
%   Detailed explanation goes here
    % Start threadpool with profile: locale
%     matlabpool('local');
    % Use 10 fold cross validation
    folds = 10;
    % Coarse interval
    C = 0:5:25;
    
    %# grid search, and cross-validation
    coarseAcc = zeros(numel(C),1);
    parfor (i=1:numel(C), 4)
        coarseAcc(i) = svmtrain(labels, data, ...
                        sprintf('-q -c %f -v %d -m 512', 2^C(i), folds));
    end
    %# pair (C,gamma) with best accuracy
    [~,idx] = max(coarseAcc);
    %# plot of paramter selection
    figure('Name','Coarse plot','NumberTitle','On')
    plot(C,coarseAcc)
    hold on
    plot(C(idx), coarseAcc(idx), 'rx')
    text(C(idx), coarseAcc(idx), sprintf('Acc = %.2f %%',coarseAcc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('Accuracy'), title('Cross-Validation Accuracy with coarse grid-search')
%     input('plot done. Continue?');
    bestC = C(idx);
    Cf =(bestC-2):0.5:(bestC+2);
    fineAcc = zeros(numel(Cf),1);
    parfor (i = 1:numel(Cf), 4)
        fineAcc(i) = svmtrain(labels, data, ...
                        sprintf('-q -c %f -v %d -m 512', 2^Cf(i), folds));
    end
%     matlabpool('close');
    [~,idx] = max(fineAcc);
    figure('Name','Fine plot','NumberTitle','On')
    %# plot of paramter selection
    plot(Cf,fineAcc)
    hold on
    plot(Cf(idx), fineAcc(idx), 'rx')
    text(Cf(idx), fineAcc(idx), sprintf('Acc = %.2f %%',fineAcc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('Accuracy'), title('Cross-Validation Accuracy with fine grid-search')
    % Retrain the model without cross validation - but with the best
    % parameters
    model = svmtrain(labels, data, ...
                     sprintf('-q -c %f -m 512', 2^Cf(idx)));
    input('asd');
end

