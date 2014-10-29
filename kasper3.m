% Make SVMlib available
addpath svm/

%trainDat = load('mnistTrain.mat');
%data   = trainDat.images;
%labels= trainDat.labels;

% Load AU database files
trainDat    = load('auTrain.mat');
data   = trainDat.au_train_digits;
labels = trainDat.au_train_labels;

% Generate more data
[data,labels] = genData(data, labels, ceil(length(data)*1.3) );
disp('done generating data');
% Reduce dimensions
data  = dimReduce(data,25*25);
disp('done reducing data');
% Scramble data
idx = randperm(numel(labels))';

% Split into a train and test set
numTrain = round(3*numel(labels)/4);
 
trainData   = data(idx(1:numTrain),:);
trainLabels = labels(idx(1:numTrain)); % Shift Labels to the Range 1-5

testData   = data(idx(numTrain+1:end),:);
testLabels = labels(idx(numTrain+1:end));   % Shift Labels to the Range 1-5
disp('Ready to train...');

% Train
tic;
  folds = 10;
    [C,gamma] = meshgrid(-5:5:15, -15:6:3);

    %# grid search, and cross-validation
    cv_acc = zeros(numel(C),1);
    for i=1:numel(C)
        tic;
        cv_acc(i) = svmtrain(labels, data, ...
                        sprintf('-t 0 -q -c %f -g %f -v %d ', 2^C(i), 2^gamma(i), folds));
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
% model = bestRBF(trainData, trainLabels);
toc;
% Test
[predicted_label, accuracy, dp] = svmpredict(testLabels, testData, model);

% Classification Score
fprintf('Test Accuracy (same data): %f%%\n', accuracy);


trainDat    = load('mnistTest.mat');
data   = trainDat.au_train_digits;
labels = trainDat.au_train_labels;

% Print the misclassified
%diff = find((predicted_label - testLabels) ~= 0);
%misDat = testData(diff, :);
%misLab = testLabels(diff);
%preLab = predicted_label(diff);

%numShow = min(16, numel(misLab));

%imgs = cell(16,1);
%for i=1:16
%    imgs{i} = reshape(misDat(i, :), 28, 28);
%end

%subsize = ceil(sqrt(length(misLab)));

%# show them in subplots
%figure(1)
%for i=1:16
%    subplot(4,4,i);
%    h = imshow(imgs{i}, 'InitialMag',100, 'Border','tight');
%    title(sprintf('Lab.: %d Pre.: %d', misLab(i), preLab(i)))
%end