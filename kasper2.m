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
data  = dimReduce(data,22*22);
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
disp('Starting training');

disp('============ LINEAR ===============');
% Train
tic;
  model = bestLin(trainData, trainLabels);
toc;
% disp('done training, predicting');
% Test
%   [predicted_label, accuracy, dp] = svmpredict(testLabels, testData, model);

% Classification Score
%  fprintf('Test Accuracy: %f%%\n', accuracy);

% input('.. click to continue  ..');
disp('============ POLYNOMIAL ===============');
% Train
tic;
model = bestPolyN(trainData, trainLabels);
toc;
disp('done training, predicting');
% Test
[predicted_label, accuracy, dp] = svmpredict(testLabels, testData, model);

% Classification Score
fprintf('Test Accuracy: %f%%\n', accuracy);
input('.. click to continue   ..');
disp('============ RBF ===============');
% Train
tic;
model = bestRBFK(trainData, trainLabels);
toc;
disp('done training, predicting');
% Test
[predicted_label, accuracy, dp] = svmpredict(testLabels, testData, model);

% Classification Score
fprintf('Test Accuracy: %f%%\n', accuracy);
disp('DONE');