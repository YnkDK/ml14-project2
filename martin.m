% Make SVMlib available
addpath svm/

trainDat = load('mnistTrain.mat');
data   = trainDat.images;
labels= trainDat.labels;

% Load AU database files
%trainDat    = load('auTrain.mat');
%data   = trainDat.au_train_digits;
%labels = trainDat.au_train_labels;

% Split into a train and test set
numTrain = round(3*numel(labels)/4);
 
trainData   = data(1:numTrain,:);
trainLabels = labels(1:numTrain)'; % Shift Labels to the Range 1-5

testData   = data(numTrain+1:end,:);
testLabels = labels(numTrain+1:end);   % Shift Labels to the Range 1-5
disp('Ready to train...');

% Train
tic;
model = bestRBF(data, labels);
toc;
% Test
[predicted_label, accuracy, dp] = svmpredict(testLabels, testData, model);

% Classification Score
fprintf('Test Accuracy: %f%%\n', accuracy);

% Print the misclassified
diff = find((predicted_label - testLabels) ~= 0);
misDat = testData(diff, :);
misLab = testLabels(diff);
preLab = predicted_label(diff);

imgs = cell(16,1);
for i=1:16
    imgs{i} = reshape(misDat(i, :), 28, 28);
end

subsize = ceil(sqrt(length(misLab)));

%# show them in subplots
figure(1)
for i=1:16
    subplot(4,4,i);
    h = imshow(imgs{i}, 'InitialMag',100, 'Border','tight');
    title(sprintf('Lab.: %d Pre.: %d', misLab(i), preLab(i)))
end