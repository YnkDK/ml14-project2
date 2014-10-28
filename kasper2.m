% Make SVMlib available
addpath svm/
intputSizeX = 21;
inputSize  = intputSizeX*intputSizeX;


trainDat    = load('auTrain.mat');
data   = trainDat.au_train_digits;
labels = trainDat.au_train_labels;
disp('Generating and reducing data');
[data,labels] = genData(data, labels, ceil(length(data)*1.5) );
data   = dimReduce(data,inputSize);
disp('done generating and reducing data');
% trainDat = load('mnistTrain.mat');
% data   = trainDat.images;
% labels= trainDat.labels;

% Load AU database files
%trainDat    = load('auTrain.mat');
%data   = trainDat.au_train_digits;
%labels = trainDat.au_train_labels;

% Split into a train and test set


% '-t 0 -c 1' for linear
%'-t 1 -g 0.2 -c 45 -d 3'  polynomial
%'-t 1 -g 0.2 -c 45 -d 3'  polynomial
%'-t 1 -c 150 -d 2 -v 10'  polynomial 
% Train
i = 250; %35 is so far the best found. 
while i < 500
tic;
fprintf(strcat('I :', num2str(i),'\r\n')); 

parms =sprintf('-t 1 -c %d  -d 2 -v 10 -q', i);
disp(parms);
model = svmtrain(labels, data,parms );
fprintf('\r\n'); 
toc;
i = i+5;
end
% Test
% [predicted_label, accuracy, dp] = svmpredict(testLabels, testData, model);

% Classification Score
% fprintf('Test Accuracy: %f%%\n', accuracy);

% Print the misclassified
% diff = find((predicted_label - testLabels) ~= 0);
% misDat = testData(diff, :);
% misLab = testLabels(diff);
% preLab = predicted_label(diff);

% imgs = cell(16,1);
% for i=1:16
%     imgs{i} = reshape(misDat(i, :), intputSizeX, intputSizeX);
% end

% subsize = ceil(sqrt(length(misLab)));

%# show them in subplots
% figure(1)
% for i=1:16
%     subplot(4,4,i);
%     h = imshow(imgs{i}, 'InitialMag',100, 'Border','tight');
%     title(sprintf('Lab.: %d Pre.: %d', misLab(i), preLab(i)))
% end