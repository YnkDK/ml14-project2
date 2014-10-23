%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; 
% inputSize  = 28 * 28;
inputSize  = 22*22;
numLabels  = 5;
hiddenSize = 200;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the sparse autoencoder  notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
%trainDat = load('mnistTrain.mat');
%mnistData   = trainDat.images;
%mnistLabels = trainDat.labels;

% Load AU database files
trainDat    = load('auTrain.mat');
mnistData   = trainDat.au_train_digits;
mnistLabels = trainDat.au_train_labels;

% [mnistData,mnistLabels] = genData(mnistData, mnistLabels, length(mnistData) );
disp('done generating data');
mnistData  = dimReduce(mnistData,inputSize);
%disp('done reducing data');
% 
% images  =mnistData;
% labels = mnistLabels;
% save('gen.mat','images');
% save('gen.mat','labels','-append');
% Set Unlabeled Set (All Images)

% Simulate a Labeled and Unlabeled set
labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
unlabeledSet = find(mnistLabels >= 5);

numTrain = round(numel(labeledSet)/2);
trainSet = labeledSet(1:numTrain);
testSet  = labeledSet(numTrain+1:end);

unlabeledData = mnistData(unlabeledSet,:);

trainData   = mnistData(trainSet,:);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

testData   = mnistData(testSet,:);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5


addpath svm/
% svmtrain....;


%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

%  Use minFunc to minimize the function for the autoencoder

addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 5;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, unlabeledData'), ...
                              theta, options);


%load('slow_opt_theta.mat')
%It is usually a good idea to save things that takes a long time to compute
%save('slow_opt_theta.mat','opttheta');

%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
% display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData');

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData');

%%======================================================================
%% STEP 4: Train a softmax classifier





model = softmaxTrain(trainFeatures',trainLabels,lambda);

%% STEP 5: Testing 

% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel
pred = softmaxPredict(model,testFeatures');

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
