function [ pred ] = runBestSVM( images )
%RUNBESTSVM Summary of this function goes here
%   Detailed explanation goes here
    bestModel = load('bestSvm.mat');
    pred =  svmclassify (bestModel,images);
end

