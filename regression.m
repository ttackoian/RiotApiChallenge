clear all; close all; clc;

%% Load datasets -- path custom to each machine!
load('../../Desktop/riot-contest/riotData.mat');

%% Pre-processing
trainingData = double(trainingData);
trainingLabels = double(trainingLabels');

numTraining = size(trainingData, 1);
numFeatures = size(trainingData, 2) + 1;

offset = ones(numTraining, 1);
trainingData = [offset trainingData];

trainingData2 = trainingData;
for i = 1 : numTraining
    for j = 1 : numFeatures
        trainingData2(i, j) = log(double(trainingData2(i, j)) + 0.1);
    end
end

%% Parameter setup
lambda = 0.5;
eta = 0.00001;
numIterations = 3000;

%% #1, (ii)
beta2 = zeros(numFeatures, 1);
mu2 = zeros(numTraining, 1);
trainingLoss2 = zeros(numIterations, 1);

for i = 1:numIterations
    % Improve beta
    mu2 = (1./(1 + exp(-1 * (beta2' * trainingData2'))))';
    beta2 = beta2 - eta*(((2*lambda).*beta2) - (trainingData2' * (double(trainingLabels) - mu2)));
    
    % Record training loss
    currTrainingLoss = lambda * (norm(beta2, 2)).^2;
    lossSum = sum(double(trainingLabels) .* log(mu2) + (1 - double(trainingLabels)) .* log(1 - mu2));
    trainingLoss2(i) = currTrainingLoss - lossSum;
end
