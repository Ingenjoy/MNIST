
clear all;

% load mnist dataset
images_Train = loadMNISTImages('train-images-idx3-ubyte');
labels_Train = loadMNISTLabels('train-labels-idx1-ubyte');
images_Test = loadMNISTImages('t10k-images-idx3-ubyte');
labels_Test = loadMNISTLabels('t10k-labels-idx1-ubyte 2');

% reshape images to 4 dimensional matrices
images_Train = reshape(images_Train, 28, 28, 1, []);
images_Test = reshape(images_Test, 28, 28, 1, []);

% split data into training and evaluation sets
images_Val = images_Train(:, :, :, 1:10000);
labels_Val = labels_Train(1:10000, :);
images_Train = images_Train(:, :, :, 10001:end);
labels_Train = labels_Train(10001:end, :);

% define network architecture
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% traninig option
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', {images_Val, categorical(labels_Val)}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train
net = trainNetwork(images_Train, categorical(labels_Train), ...
    layers, options);

% test
% TODO: should use test dataset for the final performance check!
YPred = classify(net, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)