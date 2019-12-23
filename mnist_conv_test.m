
clear all;

% load mnist dataset
images_Test = loadMNISTImages('data/t10k-images-idx3-ubyte');
labels_Test = loadMNISTLabels('data/t10k-labels-idx1-ubyte');

% reshape images to 4 dimensional matrices
images_Test = reshape(images_Test, 28, 28, 1, []);

% load network
load net;

% test
Y_Pred = classify(net, images_Test);
Y_Test = categorical(labels_Test);

accuracy = sum(Y_Pred == Y_Test) / numel(Y_Test)