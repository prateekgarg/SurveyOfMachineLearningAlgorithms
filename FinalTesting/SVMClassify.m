trainingSet = csvread('train3.csv');
trainLabels = trainingSet(:, 94);
trainingSet = trainingSet(:,1:93);
testSet = csvread('test3.csv');
testLabels = testSet(:,94);
testSet = testSet(:,1:93);

t = templateSVM('Standardize', 1, 'KernelFunction', 'gaussian');
Mdl = fitcecoc(trainingSet, trainLabels,'Learners', t);

save models.mat trainingSet trainLabels testSet testLabels Mdl;

pool = parpool;
options = statset('UseParallel', 1);

CVMdl = crossval(Mdl, 'Options', options);

save crossVal.mat CVMdl;

