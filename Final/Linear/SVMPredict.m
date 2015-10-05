trainingSet = csvread('train3.csv');
trainLabels = trainingSet(:, 94);
trainingSet = trainingSet(:,1:93);
testSet = csvread('test3.csv');
testLabels = testSet(:,94);
testSet = testSet(:,1:93);
options = statset('UseParallel', 1);
%t = templateSVM('Standardize', 1, 'KernelFunction', 'gaussian');
t = templateSVM('Standardize', 1, 'KernelFunction', 'linear');
Mdl = fitcecoc(trainingSet, trainLabels,'Learners', t, 'FitPosterior', 1,'Verbose',2, 'Options', options);

save modelsLinear.mat t options trainingSet trainLabels testSet testLabels Mdl;

%pool = parpool;


%CVMdl = crossval(Mdl, 'Options', options);

%save crossVal.mat CVMdl;

%[label, NegLoss] = predict(Mdl, testSet, 'Options', options);
customBL = @(M,s)nanmedian(1 - bsxfun(@times, M,s),2)/2;
[label, NegLoss] = predict(Mdl, testSet, 'Verbose', 2, 'BinaryLoss', customBL, 'Options', options);
save resultLinear.mat customBL label NegLoss;
