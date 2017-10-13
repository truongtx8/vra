function ResultMatrix = WrongRecognitionConfusionMatrix() 
    ResultMatrix = zeros(10, 2);

    fprintf ('\nLoading train data...');
    [imgTrainAll, lblTrainAll] = loadData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
    
    fprintf ('\nLoading test data...\n');
    [imgTestAll, lblTestAll] = loadData ('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
    
    %nTestImages  = size(imgTestAll, 2);
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll);
    
    fprintf ('\nRecognizing test images...\n');
    lblPredictTest = predict(Mdl, imgTestAll');
    
    ResultMatrix = confusionmat(lblTestAll, lblPredictTest);
end