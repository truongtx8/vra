function ResultMatrix = knnRecognitionConfusionMatrix(knnNumNeighbors, knnDistance) 
    ResultMatrix = zeros(10, 2);
    ResultCorrect= 0;

    fprintf ('\nLoading train data...');
    [imgTrainAll, lblTrainAll] = loadData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
    
    fprintf ('\nLoading test data...');
    [imgTestAll, lblTestAll] = loadData ('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
    
    nTestImages  = size(imgTestAll, 2);
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll, 'Distance', knnDistance, 'NumNeighbors', knnNumNeighbors);
    
    fprintf ('\nRecognizing test images...');
    lblPredictTest = predict(Mdl, imgTestAll');
    
    ResultMatrix = confusionmat(lblTestAll, lblPredictTest);
    
    for i = [1:10]
        ResultCorrect = ResultCorrect + ResultMatrix(i, i);
    end
    
    ResultAccurate = 100* ResultCorrect / nTestImages;
    
    fprintf ('\nDistance Metric %s with %d-nearest neighbors classifier; ', knnDistance, knnNumNeighbors);
    fprintf ('Corrected recognition: %d of %d; ', ResultCorrect, nTestImages);
    fprintf ('Accurate rate: %.2f%%\n', ResultAccurate);
end