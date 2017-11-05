function ResultMatrix = WrongRecognitionConfusionMatrix() 
    ResultMatrix = zeros(10, 2);
    ResultCorrect= 0;

    fprintf ('\nLoading train data...');
    [imgTrainAll, lblTrainAll] = loadData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
    
    fprintf ('\nLoading test data...\n');
    [imgTestAll, lblTestAll] = loadData ('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
    
    nTestImages  = size(imgTestAll, 2);
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll);
    
    fprintf ('Recognizing test images...\n');
    lblPredictTest = predict(Mdl, imgTestAll');
    
    ResultMatrix = confusionmat(lblTestAll, lblPredictTest);
    for i = [1:10]
        ResultCorrect = ResultCorrect + ResultMatrix(i, i);
    end
    ResultAccurate = 100* ResultCorrect / nTestImages;
    
    fprintf ('\nCorrected recognition: %d of %d\n', ResultCorrect, nTestImages);
    fprintf ('Accurate rate: %.2f%%\n', ResultAccurate);
end