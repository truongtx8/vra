function CountResult = WrongRecognition(n)
    CountResult = 0;
    
    fprintf ('\nLoading train data...');
    [imgTrainAll, lblTrainAll] = loadData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
    
    fprintf ('\nLoading test data...\n');
    [imgTestAll, lblTestAll] = loadData ('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
    
    nTestImages  = size(imgTestAll, 2);
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll);
    
    for i = 1:nTestImages
        img   = imgTestAll(:, i);
        lblImageTest   = lblTestAll(i);
        lblPredictTest = predict(Mdl, img');
        
        if (lblImageTest == n)
            if (lblImageTest ~= lblPredictTest)
                CountResult = CountResult + 1;
            end
        end
    end
    
    fprintf ('\nTotal number %d wrong recognition: %d\n', n, CountResult);

end