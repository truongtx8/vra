function ResultMatrix = WrongRecognitionConfusionMatrix(n) 
    ResultMatrix = zeros(10, 2);

    fprintf ('\nLoading train data...');
    [imgTrainAll, lblTrainAll] = loadData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
    
    fprintf ('\nLoading test data...\n');
    [imgTestAll, lblTestAll] = loadData ('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
    
    nTestImages  = size(imgTestAll, 2);
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll);
    %for n = 0:9
        ResultMatrix (n + 1, 1) = n;
        for i = 1:nTestImages
            img   = imgTestAll(:, i);
            lblImageTest   = lblTestAll(i);
            lblPredictTest = predict(Mdl, img');

            if (lblImageTest == n)
                if (lblImageTest ~= lblPredictTest)
                    ResultMatrix(n + 1, 2) = ResultMatrix(n + 1, 2) + 1;
                end
            end
        end
    %end
end