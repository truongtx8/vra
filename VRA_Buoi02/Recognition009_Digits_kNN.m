function Recognition008_Digits_kNN()
    global counter_wrong;
    counter_wrong = 0;
    
    fprintf ('\nLoading train data...');
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
        
    fprintf ('\nLoading test data...\n');
    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    imgI1D = imgTrainAll(:,1);
    imgI2D = reshape(imgI1D,28,28);
    
    featureVector = extractLBPFeatures(imgI2D);
    nSize = length(featureVector);
    
    fprintf (num2str(nSize));
    
    nTrainImages = size(imgTrainAll, 2);
    nTestImages  = size(imgTestAll, 2);
    
    featuresDataTrain = zeros(nSize,nTrainImages);
    featuresDataTest  = zeros(nSize,nTestImages);
    
    for i = 1:nTrainImages
        imgI1D = imgTrainAll(:,i);
        imgI2D = reshape(imgI1D,28,28);
        featuresDataTrain(:,i) = extractLBPFeatures(imgI2D);
    end
    
    for i = 1:nTestImages
        imgI1D = imgTestAll(:,i);
        imgI2D = reshape(imgI1D,28,28);
        featuresDataTest(:,i) = extractLBPFeatures(imgI2D);
    end
    
    Mdl = fitcknn(featuresDataTrain', lblTrainAll, 'NumNeighbors', 3);
    
    lblResult = predict(Mdl, featuresDataTest');
    nResult = (lblResult == lblTestAll);
    nCount = sum(nResult);
    
    fprintf ('\nTotal correct recognition: %d\n', nCount);

end