function Recognition008_Digits_kNN()
    global counter_wrong;
    counter_wrong = 0;
    
    fprintf ('\nLoading train data...');
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
        
    fprintf ('\nLoading test data...\n');
    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    nTrainImages = size(imgTrainAll, 2);
    nTestImages  = size(imgTestAll, 2);
    
    nBins = 256;
    imgTrainAll_hist = zeros(nBins, nTrainImages);
    
    for i = 1:nTrainImages
        imgTrainAll_hist(:,i) = imhist(imgTrainAll(:,i),nBins);
    end
    
    for i = 1:5000
        imgTestAll_hist(:,i) = imhist(imgTestAll(:,i),nBins);
    end
    
    Mdl = fitcknn(imgTrainAll_hist', lblTrainAll, 'NumNeighbors',3);
    
    lblResult = predict(Mdl, imgTestAll_hist');
    nResult = (lblResult == lblTestAll(1:5000));
    nCount = sum(nResult);
    
    fprintf ('\nTotal wrong recognition: %d\n', nCount);

end

function ShowImgWithLabel(n, imgAll, lblAll, type, Mdl)
    global counter_wrong;
    
    fprintf('Processing image %d...\n', n);
    
    img   = imgAll(:, n);
    lblImageTest   = lblAll(n);
    lblPredictTest = predict(Mdl, img');

    strLabelImage = 'Original ';
    strLabelImage = [strLabelImage, num2str(lblImageTest), ' | '];
    strLabelImage = [strLabelImage, num2str(lblPredictTest)]
    strLabelImage = [strLabelImage, ' Predict']
    
    if(lblPredictTest == lblImageTest)
        strResult = 'Correct';
    else
        strResult = 'Wrong';
        fTitle = num2str(n);
        
        counter_wrong = counter_wrong + 1;
        fprintf ('\nWrong recognition: %d\n', counter_wrong);
    end
end