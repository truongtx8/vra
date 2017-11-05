function Result = Confusion_Counter(n)
    global counter_wrong;
    counter_wrong = 0;
    
    fprintf ('\nLoading train data...');
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll);
    
    fprintf ('\nLoading test data...\n');
    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    nTrainImages = size(imgTrainAll, 2);
    nTestImages  = size(imgTestAll, 2);
    
    for i = 1:nTestImages
        nNumber = i;
        ShowImgWithLabel(nNumber, imgTestAll, lblTestAll, 'Test', Mdl);
    end
    
    fprintf ('\nTotal wrong recognition: %d\n', counter_wrong);

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
        counter_wrong = counter_wrong + 1;
    end
end