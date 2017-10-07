function Recognition005_Digits_kNN()
    fprintf ('\nLoading train data...');
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll);
    
    fprintf ('\nLoading test data...');
    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    nTrainImages = size(imgTrainAll, 2);
    nTestImages  = size(imgTestAll, 2);
    
    nNumber = randi([1 nTestImages]);
    ShowImgWithLabel(nNumber, imgTestAll, lblTestAll, 'Test', Mdl);

end

function ShowImgWithLabel(n, imgAll, lblAll, type, Mdl)
    fprintf('\nProcessing image %d...\n', n);
    fTitle = [type, ' Image ', num2str(n)];
    figure ('Name', fTitle, 'NumberTitle','off');
    img   = imgAll(:, n);
    lblImageTest   = lblAll(n);
    lblPredictTest = predict(Mdl, img');
    img2D = reshape(img, 28, 28); %reshape
    imshow(img2D); % show image
    strLabelImage = 'Original ';
    strLabelImage = [strLabelImage, num2str(lblImageTest), ' | '];
    strLabelImage = [strLabelImage, num2str(lblPredictTest)]
    strLabelImage = [strLabelImage, ' Predict']
    
    
    if(lblPredictTest == lblImageTest)
        strResult = 'Correct';
    else
        strResult = 'Wrong';
    end
    
    title(strResult);
    xlabel(strLabelImage);
end