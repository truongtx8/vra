function Recognition004_Digits()
    fprintf ('\nLoading train data...');
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
    
    fprintf ('\nLoading test data...');
    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    fprintf ('\nAll data loaded.\n');
    
    nTrainImages = size(imgTrainAll, 2);
    nTestImages  = size(imgTestAll, 2);
    
    nNumber = randi([1 nTrainImages]);
    ShowImgWithLabel(nNumber, imgTrainAll, lblTrainAll, 'Train');
    
    nNumber = randi([1 nTestImages]);
    ShowImgWithLabel(nNumber, imgTestAll, lblTestAll, 'Test');
end

function ShowImgWithLabel(n, imgAll, lblAll, type)
    fprintf('Processing image %d...\n', n);
    fTitle = [type, ' Image ', num2str(n)];
    figure ('Name', fTitle, 'NumberTitle','off');
    img   = imgAll(:, n);
    img2D = reshape(img, 28, 28); %reshape
    strLabelImage = num2str(lblAll(n));
    imshow(img2D); % show image
    title(strLabelImage);
end