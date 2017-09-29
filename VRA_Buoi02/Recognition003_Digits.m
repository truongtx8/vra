function Recognition003_Digits()
    fprintf ('\nLoading train data...');
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
    
    fprintf ('\nLoading test data...');
    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    fprintf ('\nAll data loaded.\n');
    
    nTrainImages = size(imgTrainAll, 2);
    
    ShowImgWithLabel(1, imgTrainAll, lblTrainAll);
    ShowImgWithLabel(nTrainImages, imgTrainAll, lblTrainAll);
end

function ShowImgWithLabel(n, imgTrainAll, lblTrainAll)
    fprintf('Processing image %d...\n', n);
    figure;
    img   = imgTrainAll(:, n);
    img2D = reshape(img, 28, 28); %reshape
    strLabelImage = num2str(lblTrainAll(n));
    imshow(img2D); % show image
    title([strLabelImage, ' | Image #', num2str(n)]);
end