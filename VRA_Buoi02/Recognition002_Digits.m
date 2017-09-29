function Recognition002_Digits()
    fprintf ('\nLoading train data...');
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
    
    fprintf ('\nLoading test data...');
    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    fprintf ('\nAll data loaded.\n');
    
    nTrainImages = size(imgTrainAll, 2);
    nTrainLabels = size(lblTrainAll, 1);
    
    nTestImages = size(imgTestAll, 2);
    nTestLabels = size(lblTestAll, 1);
    
    nSizeofImage = size(imgTrainAll, 1);
    
    fprintf('\nNumber of train images: %d.\n', nTrainImages);
    fprintf('\nNumber of train labels: %d.', nTrainLabels);
    fprintf('\nNumber of test images: %d.', nTestImages);
    fprintf('\nNumber of test labels: %d.', nTestLabels);
    fprintf('\nSize of an image: %d.\n', nSizeofImage);
end