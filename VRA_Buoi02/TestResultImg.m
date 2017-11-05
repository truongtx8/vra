function TestResult = TestResultImg(n)
    fprintf ('\nLoading train data...');
    imgTrainAll = loadMNISTImages('./train-images.idx3-ubyte');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
    
    Mdl = fitcknn(imgTrainAll', lblTrainAll);
    
    fprintf ('\nLoading test data...');
    imgTestAll = loadMNISTImages('./t10k-images.idx3-ubyte');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    nTrainImages = size(imgTrainAll, 2);
    nTestImages  = size(imgTestAll, 2);
    
    TestResult = ShowTestResult(n, imgTestAll, lblTestAll, 'Test', Mdl);
end

function TestResult = ShowTestResult(n, imgAll, lblAll, type, Mdl)
    fprintf('\nProcessing image %d...\n', n);
    img   = imgAll(:, n);
    lblImageTest   = lblAll(n);
    TestResult     = predict(Mdl, img');
end