function demoHoG()
    fprintf ('\nLoading train data...');
    [imgTrainAll, lblTrainAll] = loadData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
    
    img1D = imgTrainAll(:, 1990);
    img2D = reshape(img1D, 28, 28);
    
    subplot (2, 5, 1);
    imshow (img2D);
    
    [featuresVector, visualHoG] = extractHOGFeatures (img2D,'CellSize', [2, 2]);
    plot(visualHoG);
    
    subplot (2, 5, 2);
    [featuresVector, visualHoG] = extractHOGFeatures (img2D,'CellSize', [4, 4]);
    plot(visualHoG);
    
    subplot (2, 5, 3);
    [featuresVector, visualHoG] = extractHOGFeatures (img2D,'CellSize', [8, 8]);
    plot(visualHoG);
    
    subplot (2, 5, 4);
    plot(visualHoG);
    
    subplot (2, 5, 5);
    hist(featuresVector);
    
    img1D = imgTrainAll(:, 1988);
    img2D = reshape(img1D, 28, 28);
    
    imshow (img2D);
    
    subplot (1, 5, 1);
    [featuresVector, visualHoG] = extractHOGFeatures (img2D,'CellSize', [2, 2]);
    plot(visualHoG);
    
    subplot (1, 5, 2);
    [featuresVector, visualHoG] = extractHOGFeatures (img2D,'CellSize', [4, 4]);
    plot(visualHoG);
    
    subplot (1, 5, 3);
    [featuresVector, visualHoG] = extractHOGFeatures (img2D,'CellSize', [8, 8]);
    plot(visualHoG);
    
    subplot (1, 5, 4);
    plot(visualHoG);
    
    subplot (1, 5, 5);
    hist(featuresVector);
end