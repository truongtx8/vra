function ListImages = Recognition007_Digits_ListImg(ImgType)
    fprintf ('\nLoading train data...');
    lblTrainAll = loadMNISTLabels('./train-labels.idx1-ubyte');
    
    fprintf ('\nLoading test data...');
    lblTestAll = loadMNISTLabels('./t10k-labels.idx1-ubyte');
    
    fprintf ('\nAll data loaded.\n');
    
	nTrainImages = size(lblTrainAll, 1);
    nTestImages  = size(lblTestAll, 1);
    
    if (ImgType == "train")
        ListImages = zeros (nTrainImages, 2);
        for i = 1:nTrainImages
            ListImages(i, 1) = i;
            ListImages(i, 2) = lblTrainAll(i);
        end
        csvwrite("TrainList.csv", ListImages)
    elseif (ImgType == "test")
        ListImages = zeros (nTestImages,2 );
        for i = 1:nTestImages
            ListImages(i, 1) = i;
            ListImages(i, 2) = lblTestAll(i);
        end
        csvwrite("TestList.csv", ListImages)
    end
end