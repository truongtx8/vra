function BaiTap029
    strFolderDataTrain = fullfile('DataTrain');
    categories = {'0', '1', '2', '3', '4', ...
                  '5', '6', '7', '8', '9'};
    imdsDataTrain = imageDatastore(fullfile(strFolderDataTrain, categories), ...
                    'LabelSource', 'foldernames');
    imdsDataTrain.ReadFcn = @(filename) readAndPreprocessImage(filename);
    
end