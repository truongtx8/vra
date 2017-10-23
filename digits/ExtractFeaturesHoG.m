function [featuresData] = ExtractFeaturesHoG(imgData, CellSizeR, CellSizeC)
    imgI1D = imgData(:,1);
    imgI2D = reshape(imgI1D, 28, 28);
    
    [featureVector, hogVisualization] = extractHOGFeatures(imgI2D, 'CellSize',[CellSizeR CellSizeC]);
    
    nSize = length(featureVector);
    nData = size(imgData, 2);
    
    featuresData = zeros(nSize, nData);
    
    for i = 1:nData
        imgI1D = imgData(:, i);
        imgI2D = reshape(imgI1D, 28, 28);
        
        featuresData(:, i) = extractHOGFeatures(imgI2D, 'CellSize',[CellSizeR CellSizeC]);
    end
end