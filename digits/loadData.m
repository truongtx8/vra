function [imgAll, lblAll] = loadData (pathImg, pathLbl)
    imgAll = loadMNISTImages(pathImg);
    lblAll = loadMNISTLabels(pathLbl);
end